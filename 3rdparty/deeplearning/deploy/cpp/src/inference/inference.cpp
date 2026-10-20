// 模型推理层实现：TRT engine 编排 + CPU 自定义算子。
// 镜像 Python 侧 deploy/inference.py（Inferencer）+ 部分 C++ ops。

#include "inference/inference.hpp"

#include "ops/ops.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace dsb {

namespace {

using Clock = std::chrono::steady_clock;
double ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

std::string find_engine(const std::string &dir, const std::string &stem) {
  for (const char *ext : {".engine", ".trt", ".plan"}) {
    std::string p = dir + "/" + stem + ext;
    if (std::filesystem::exists(p)) return p;
  }
  throw std::runtime_error("engine not found: " + dir + "/" + stem +
                           "{.engine,.trt,.plan}");
}

std::vector<float> read_blob(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open weights: " + path);
  std::vector<char> buf((std::istreambuf_iterator<char>(f)), {});
  return std::vector<float>(
      reinterpret_cast<float *>(buf.data()),
      reinterpret_cast<float *>(buf.data() + buf.size()));
}

// depthnet logits -> 深度概率折入特征：x (B,N,D,fH,fW,C)。
// 与 torch softmax(dim=1) 逐位一致（沿 D 通道 softmax 后乘特征）。
void fold_depth_features(const Config &cfg, const Tensor &depthnet,
                         Tensor &x) {
  const int D = cfg.depth_bins(), C = cfg.vt_out_c;
  const int fH = cfg.feat_h, fW = cfg.feat_w;
  const int N = cfg.N;
  const int area = fH * fW;
  const float *dn = depthnet.data.data();

  std::vector<float> depth_p(D * area * N);
  std::vector<float> e(D);  // 复用 softmax 暂存，避免每次循环堆分配
  for (int n = 0; n < N; ++n) {
    const int n_off = n * (D + C) * area;
    for (int hw = 0; hw < area; ++hw) {
      const float *logits = dn + n_off + hw;
      float mx = -1e30f;
      for (int d = 0; d < D; ++d) mx = std::max(mx, logits[d * area]);
      double sum = 0;
      for (int d = 0; d < D; ++d) {
        e[d] = std::exp(logits[d * area] - mx);
        sum += e[d];
      }
      float *dp = depth_p.data() + n * D * area + hw;
      const float inv = (float)(1.0 / sum);
      for (int d = 0; d < D; ++d) dp[d * area] = e[d] * inv;
    }
  }

  const float *ft = dn + D * area;
  for (int n = 0; n < N; ++n) {
    const float *fc = ft + n * C * area;
    float *xbase = x.data.data() + (int64_t)n * D * area * C;
    for (int hw = 0; hw < area; ++hw) {
      const float *feat_c = fc + hw;
      for (int d = 0; d < D; ++d) {
        const float dp = depth_p[n * D * area + d * area + hw];
        float *xc = xbase + ((int64_t)d * area + hw) * C;
        for (int c = 0; c < C; ++c) xc[c] = dp * feat_c[c * area];
      }
    }
  }
}

}  // namespace

Inferencer::Inferencer(const Config &cfg) : cfg_(cfg) {}

void Inferencer::load_engines(const std::string &engine_dir) {
  backbone_.reset(new TrtEngine(find_engine(engine_dir, "camera.backbone")));
  vtransform_.reset(new TrtEngine(find_engine(engine_dir, "camera.vtransform")));
  fuser_.reset(new TrtEngine(find_engine(engine_dir, "fuser")));
  head_.reset(new TrtEngine(find_engine(engine_dir, "head.bbox")));
}

void Inferencer::load_weights(const std::string &input_dir) {
  downsample_blob_ = read_blob(input_dir + "/weights/downsample.bin");
  sparse_blob_ = read_blob(input_dir + "/weights/sparse_encoder.bin");
}

Tensor Inferencer::camera_backbone(const Tensor &img) const {
  auto out = backbone_->run({{"img", img}});
  return out.at("feats");  // (1,6,256,32,88)
}

Tensor Inferencer::camera_vtransform(const Tensor &feats,
                                     const Tensor &depth) const {
  auto out = vtransform_->run({{"feats", feats}, {"depth", depth}});
  return out.at("depthnet");  // (1,6,198,32,88)
}

Tensor Inferencer::camera_bev(const Tensor &depthnet,
                              const Preprocessor::Inputs &in) const {
  const int D = cfg_.depth_bins(), C = cfg_.vt_out_c;
  const int fH = cfg_.feat_h, fW = cfg_.feat_w;
  Clock::time_point t0 = Clock::now();
  auto stamp = [&](const char *tag) {
    Clock::time_point t1 = Clock::now();
    std::printf("[dsb] [time]   %-20s %8.2f ms\n", tag, ms(t0, t1));
    t0 = t1;
  };
  std::printf("[dsb] geometry + bev_pool ...\n");
  Tensor geom = compute_geometry(cfg_, in.camera2lidar, in.intrins, in.img_aug,
                                 in.lidar_aug);  // (6,D,fH,fW,3)
  stamp("compute_geometry");
  Tensor x({cfg_.B, cfg_.N, D, fH, fW, C});
  fold_depth_features(cfg_, depthnet, x);
  stamp("fold_depth_features");
  Tensor bev = bev_pool(cfg_, geom, x);  // (C, 360, 360)
  stamp("bev_pool");
  bev.reshape({cfg_.B, C, bev.shape[1], bev.shape[2]});
  std::printf("[dsb]   bev_pool out %ldx%ldx%ld\n", (long)bev.shape[1],
              (long)bev.shape[2], (long)bev.shape[3]);
  Tensor cam_bev = downsample_conv(cfg_, bev, downsample_blob_);  // (1,80,180,180)
  stamp("downsample_conv");
  std::printf("[dsb]   cam_bev %ldx%ldx%ldx%ld\n", (long)cam_bev.shape[0],
              (long)cam_bev.shape[1], (long)cam_bev.shape[2],
              (long)cam_bev.shape[3]);
  return cam_bev;
}

Tensor Inferencer::lidar_branch(const Preprocessor::Inputs &in) const {
  std::printf("[dsb] voxelize + sparse encoder ...\n");
  Clock::time_point t0 = Clock::now();
  Tensor vfeats, vcoords, vsizes;
  Preprocessor(cfg_).voxelize(in, vfeats, vcoords, vsizes);
  Clock::time_point t1 = Clock::now();
  std::printf("[dsb] [time]   %-20s %8.2f ms\n", "voxelize", ms(t0, t1));
  Tensor lidar_bev = sparse_encoder_forward(cfg_, vfeats, vcoords, vsizes,
                                            sparse_blob_);
  Clock::time_point t2 = Clock::now();
  std::printf("[dsb] [time]   %-20s %8.2f ms\n", "sparse_encoder", ms(t1, t2));
  std::printf("[dsb]   lidar_bev %ldx%ldx%ld (voxels %ld)\n",
              (long)lidar_bev.shape[0], (long)lidar_bev.shape[1],
              (long)lidar_bev.shape[2], (long)vcoords.shape[0]);
  return lidar_bev;
}

Tensor Inferencer::fuser(const Tensor &cam_bev,
                         const Tensor &lidar_bev) const {
  std::printf("[dsb] fuser ...\n");
  auto out = fuser_->run({{"cam_bev", cam_bev}, {"lidar_bev", lidar_bev}});
  return out.at("decoder_feat");  // (1,512,180,180)
}

std::vector<Tensor> Inferencer::head(const Tensor &decoder_feat) const {
  std::printf("[dsb] head.bbox ...\n");
  auto out = head_->run({{"decoder_feat", decoder_feat}});
  std::vector<Tensor> preds;
  for (const char *k :
       {"dense_heatmap", "heatmap", "query_heatmap_score", "center",
        "height", "dim", "rot", "vel", "query_labels"})
    preds.push_back(out.at(k));
  return preds;
}

}  // namespace dsb
