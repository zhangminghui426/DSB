// 模型推理层实现：engine 编排 + CPU 算子（几何/bev_pool/降采样/稀疏编码）。
#include "dsb_infer/inference.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>

#include "dsb_infer/ops.hpp"

namespace dsb {

namespace {

void read_bytes(const std::string &path, void *dst, size_t nbytes) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  f.read(reinterpret_cast<char *>(dst), static_cast<std::streamsize>(nbytes));
}

// softmax(depth) * feats 折叠：depthnet (1,6,D+C,32,88) 的 D 维 softmax 后
// 与 feats (1,6,C,32,88) 加权求和 -> (1,6,C,32,88)，供 bev_pool 使用。
Tensor fold_depth_features(const Tensor &depthnet, const Config &cfg) {
  const int N = cfg.num_camera;
  const int D = cfg.num_depth;
  const int C = cfg.camera_embed_dim;
  const int fH = cfg.feat_h, fW = cfg.feat_w;
  const float *x = depthnet.data();
  Tensor out(std::vector<int>{1, N, C, fH, fW}, Tensor::kFloat);
  float *o = out.data();
  std::fill(o, o + out.numel(), 0.f);
  for (int n = 0; n < N; ++n) {
    for (int h = 0; h < fH; ++h) {
      for (int w = 0; w < fW; ++w) {
        // softmax over D
        const float *dlogit = x + ((n * (D + C) + 0) * fH + h) * fW + w;
        float maxv = *dlogit;
        for (int d = 1; d < D; ++d)
          maxv = std::max(maxv, dlogit[d * fH * fW]);
        std::vector<float> prob(D);
        float sum = 0.f;
        for (int d = 0; d < D; ++d)
          prob[d] = std::exp(dlogit[d * fH * fW] - maxv), sum += prob[d];
        for (int d = 0; d < D; ++d) prob[d] /= sum;
        // 加权 feats 通道
        float *oc = o + ((n * C + 0) * fH + h) * fW + w;
        const float *fc = x + ((n * (D + C) + D) * fH + h) * fW + w;
        for (int c = 0; c < C; ++c) {
          float acc = 0.f;
          for (int d = 0; d < D; ++d) acc += prob[d] * fc[c * fH * fW + d * fH * fW];
          oc[c * fH * fW] = acc;
        }
      }
    }
  }
  return out;
}

}  // namespace

Inferencer::Inferencer(const Config &cfg) : cfg_(cfg) {}

void Inferencer::load_engines(const std::string &engine_dir) {
  backbone_ = std::make_unique<TrtEngine>(engine_dir + "/camera.backbone.engine");
  vtransform_ = std::make_unique<TrtEngine>(engine_dir + "/camera.vtransform.engine");
  fuser_ = std::make_unique<TrtEngine>(engine_dir + "/fuser.engine");
  head_ = std::make_unique<TrtEngine>(engine_dir + "/head.bbox.engine");
}

void Inferencer::load_weights(const std::string &input_dir) {
  const auto wdir = input_dir + "/weights";
  std::ifstream f(wdir + "/downsample.bin", std::ios::binary | std::ios::ate);
  const auto sz = f.tellg();
  downsample_blob_.resize(static_cast<size_t>(sz) / sizeof(float));
  f.seekg(0);
  f.read(reinterpret_cast<char *>(downsample_blob_.data()), sz);

  std::ifstream g(wdir + "/sparse_encoder.bin", std::ios::binary | std::ios::ate);
  const auto sz2 = g.tellg();
  sparse_blob_.resize(static_cast<size_t>(sz2) / sizeof(float));
  g.seekg(0);
  g.read(reinterpret_cast<char *>(sparse_blob_.data()), sz2);
}

Tensor Inferencer::camera_backbone(const Tensor &img) const {
  return backbone_->run({{img}})["feats"];
}

Tensor Inferencer::camera_vtransform(const Tensor &feats, const Tensor &depth) const {
  return vtransform_->run({{feats, depth}})["depthnet"];
}

Tensor Inferencer::camera_bev(const Tensor &depthnet,
                              const Preprocessor::Inputs &in) const {
  const Tensor feats = fold_depth_features(depthnet, cfg_);
  const Tensor geom = compute_geometry(cfg_, in.camera2lidar, in.intrins,
                                       in.img_aug, in.lidar_aug);
  const Tensor pooled = bev_pool(cfg_, geom, feats);
  return downsample_conv(cfg_, pooled, downsample_blob_);
}

Tensor Inferencer::lidar_branch(const Preprocessor::Inputs &in) const {
  Tensor feats, coords, sizes;
  Preprocessor(cfg_).voxelize(in, feats, coords, sizes);
  return sparse_encoder_forward(cfg_, feats, coords, sizes, sparse_blob_);
}

Tensor Inferencer::fuser(const Tensor &cam_bev, const Tensor &lidar_bev) const {
  return fuser_->run({{cam_bev, lidar_bev}})["decoder_feat"];
}

std::vector<Tensor> Inferencer::head(const Tensor &decoder_feat) const {
  return head_->run({{decoder_feat}})["outputs"];
}

}  // namespace dsb
