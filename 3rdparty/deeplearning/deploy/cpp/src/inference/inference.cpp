// 模型推理层实现：TRT engine 编排 + CUDA/CPU 自定义算子。
// 镜像 Python 侧 deploy/inference.py（Inferencer）+ 部分 C++ ops。
// camera 链算子（geometry / fold+bev_pool / downsample）走 CUDA kernel
// （src/ops/cuda_kernels.cu），与 TRT 引擎共用 stream；lidar 链暂为 CPU。

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

// ---- CPU 参考实现（对拍用；推理路径已走 CUDA kernel cuda_fold_bevpool）
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

Inferencer::Inferencer(const Config &cfg) : cfg_(cfg) {
  // CUDA 自定义算子资源（与 TRT 引擎共用 stream_）
  cudaStreamCreate(&stream_);
  const int D = cfg_.depth_bins(), C = cfg_.vt_out_c;
  const int area = cfg_.feat_h * cfg_.feat_w;
  const size_t geom_n = (size_t)cfg_.N * D * area * 3;
  const size_t depthnet_n = (size_t)cfg_.N * (D + C) * area;
  const size_t bev_n = (size_t)C * cfg_.nx[0] * cfg_.nx[1];
  cudaMalloc(&geom_buf_, geom_n * sizeof(float));
  cudaMalloc(&depthnet_buf_, depthnet_n * sizeof(float));
  cudaMalloc(&bev_buf_, bev_n * sizeof(float));
  cudaMalloc(&ds_a_, bev_n * sizeof(float));
  cudaMalloc(&ds_b_, bev_n * sizeof(float));
  cudaMalloc(&ds_out_, bev_n * sizeof(float));
}

Inferencer::~Inferencer() {
  cudaFree(geom_buf_);
  cudaFree(depthnet_buf_);
  cudaFree(bev_buf_);
  cudaFree(ds_a_);
  cudaFree(ds_b_);
  cudaFree(ds_out_);
  cudaFree(blob_ds_);
  cudaFree(layers_dev_);
  cudaStreamDestroy(stream_);
}

void Inferencer::load_engines(const std::string &engine_dir) {
  backbone_.reset(new TrtEngine(find_engine(engine_dir, "camera.backbone")));
  vtransform_.reset(new TrtEngine(find_engine(engine_dir, "camera.vtransform")));
  fuser_.reset(new TrtEngine(find_engine(engine_dir, "fuser")));
  head_.reset(new TrtEngine(find_engine(engine_dir, "head.bbox")));
}

void Inferencer::load_weights(const std::string &input_dir) {
  downsample_blob_ = read_blob(input_dir + "/weights/downsample.bin");
  sparse_blob_ = read_blob(input_dir + "/weights/sparse_encoder.bin");
  // downsample 权重 + 层表 -> 设备端（CUDA kernel 读取）
  cudaMalloc(&blob_ds_, downsample_blob_.size() * sizeof(float));
  cudaMemcpy(blob_ds_, downsample_blob_.data(),
             downsample_blob_.size() * sizeof(float), cudaMemcpyHostToDevice);
  const int n = (int)cfg_.down_layers.size();
  if (n > 0) {
    cudaMalloc(&layers_dev_, n * sizeof(CudaDownsampleLayer));
    std::vector<CudaDownsampleLayer> layers(n);
    for (int i = 0; i < n; ++i) {
      const auto &ly = cfg_.down_layers[i];
      layers[i] = {ly.in_c, ly.out_c, ly.k, ly.stride, ly.pad,
                   (long long)ly.w_off, (long long)ly.w_len,
                   (long long)ly.b_len, ly.bn_eps};
    }
    cudaMemcpy(layers_dev_, layers.data(), n * sizeof(CudaDownsampleLayer),
               cudaMemcpyHostToDevice);
  }
}

Tensor Inferencer::camera_backbone(const Tensor &img) const {
  auto out = backbone_->run({{"img", img}}, stream_);
  cudaStreamSynchronize(stream_);
  return out.at("feats");  // (1,6,256,32,88)
}

Tensor Inferencer::camera_vtransform(const Tensor &feats,
                                     const Tensor &depth) const {
  auto out = vtransform_->run({{"feats", feats}, {"depth", depth}}, stream_);
  cudaStreamSynchronize(stream_);
  return out.at("depthnet");  // (1,6,198,32,88)
}

Tensor Inferencer::camera_bev(const Tensor &depthnet,
                              const Preprocessor::Inputs &in) const {
  const int D = cfg_.depth_bins(), C = cfg_.vt_out_c;
  const int fH = cfg_.feat_h, fW = cfg_.feat_w;
  const int area = fH * fW;
  const int N = cfg_.N;
  Clock::time_point t0 = Clock::now();
  auto stamp = [&](const char *tag) {
    Clock::time_point t1 = Clock::now();
    std::printf("[dsb] [time]   %-20s %8.2f ms\n", tag, ms(t0, t1));
    t0 = t1;
  };
  std::printf("[dsb] geometry + bev_pool (cuda) ...\n");

  // ---- 组装每相机合成矩阵（post_inv / post_inv*t / c2l*intr_inv / c2l*t）
  CudaGeometryParams gp{};
  gp.D = D; gp.fH = fH; gp.fW = fW; gp.N = N;
  gp.image_w = (float)cfg_.image_w; gp.image_h = (float)cfg_.image_h;
  gp.dbound0 = cfg_.dbound[0]; gp.dbound2 = cfg_.dbound[2];
  gp.bx = cfg_.xbound[0]; gp.by = cfg_.ybound[0]; gp.bz = cfg_.zbound[0];
  gp.dx_ = cfg_.xbound[2]; gp.dy_ = cfg_.ybound[2]; gp.dz_ = cfg_.zbound[2];
  gp.nx0 = cfg_.nx[0]; gp.nx1 = cfg_.nx[1]; gp.nx2 = cfg_.nx[2];
  for (int c = 0; c < N; ++c) {
    const Mat3 post_inv = Mat3::inverse(in.img_aug[c].rot3());
    float post_t[3];
    in.img_aug[c].col3(3, post_t);
    float pit[3];
    post_inv.matvec(post_t, pit);
    const Mat3 intr_inv = Mat3::inverse(in.intrins[c].rot3());
    const Mat3 c2l = in.camera2lidar[c].rot3();
    float c2l_t[3];
    in.camera2lidar[c].col3(3, c2l_t);
    const Mat3 combine = Mat3::mul(c2l, intr_inv);
    float *pm = gp.per_cam + c * 24;
    for (int i = 0; i < 9; ++i) {
      pm[i] = post_inv.m[i];
      pm[12 + i] = combine.m[i];
    }
    pm[9] = pit[0]; pm[10] = pit[1]; pm[11] = pit[2];
    pm[21] = c2l_t[0]; pm[22] = c2l_t[1]; pm[23] = c2l_t[2];
  }
  const Mat3 la_rot = in.lidar_aug.rot3();
  float la_t[3];
  in.lidar_aug.col3(3, la_t);
  for (int i = 0; i < 9; ++i) gp.la[i] = la_rot.m[i];
  gp.la[9] = la_t[0]; gp.la[10] = la_t[1]; gp.la[11] = la_t[2];

  // ---- 流上串行：memset -> depthnet 上设备 -> geometry -> fold+bev_pool
  //      -> downsample -> 同步拷回 host
  const size_t bev_n = (size_t)C * cfg_.nx[0] * cfg_.nx[1];
  cudaMemsetAsync(bev_buf_, 0, bev_n * sizeof(float), stream_);
  cudaMemcpyAsync(depthnet_buf_, depthnet.data.data(),
                  depthnet.data.size() * sizeof(float), cudaMemcpyHostToDevice,
                  stream_);
  cuda_compute_geometry(gp, geom_buf_, stream_);
  stamp("compute_geometry");
  CudaFoldPoolParams fp{};
  fp.N = N; fp.D = D; fp.C = C; fp.area = area;
  fp.nx0 = gp.nx0; fp.nx1 = gp.nx1; fp.nx2 = gp.nx2;
  fp.bx = gp.bx; fp.by = gp.by; fp.bz = gp.bz;
  fp.dx_ = gp.dx_; fp.dy_ = gp.dy_; fp.dz_ = gp.dz_;
  cuda_fold_bevpool(fp, depthnet_buf_, geom_buf_, bev_buf_, stream_);
  stamp("fold + bev_pool");
  // bev (C,H,W) 与 4D (1,C,H,W) 连续等价
  cuda_downsample(layers_dev_, (int)cfg_.down_layers.size(), blob_ds_,
                  bev_buf_, ds_a_, ds_b_, ds_out_, C, cfg_.nx[0], cfg_.nx[1],
                  stream_);
  stamp("downsample_conv");
  cudaStreamSynchronize(stream_);

  // 输出形状沿层链推导
  int Ho = cfg_.nx[0], Wo = cfg_.nx[1], Cout = C;
  for (const auto &ly : cfg_.down_layers) {
    Ho = (Ho - ly.k + 2 * ly.pad) / ly.stride + 1;
    Wo = (Wo - ly.k + 2 * ly.pad) / ly.stride + 1;
    Cout = ly.out_c;
  }
  Tensor cam_bev({cfg_.B, Cout, Ho, Wo});
  cudaMemcpy(cam_bev.data.data(), ds_out_, cam_bev.data.size() * sizeof(float),
             cudaMemcpyDeviceToHost);
  stamp("host sync + copy");
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
  auto out = fuser_->run({{"cam_bev", cam_bev}, {"lidar_bev", lidar_bev}},
                         stream_);
  cudaStreamSynchronize(stream_);
  return out.at("decoder_feat");  // (1,512,180,180)
}

std::vector<Tensor> Inferencer::head(const Tensor &decoder_feat) const {
  std::printf("[dsb] head.bbox ...\n");
  auto out = head_->run({{"decoder_feat", decoder_feat}}, stream_);
  cudaStreamSynchronize(stream_);
  std::vector<Tensor> preds;
  for (const char *k :
       {"dense_heatmap", "heatmap", "query_heatmap_score", "center",
        "height", "dim", "rot", "vel", "query_labels"})
    preds.push_back(out.at(k));
  return preds;
}

}  // namespace dsb
