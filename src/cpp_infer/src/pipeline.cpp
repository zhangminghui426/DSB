// InferencePipeline implementation. Pipeline (batch=1), mirrors
// deploy/export.py + deploy/inferencer.py + models/* at det config:
//   img -> [TRT camera.backbone] -> feats
//       -> depth (project_depth, CPU) -> [TRT camera.vtransform] -> depthnet
//       -> softmax depth × feats -> geometry + bev_pool (CPU)
//       -> (80,360,360) -> downsample (CPU) -> cam_bev (1,80,180,180)
//   points -> voxelize + sparse encoder (CPU) -> lidar_bev (1,256,180,180)
//   [TRT fuser] -> decoder_feat (1,512,180,180)
//   [TRT head.bbox] -> 8 preds -> decode (CPU) -> boxes / scores / labels

#include "dsb_infer/pipeline.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <utility>

namespace dsb {

namespace {

std::string find_engine(const std::string &dir, const std::string &stem) {
  for (const char *ext : {".engine", ".trt", ".plan"}) {
    std::string p = dir + "/" + stem + ext;
    if (std::filesystem::exists(p)) return p;
  }
  throw std::runtime_error("engine not found: " + dir + "/" + stem +
                           "{.engine,.trt,.plan}");
}

std::vector<Mat4> read_mat4_block(const std::vector<float> &all, int idx) {
  std::vector<Mat4> v;
  for (int k = 0; k < idx; ++k) {
    Mat4 m{};
    std::memcpy(m.m, all.data() + k * 16, 16 * sizeof(float));
    v.push_back(m);
  }
  return v;
}

std::vector<float> read_blob(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open weights: " + path);
  std::vector<char> buf((std::istreambuf_iterator<char>(f)), {});
  return std::vector<float>(
      reinterpret_cast<float *>(buf.data()),
      reinterpret_cast<float *>(buf.data() + buf.size()));
}

double rel_err(const Tensor &a, const Tensor &b) {
  double sum = 0, mx = 0;
  for (size_t i = 0; i < a.data.size(); ++i) {
    double denom = std::abs((double)b.data[i]) + 1e-6;
    double e = std::abs((double)a.data[i] - b.data[i]) / denom;
    sum += e;
    mx = std::max(mx, e);
  }
  return mx;  // max rel err
}

}  // namespace

InferencePipeline::InferencePipeline(const Config &cfg) : cfg_(cfg) {}

InferencePipeline::Inputs InferencePipeline::load_inputs(
    const std::string &input_dir) const {
  Inputs in;
  in.img = Tensor({cfg_.B, cfg_.N, 3, cfg_.image_h, cfg_.image_w});
  in.img.data = read_bin(input_dir + "/inputs/img.bin", in.img.data.size());

  std::ifstream f(input_dir + "/inputs/points.bin", std::ios::binary);
  if (!f) throw std::runtime_error("cannot open inputs/points.bin");
  int32_t n = 0;
  f.read(reinterpret_cast<char *>(&n), sizeof(n));
  in.points = Tensor({n, 5});
  f.read(reinterpret_cast<char *>(in.points.data.data()),
         (std::streamsize)(n * 5 * sizeof(float)));
  std::printf("[dsb] points: %d\n", n);

  std::vector<float> mats = read_bin(input_dir + "/inputs/matrices.bin",
                                     19 * 16);
  in.camera2lidar = read_mat4_block(mats, 6);  // idx 0..5
  in.intrins = read_mat4_block(
      std::vector<float>(mats.begin() + 6 * 16, mats.end()), 6);
  in.img_aug = read_mat4_block(
      std::vector<float>(mats.begin() + 12 * 16, mats.end()), 6);
  std::memcpy(in.lidar_aug.m, mats.data() + 18 * 16, 16 * sizeof(float));
  return in;
}

void InferencePipeline::load_engines(const std::string &engine_dir) {
  backbone_.reset(new TrtEngine(find_engine(engine_dir, "camera.backbone")));
  vtransform_.reset(new TrtEngine(find_engine(engine_dir, "camera.vtransform")));
  fuser_.reset(new TrtEngine(find_engine(engine_dir, "fuser")));
  head_.reset(new TrtEngine(find_engine(engine_dir, "head.bbox")));
}

void InferencePipeline::load_weights(const std::string &input_dir) {
  downsample_blob_ = read_blob(input_dir + "/weights/downsample.bin");
  sparse_blob_ = read_blob(input_dir + "/weights/sparse_encoder.bin");
}

InferencePipeline::Outputs InferencePipeline::infer(const Inputs &in) {
  const int D = cfg_.depth_bins(), C = cfg_.vt_out_c;
  const int fH = cfg_.feat_h, fW = cfg_.feat_w;

  // ---- camera branch ---------------------------------------------------
  std::printf("[dsb] backbone ...\n");
  auto out = backbone_->run({{"img", in.img}});
  Tensor feats = out.at("feats");  // (1,6,256,32,88)
  std::printf("[dsb]   feats %ldx%ldx%ldx%ldx%ld\n", (long)feats.shape[0],
              (long)feats.shape[1], (long)feats.shape[2],
              (long)feats.shape[3], (long)feats.shape[4]);

  std::printf("[dsb] project_depth ...\n");
  Tensor depth1 = project_depth(cfg_, in.points, in.camera2lidar, in.img_aug,
                                in.lidar_aug);  // (6,1,H,W)
  Tensor depth({1, cfg_.N, 1, cfg_.image_h, cfg_.image_w});
  depth.data = depth1.data;

  std::printf("[dsb] vtransform ...\n");
  out = vtransform_->run({{"feats", feats}, {"depth", depth}});
  Tensor depthnet = out.at("depthnet");  // (1,6,198,32,88)

  // fold depth prob into features -> x (B,N,D,fH,fW,C)
  std::printf("[dsb] geometry + bev_pool ...\n");
  Tensor geom = compute_geometry(cfg_, in.camera2lidar, in.intrins, in.img_aug,
                                 in.lidar_aug);  // (6,D,fH,fW,3)
  Tensor x({cfg_.B, cfg_.N, D, fH, fW, C});
  {
    // softmax over D per (n, h, w)
    std::vector<float> depth_p(D * fH * fW * cfg_.N);
    const float *dn = depthnet.data.data();
    for (int n = 0; n < cfg_.N; ++n)
      for (int h = 0; h < fH; ++h)
        for (int w = 0; w < fW; ++w) {
          const float *logits = dn + ((n * (D + C) + 0) * fH + h) * fW + w;
          float mx = -1e30f;
          for (int d = 0; d < D; ++d) mx = std::max(mx, logits[d * fH * fW]);
          double sum = 0;
          std::vector<float> e(D);
          for (int d = 0; d < D; ++d) {
            e[d] = std::exp(logits[d * fH * fW] - mx);
            sum += e[d];
          }
          float *dp = depth_p.data() + ((n * D + 0) * fH + h) * fW + w;
          for (int d = 0; d < D; ++d) dp[d * fH * fW] = (float)(e[d] / sum);
        }
    const float *ft = dn + D * fH * fW;
    for (int n = 0; n < cfg_.N; ++n)
      for (int d = 0; d < D; ++d)
        for (int h = 0; h < fH; ++h)
          for (int w = 0; w < fW; ++w) {
            float dp = depth_p[((n * D + d) * fH + h) * fW + w];
            const float *fc = ft + n * (C * fH * fW) + h * fW + w;
            float *xc = x.data.data() + (((n * D + d) * fH + h) * fW + w) * C;
            for (int c = 0; c < C; ++c) xc[c] = dp * fc[c * fH * fW];
          }
  }
  Tensor bev = bev_pool(cfg_, geom, x);  // (C, 360, 360)
  bev.reshape({cfg_.B, C, bev.shape[1], bev.shape[2]});
  std::printf("[dsb]   bev_pool out %ldx%ldx%ld\n", (long)bev.shape[1],
              (long)bev.shape[2], (long)bev.shape[3]);

  Outputs out_res;
  out_res.cam_bev = downsample_conv(cfg_, bev, downsample_blob_);  // (1,80,180,180)
  std::printf("[dsb]   cam_bev %ldx%ldx%ldx%ld\n", (long)out_res.cam_bev.shape[0],
              (long)out_res.cam_bev.shape[1], (long)out_res.cam_bev.shape[2],
              (long)out_res.cam_bev.shape[3]);

  // ---- lidar branch ----------------------------------------------------
  std::printf("[dsb] voxelize + sparse encoder ...\n");
  Tensor vfeats, vcoords, vsizes;
  voxelize(in.points, cfg_, vfeats, vcoords, vsizes);
  out_res.lidar_bev = sparse_encoder_forward(cfg_, vfeats, vcoords, vsizes,
                                             sparse_blob_);
  std::printf("[dsb]   lidar_bev %ldx%ldx%ldx%ld (voxels %ld)\n",
              (long)out_res.lidar_bev.shape[0], (long)out_res.lidar_bev.shape[1],
              (long)out_res.lidar_bev.shape[2], (long)out_res.lidar_bev.shape[3],
              (long)vcoords.shape[0]);

  // ---- fuser + head ----------------------------------------------------
  std::printf("[dsb] fuser ...\n");
  out = fuser_->run({{"cam_bev", out_res.cam_bev},
                     {"lidar_bev", out_res.lidar_bev}});
  out_res.decoder_feat = out.at("decoder_feat");  // (1,512,180,180)

  std::printf("[dsb] head.bbox ...\n");
  out = head_->run({{"decoder_feat", out_res.decoder_feat}});
  std::vector<Tensor> preds;
  for (const char *k :
       {"dense_heatmap", "heatmap", "query_heatmap_score", "center",
        "height", "dim", "rot", "vel"})
    preds.push_back(out.at(k));

  out_res.det = decode_det(cfg_, preds);
  std::printf("[dsb] det boxes: %ld\n", (long)out_res.det.boxes.shape[0]);
  return out_res;
}

void InferencePipeline::verify(const std::string &input_dir,
                               const Outputs &out) const {
  const std::string ref_feat = input_dir + "/ref/decoder_feat.bin";
  if (!std::filesystem::exists(ref_feat)) return;
  std::vector<float> ref =
      read_bin(ref_feat, out.decoder_feat.data.size());
  Tensor ref_t(out.decoder_feat.shape);
  ref_t.data = std::move(ref);
  double e = rel_err(out.decoder_feat, ref_t);
  std::printf("[dsb] decoder_feat vs torch: max rel err %.4e\n", e);

  const std::string ref_boxes = input_dir + "/ref/boxes.json";
  if (std::filesystem::exists(ref_boxes)) {
    std::ifstream f(ref_boxes);
    std::string js((std::istreambuf_iterator<char>(f)), {});
    std::printf("[dsb] ref boxes present (json %zu bytes)\n", js.size());
  }
}

void InferencePipeline::dump(const Outputs &out,
                             const std::string &dump_dir) const {
  std::filesystem::create_directories(dump_dir);
  auto dump_tensor = [&](const std::string &path, const Tensor &t) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char *>(t.data.data()),
            t.data.size() * sizeof(float));
    std::printf("  dumped %s (%zu floats)\n", path.c_str(), t.data.size());
  };
  dump_tensor(dump_dir + "/cam_bev.bin", out.cam_bev);
  dump_tensor(dump_dir + "/lidar_bev.bin", out.lidar_bev);
  dump_tensor(dump_dir + "/decoder_feat.bin", out.decoder_feat);
  write_bin(dump_dir + "/boxes.bin", out.det.boxes.data);
  write_bin(dump_dir + "/scores.bin", out.det.scores.data);
  write_bin(dump_dir + "/labels.bin", out.det.labels.data);
}

}  // namespace dsb
