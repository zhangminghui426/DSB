// DSB det C++ inference entry.
//
// Pipeline (batch=1), mirrors scripts/export.py + deploy/infer_trt.py +
// models/* at det config:
//   img -> [TRT camera.backbone] -> feats
//       -> depth (project_depth, CPU) -> [TRT camera.vtransform] -> depthnet
//       -> geometry + bev_pool (CPU) -> (80,360,360) -> downsample (CPU)
//       -> cam_bev (1,80,180,180)
//   points -> voxelize + sparse encoder (CPU) -> lidar_bev (1,256,180,180)
//   [TRT fuser] -> decoder_feat (1,512,180,180)
//   [TRT head.bbox] -> 8 preds -> decode (CPU) -> boxes / scores / labels
//
// Usage:
//   dsb_infer --input-dir <export_out> --engine-dir <engines> [--dump-dir <d>]
// where <export_out> comes from tools/export_cpp_input.py and <engines> holds
// camera.backbone / camera.vtransform / fuser / head.bbox TRT engines.

#include "dsb_infer/ops.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>

using namespace dsb;

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

void dump_tensor(const std::string &path, const Tensor &t) {
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char *>(t.data.data()),
          t.data.size() * sizeof(float));
  std::printf("  dumped %s (%zu floats)\n", path.c_str(), t.data.size());
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

int main(int argc, char **argv) {
  std::string input_dir, engine_dir, dump_dir;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--input-dir" && i + 1 < argc) input_dir = argv[++i];
    else if (a == "--engine-dir" && i + 1 < argc) engine_dir = argv[++i];
    else if (a == "--dump-dir" && i + 1 < argc) dump_dir = argv[++i];
    else {
      std::fprintf(stderr, "unknown arg: %s\n", a.c_str());
      return 2;
    }
  }
  if (input_dir.empty() || engine_dir.empty()) {
    std::fprintf(stderr,
                 "usage: dsb_infer --input-dir <export_out> "
                 "--engine-dir <engines> [--dump-dir <d>]\n");
    return 2;
  }

  try {
    Config cfg = load_config(input_dir + "/meta.txt");
    std::printf("[dsb] B=%d N=%d depth_bins=%d\n", cfg.B, cfg.N,
                cfg.depth_bins());

    // ---- inputs ---------------------------------------------------------
    Tensor img({cfg.B, cfg.N, 3, cfg.image_h, cfg.image_w});
    img.data = read_bin(input_dir + "/inputs/img.bin", img.data.size());
    {
      std::ifstream f(input_dir + "/inputs/points.bin", std::ios::binary);
      int32_t n = 0;
      f.read(reinterpret_cast<char *>(&n), sizeof(n));
      Tensor pts({n, 5});
      f.read(reinterpret_cast<char *>(pts.data.data()),
             (std::streamsize)(n * 5 * sizeof(float)));
      std::printf("[dsb] points: %d\n", n);
      // read matrices
      std::vector<float> mats = read_bin(input_dir + "/inputs/matrices.bin",
                                         19 * 16);
      auto c2l = read_mat4_block(mats, 6);        // idx 0..5
      auto intr = read_mat4_block(
          std::vector<float>(mats.begin() + 6 * 16, mats.end()), 6);
      auto iaug = read_mat4_block(
          std::vector<float>(mats.begin() + 12 * 16, mats.end()), 6);
      Mat4 laug{};
      std::memcpy(laug.m, mats.data() + 18 * 16, 16 * sizeof(float));

      // ---- engines -------------------------------------------------------
      TrtEngine bb(find_engine(engine_dir, "camera.backbone"));
      TrtEngine vt(find_engine(engine_dir, "camera.vtransform"));
      TrtEngine fu(find_engine(engine_dir, "fuser"));
      TrtEngine hd(find_engine(engine_dir, "head.bbox"));

      // ---- camera branch -------------------------------------------------
      std::printf("[dsb] backbone ...\n");
      auto out = bb.run({{"img", img}});
      Tensor feats = out.at("feats");  // (1,6,256,32,88)
      std::printf("[dsb]   feats %ldx%ldx%ldx%ldx%ld\n", (long)feats.shape[0],
                  (long)feats.shape[1], (long)feats.shape[2],
                  (long)feats.shape[3], (long)feats.shape[4]);

      std::printf("[dsb] project_depth ...\n");
      Tensor depth1 = project_depth(cfg, pts, c2l, iaug, laug);  // (6,1,H,W)
      Tensor depth({1, cfg.N, 1, cfg.image_h, cfg.image_w});
      depth.data = depth1.data;

      std::printf("[dsb] vtransform ...\n");
      out = vt.run({{"feats", feats}, {"depth", depth}});
      Tensor depthnet = out.at("depthnet");  // (1,6,198,32,88)
      int D = cfg.depth_bins(), C = cfg.vt_out_c;
      int fH = cfg.feat_h, fW = cfg.feat_w;

      // fold depth prob into features -> x (B,N,D,fH,fW,C)
      std::printf("[dsb] geometry + bev_pool ...\n");
      Tensor geom = compute_geometry(cfg, c2l, intr, iaug, laug);  // (6,D,fH,fW,3)
      Tensor x({cfg.B, cfg.N, D, fH, fW, C});
      {
        // softmax over D per (n, h, w)
        std::vector<float> depth_p(D * fH * fW * cfg.N);
        const float *dn = depthnet.data.data();
        for (int n = 0; n < cfg.N; ++n)
          for (int h = 0; h < fH; ++h)
            for (int w = 0; w < fW; ++w) {
              const float *logits = dn + ((n * (D + C) + 0) * fH + h) * fW + w;
              float mx = -1e30f;
              for (int d = 0; d < D; ++d)
                mx = std::max(mx, logits[d * fH * fW]);
              double sum = 0;
              std::vector<float> e(D);
              for (int d = 0; d < D; ++d) {
                e[d] = std::exp(logits[d * fH * fW] - mx);
                sum += e[d];
              }
              float *dp = depth_p.data() +
                          ((n * D + 0) * fH + h) * fW + w;
              for (int d = 0; d < D; ++d) dp[d * fH * fW] = (float)(e[d] / sum);
            }
        const float *ft = dn + D * fH * fW;
        for (int n = 0; n < cfg.N; ++n)
          for (int d = 0; d < D; ++d)
            for (int h = 0; h < fH; ++h)
              for (int w = 0; w < fW; ++w) {
                float dp = depth_p[((n * D + d) * fH + h) * fW + w];
                const float *fc = ft + n * (C * fH * fW) + h * fW + w;
                float *xc = x.data.data() +
                            (((n * D + d) * fH + h) * fW + w) * C;
                for (int c = 0; c < C; ++c) xc[c] = dp * fc[c * fH * fW];
              }
      }
      Tensor bev = bev_pool(cfg, geom, x);  // (C, 360, 360)
      bev.reshape({cfg.B, C, bev.shape[1], bev.shape[2]});
      std::printf("[dsb]   bev_pool out %ldx%ldx%ld\n", (long)bev.shape[1],
                  (long)bev.shape[2], (long)bev.shape[3]);

      std::vector<float> ds_blob;
      {
        std::ifstream f(input_dir + "/weights/downsample.bin",
                        std::ios::binary);
        std::vector<char> buf((std::istreambuf_iterator<char>(f)), {});
        ds_blob.assign(reinterpret_cast<float *>(buf.data()),
                       reinterpret_cast<float *>(buf.data() + buf.size()));
      }
      Tensor cam_bev = downsample_conv(cfg, bev, ds_blob);  // (1,80,180,180)
      std::printf("[dsb]   cam_bev %ldx%ldx%ldx%ld\n", (long)cam_bev.shape[0],
                  (long)cam_bev.shape[1], (long)cam_bev.shape[2],
                  (long)cam_bev.shape[3]);

      // ---- lidar branch --------------------------------------------------
      std::printf("[dsb] voxelize + sparse encoder ...\n");
      Tensor vfeats, vcoords, vsizes;
      voxelize(pts, cfg, vfeats, vcoords, vsizes);
      std::vector<float> sp_blob;
      {
        std::ifstream f(input_dir + "/weights/sparse_encoder.bin",
                        std::ios::binary);
        std::vector<char> buf((std::istreambuf_iterator<char>(f)), {});
        sp_blob.assign(reinterpret_cast<float *>(buf.data()),
                       reinterpret_cast<float *>(buf.data() + buf.size()));
      }
      Tensor lidar_bev =
          sparse_encoder_forward(cfg, vfeats, vcoords, vsizes, sp_blob);
      std::printf("[dsb]   lidar_bev %ldx%ldx%ldx%ld (voxels %ld)\n",
                  (long)lidar_bev.shape[0], (long)lidar_bev.shape[1],
                  (long)lidar_bev.shape[2], (long)lidar_bev.shape[3],
                  (long)vcoords.shape[0]);

      // ---- fuser + head ----------------------------------------------------
      std::printf("[dsb] fuser ...\n");
      out = fu.run({{"cam_bev", cam_bev}, {"lidar_bev", lidar_bev}});
      Tensor decoder_feat = out.at("decoder_feat");  // (1,512,180,180)

      std::printf("[dsb] head.bbox ...\n");
      out = hd.run({{"decoder_feat", decoder_feat}});
      std::vector<Tensor> preds;
      for (const char *k :
           {"dense_heatmap", "heatmap", "query_heatmap_score", "center",
            "height", "dim", "rot", "vel"})
        preds.push_back(out.at(k));

      DetOutput det = decode_det(cfg, preds);
      std::printf("[dsb] det boxes: %ld\n", (long)det.boxes.shape[0]);

      // ---- reference check -------------------------------------------------
      if (std::filesystem::exists(input_dir + "/ref/decoder_feat.bin")) {
        std::vector<float> ref =
            read_bin(input_dir + "/ref/decoder_feat.bin",
                     decoder_feat.data.size());
        Tensor ref_t(decoder_feat.shape);
        ref_t.data = std::move(ref);
        double e = rel_err(decoder_feat, ref_t);
        std::printf("[dsb] decoder_feat vs torch: max rel err %.4e\n", e);
      }
      if (std::filesystem::exists(input_dir + "/ref/boxes.json")) {
        std::ifstream f(input_dir + "/ref/boxes.json");
        std::string js((std::istreambuf_iterator<char>(f)), {});
        std::printf("[dsb] ref boxes present (json %zu bytes)\n", js.size());
        (void)js;
      }

      // ---- dump ----------------------------------------------------------
      if (!dump_dir.empty()) {
        std::filesystem::create_directories(dump_dir);
        dump_tensor(dump_dir + "/cam_bev.bin", cam_bev);
        dump_tensor(dump_dir + "/lidar_bev.bin", lidar_bev);
        dump_tensor(dump_dir + "/decoder_feat.bin", decoder_feat);
        write_bin(dump_dir + "/boxes.bin", det.boxes.data);
        write_bin(dump_dir + "/scores.bin", det.scores.data);
        write_bin(dump_dir + "/labels.bin", det.labels.data);
      }
    }
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[dsb] ERROR: %s\n", e.what());
    return 1;
  }
  return 0;
}
