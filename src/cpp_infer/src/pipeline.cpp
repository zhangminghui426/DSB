// 推理编排实现：Preprocessor -> Inferencer -> Postprocessor + verify/dump。
// 镜像 Python 侧 deploy/pipeline.py（TrtPipeline）。

#include "dsb_infer/pipeline.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>

namespace dsb {

namespace {

double rel_err(const Tensor &a, const Tensor &b) {
  double mx = 0;
  for (size_t i = 0; i < a.data.size(); ++i) {
    const double denom = std::abs((double)b.data[i]) + 1e-6;
    const double e = std::abs((double)a.data[i] - b.data[i]) / denom;
    mx = std::max(mx, e);
  }
  return mx;  // max rel err
}

}  // namespace

InferencePipeline::InferencePipeline(const Config &cfg)
    : cfg_(cfg), preprocessor_(cfg_), inferencer_(cfg_), postprocessor_(cfg_) {}

Preprocessor::Inputs InferencePipeline::load(const std::string &input_dir,
                                             const std::string &engine_dir) {
  auto inputs = preprocessor_.load_inputs(input_dir);
  inferencer_.load_engines(engine_dir);
  inferencer_.load_weights(input_dir);
  return inputs;
}

InferencePipeline::Result InferencePipeline::infer(
    const Preprocessor::Inputs &in) {
  Result out;

  // ---- camera 分支 ------------------------------------------------------
  std::printf("[dsb] backbone ...\n");
  Tensor feats = inferencer_.camera_backbone(in.img);  // (1,6,256,32,88)
  std::printf("[dsb]   feats %ldx%ldx%ldx%ldx%ld\n", (long)feats.shape[0],
              (long)feats.shape[1], (long)feats.shape[2], (long)feats.shape[3],
              (long)feats.shape[4]);

  std::printf("[dsb] project_depth ...\n");
  Tensor depth1 = preprocessor_.project_depth(in);  // (6,1,H,W)
  Tensor depth({cfg_.B, cfg_.N, 1, cfg_.image_h, cfg_.image_w});
  depth.data = depth1.data;

  std::printf("[dsb] vtransform ...\n");
  Tensor depthnet =
      inferencer_.camera_vtransform(feats, depth);  // (1,6,198,32,88)

  out.cam_bev = inferencer_.camera_bev(depthnet, in);  // (1,80,180,180)

  // ---- lidar 分支 --------------------------------------------------------
  out.lidar_bev = inferencer_.lidar_branch(in);  // (1,256,180,180)

  // ---- fuser + head ------------------------------------------------------
  out.decoder_feat =
      inferencer_.fuser(out.cam_bev, out.lidar_bev);  // (1,512,180,180)
  out.det = postprocessor_.decode(inferencer_.head(out.decoder_feat));
  std::printf("[dsb] det boxes: %ld\n", (long)out.det.boxes.shape[0]);
  return out;
}

void InferencePipeline::verify(const std::string &input_dir,
                               const Result &out) const {
  const std::string ref_feat = input_dir + "/ref/decoder_feat.bin";
  if (!std::filesystem::exists(ref_feat)) return;
  std::vector<float> ref = read_bin(ref_feat, out.decoder_feat.data.size());
  Tensor ref_t(out.decoder_feat.shape);
  ref_t.data = std::move(ref);
  const double e = rel_err(out.decoder_feat, ref_t);
  std::printf("[dsb] decoder_feat vs torch: max rel err %.4e\n", e);

  const std::string ref_boxes = input_dir + "/ref/boxes.json";
  if (std::filesystem::exists(ref_boxes)) {
    std::ifstream f(ref_boxes);
    std::string js((std::istreambuf_iterator<char>(f)), {});
    std::printf("[dsb] ref boxes present (json %zu bytes)\n", js.size());
  }
}

void InferencePipeline::dump(const Result &out,
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
