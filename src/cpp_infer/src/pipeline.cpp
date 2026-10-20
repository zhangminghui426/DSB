// 编排层实现：Preprocessor -> Inferencer -> Postprocessor（+ verify / dump）。
#include "dsb_infer/pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>

namespace dsb {

namespace {

void write_bin(const std::string &path, const void *data, size_t nbytes) {
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(nbytes));
}

}  // namespace

InferencePipeline::InferencePipeline(const Config &cfg)
    : cfg_(cfg), preprocessor_(cfg), inferencer_(cfg), postprocessor_(cfg) {}

Preprocessor::Inputs InferencePipeline::load(const std::string &input_dir,
                                             const std::string &engine_dir) {
  inferencer_.load_engines(engine_dir);
  inferencer_.load_weights(input_dir);
  return preprocessor_.load_inputs(input_dir);
}

InferencePipeline::Result InferencePipeline::infer(const Preprocessor::Inputs &in) {
  Result out;
  const Tensor feats = inferencer_.camera_backbone(in.img);
  const Tensor depth = preprocessor_.project_depth(in);
  const Tensor depthnet = inferencer_.camera_vtransform(feats, depth);
  out.cam_bev = inferencer_.camera_bev(depthnet, in);
  out.lidar_bev = inferencer_.lidar_branch(in);
  out.decoder_feat = inferencer_.fuser(out.cam_bev, out.lidar_bev);
  out.det = postprocessor_.decode(inferencer_.head(out.decoder_feat));
  return out;
}

void InferencePipeline::verify(const std::string &input_dir,
                               const Result &out) const {
  const std::string ref = input_dir + "/ref/decoder_feat.bin";
  std::ifstream f(ref, std::ios::binary | std::ios::ate);
  if (!f) {
    std::cout << "[verify] no " << ref << "; skip decoder_feat check\n";
    return;
  }
  const auto sz = f.tellg();
  std::vector<float> ref_data(static_cast<size_t>(sz) / sizeof(float));
  f.seekg(0);
  f.read(reinterpret_cast<char *>(ref_data.data()), sz);

  float worst = 0.f, scale = 1.f;
  const size_t n = std::min(ref_data.size(), out.decoder_feat.numel());
  for (size_t i = 0; i < n; ++i) {
    const float d = std::abs(out.decoder_feat.data()[i] - ref_data[i]);
    scale = std::max(scale, std::abs(ref_data[i]));
    worst = std::max(worst, d / scale);
  }
  std::cout << "[verify] decoder_feat rel diff vs torch: " << worst
            << " (tol 5e-3)\n";
}

void InferencePipeline::dump(const Result &out, const std::string &dump_dir) const {
  write_bin(dump_dir + "/cam_bev.bin", out.cam_bev.data(),
            out.cam_bev.numel() * sizeof(float));
  write_bin(dump_dir + "/lidar_bev.bin", out.lidar_bev.data(),
            out.lidar_bev.numel() * sizeof(float));
  write_bin(dump_dir + "/decoder_feat.bin", out.decoder_feat.data(),
            out.decoder_feat.numel() * sizeof(float));
  write_bin(dump_dir + "/boxes.bin", out.det.boxes.data(),
            out.det.boxes.numel() * sizeof(float));
  write_bin(dump_dir + "/scores.bin", out.det.scores.data(),
            out.det.scores.numel() * sizeof(float));
  write_bin(dump_dir + "/labels.bin", out.det.labels.data(),
            out.det.labels.numel() * sizeof(int32_t));
}

}  // namespace dsb
