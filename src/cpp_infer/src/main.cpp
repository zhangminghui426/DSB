// dsb_infer 入口：加载输入/engine -> 三层推理 -> verify/dump。
#include "dsb_infer/pipeline.hpp"

#include <cstring>
#include <iostream>

#include "dsb_infer/config.hpp"

namespace {

struct Args {
  std::string input_dir;
  std::string engine_dir;
  std::string dump_dir;
};

void usage() {
  std::cout << "usage: dsb_infer --engine-dir DIR --input-dir DIR [--dump DIR]\n"
               "  --input-dir   导出样本目录（tools/export_cpp_input.py 产出）\n"
               "  --engine-dir  TRT engine 目录（camera.backbone / camera.vtransform"
               " / fuser / head.bbox .engine）\n"
               "  --dump        输出目录（cam_bev/lidar_bev/decoder_feat/boxes/"
               "scores/labels .bin）\n"
               "  存在 <input-dir>/ref/decoder_feat.bin 时自动对比 decoder_feat\n";
}

}  // namespace

int main(int argc, char **argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--input-dir" && i + 1 < argc) a.input_dir = argv[++i];
    else if (arg == "--engine-dir" && i + 1 < argc) a.engine_dir = argv[++i];
    else if (arg == "--dump" && i + 1 < argc) a.dump_dir = argv[++i];
    else if (arg == "--help" || arg == "-h") { usage(); return 0; }
    else {
      std::cerr << "unknown arg: " << arg << "\n";
      usage();
      return 2;
    }
  }
  if (a.input_dir.empty() || a.engine_dir.empty()) {
    usage();
    return 2;
  }

  try {
    const auto cfg_path = a.input_dir + "/meta.txt";
    dsb::InferencePipeline pipe(dsb::load_config(cfg_path));
    auto in = pipe.load(a.input_dir, a.engine_dir);
    auto out = pipe.infer(in);
    std::cout << "[trt] decoder_feat " << out.decoder_feat.shape_str() << "\n";
    std::cout << "[trt] decoded " << out.det.scores.numel()
              << " boxes\n";
    pipe.verify(a.input_dir, out);
    if (!a.dump_dir.empty()) {
      pipe.dump(out, a.dump_dir);
      std::cout << "[trt] dumped -> " << a.dump_dir << "\n";
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "[fatal] " << e.what() << "\n";
    return 1;
  }
}
