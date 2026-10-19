// DSB det C++ inference entry (thin CLI over InferencePipeline).

#include "dsb_infer/pipeline.hpp"

#include <cstdio>
#include <string>

using namespace dsb;

namespace {

void print_usage(const char *prog) {
  std::fprintf(stderr,
               "usage: %s --input-dir <export_out> --engine-dir <engines> "
               "[--dump-dir <d>]\n",
               prog);
  std::fprintf(stderr,
               "  <export_out> from tools/export_cpp_input.py; <engines> "
               "holds camera.backbone / camera.vtransform / fuser / head.bbox\n");
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
      print_usage(argv[0]);
      return 2;
    }
  }
  if (input_dir.empty() || engine_dir.empty()) {
    print_usage(argv[0]);
    return 2;
  }

  try {
    Config cfg = load_config(input_dir + "/meta.txt");
    std::printf("[dsb] B=%d N=%d depth_bins=%d\n", cfg.B, cfg.N,
                cfg.depth_bins());

    InferencePipeline pipe(cfg);
    auto inputs = pipe.load_inputs(input_dir);
    pipe.load_engines(engine_dir);
    pipe.load_weights(input_dir);
    auto outputs = pipe.infer(inputs);
    pipe.verify(input_dir, outputs);
    if (!dump_dir.empty()) pipe.dump(outputs, dump_dir);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[dsb] ERROR: %s\n", e.what());
    return 1;
  }
  return 0;
}
