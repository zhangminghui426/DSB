// DSB det C++ 推理入口（InferencePipeline 三层编排的薄 CLI）。

#include "pipeline.hpp"

#include <chrono>
#include <cstdio>
#include <string>

using namespace dsb;

namespace {

using Clock = std::chrono::steady_clock;
double ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

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

    Clock::time_point t_load = Clock::now();
    InferencePipeline pipe(cfg);
    auto inputs = pipe.load(input_dir, engine_dir);
    Clock::time_point t_infer = Clock::now();
    std::printf("[dsb] [time] %-22s %8.2f ms\n", "load inputs+engines",
                ms(t_load, t_infer));
    auto result = pipe.infer(inputs);
    Clock::time_point t_verify = Clock::now();
    std::printf("[dsb] [time] %-22s %8.2f ms\n", "infer (total)",
                ms(t_infer, t_verify));
    pipe.verify(input_dir, result);
    if (!dump_dir.empty()) pipe.dump(result, dump_dir);
    Clock::time_point t_done = Clock::now();
    std::printf("[dsb] [time] %-22s %8.2f ms\n", "verify + dump",
                ms(t_verify, t_done));
    std::printf("[dsb] [time] %-22s %8.2f ms\n", "TOTAL", ms(t_load, t_done));
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[dsb] ERROR: %s\n", e.what());
    return 1;
  }
  return 0;
}
