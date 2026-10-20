#pragma once
// TensorRT engine wrapper (trt_engine.cpp): deserialize + execute.

#include "tensor.hpp"

#include <cuda_runtime_api.h>

#include <map>
#include <string>

namespace dsb {

class TrtEngine {
 public:
  explicit TrtEngine(const std::string &engine_path);
  ~TrtEngine();
  // inputs: name -> flat float tensor (static shapes). Returns outputs map
  // (shapes read from the engine bindings).
  // stream: 外部 CUDA stream（与 CUDA 自定义算子同流串行）；nullptr 用内部 stream。
  std::map<std::string, Tensor> run(
      const std::map<std::string, Tensor> &inputs,
      cudaStream_t stream = nullptr);

 private:
  struct Impl;
  Impl *impl_;
};

}  // namespace dsb
