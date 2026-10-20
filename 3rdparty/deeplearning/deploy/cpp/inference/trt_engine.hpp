#pragma once
// TensorRT engine wrapper (trt_engine.cpp): deserialize + execute.

#include "tensor.hpp"

#include <map>
#include <string>

namespace dsb {

class TrtEngine {
 public:
  explicit TrtEngine(const std::string &engine_path);
  ~TrtEngine();
  // inputs: name -> flat float tensor (static shapes). Returns outputs map
  // (shapes read from the engine bindings).
  std::map<std::string, Tensor> run(
      const std::map<std::string, Tensor> &inputs);

 private:
  struct Impl;
  Impl *impl_;
};

}  // namespace dsb
