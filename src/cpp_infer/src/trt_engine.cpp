#include "dsb_infer/inference.hpp"
#include "dsb_infer/ops.hpp"

#include <cuda_runtime_api.h>
#include <NvInfer.h>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace dsb {

using nvinfer1::Dims;
using nvinfer1::Dims2;
using nvinfer1::Dims4;

TrtEngine::TrtEngine(const std::string &path, int device) : dev_(device) {
  if (cudaSetDevice(dev_) != cudaSuccess)
    throw std::runtime_error("cudaSetDevice failed");
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open engine: " + path);
  std::string data((std::istreambuf_iterator<char>(f)),
                   std::istreambuf_iterator<char>());
  runtime_.reset(nvinfer1::createInferRuntime(logger_));
  engine_.reset(runtime_->deserializeCudaEngine(data.data(), data.size()));
  if (!engine_) throw std::runtime_error("failed to deserialize: " + path);
  context_ = std::unique_ptr<nvinfer1::IExecutionContext>(
      engine_->createExecutionContext());
  if (!context_)
    throw std::runtime_error("failed to create context: " + path);
}

void TrtEngine::run(const std::vector<Tensor> &inputs,
                    std::vector<Tensor> &outputs) {
  if (inputs.size() != input_names_.size())
    throw std::runtime_error("input count mismatch");
  if (outputs.size() != output_names_.size())
    throw std::runtime_error("output count mismatch");

  // fixed-shape engines: pin each I/O dims from the ONNX batch-1 shapes.
  std::vector<void *> bindings;
  for (size_t i = 0; i < input_names_.size(); ++i) {
    const auto *dims = context_->getBindingShape(i);
    const Tensor &t = inputs[i];
    if ((int)t.shape.size() != dims->nbDims) {
      // camera.backbone img is (B, N, C, H, W); engine binding is (B*N, C, H, W)
      if (dims->nbDims == 4 && t.shape.size() == 5) {
        bindings.push_back(t.data.data());
        continue;
      }
      throw std::runtime_error("input rank mismatch");
    }
    bindings.push_back(t.data.data());
  }
  size_t offset = input_names_.size();
  for (size_t i = 0; i < output_names_.size(); ++i)
    bindings.push_back(outputs[i].data.data());
  (void)offset;

  if (!context_->executeV2(bindings.data()))
    throw std::runtime_error("engine execute failed");
}

void TrtEngine::setTensorAddresses(const std::vector<void *> &ptrs) {
  for (size_t i = 0; i < input_names_.size(); ++i)
    context_->setTensorAddress(input_names_[i].c_str(), ptrs[i]);
  for (size_t i = 0; i < output_names_.size(); ++i)
    context_->setTensorAddress(output_names_[i].c_str(),
                               ptrs[input_names_.size() + i]);
}

std::vector<Dims> TrtEngine::inputShapes() const {
  std::vector<Dims> shapes;
  for (size_t i = 0; i < input_names_.size(); ++i)
    shapes.push_back(context_->getBindingShape((int)i));
  return shapes;
}

std::vector<Dims> TrtEngine::outputShapes() const {
  std::vector<Dims> shapes;
  for (size_t i = 0; i < output_names_.size(); ++i)
    shapes.push_back(context_->getBindingShape(
        (int)(input_names_.size() + i)));
  return shapes;
}

}  // namespace dsb
