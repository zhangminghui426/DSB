#include "dsb_infer/trt_engine.hpp"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <cstdio>
#include <fstream>
#include <memory>

namespace dsb {

namespace {
// TRT logger: keep quiet unless errors.
class Logger : public nvinfer1::ILogger {
  void log(Severity sev, const char *msg) noexcept override {
    if (sev == Severity::kERROR || sev == Severity::kINTERNAL_ERROR)
      std::fprintf(stderr, "[trt] %s\n", msg);
  }
};
Logger g_logger;

// RAII stream.
struct CudaStream {
  cudaStream_t s = nullptr;
  CudaStream() { cudaStreamCreate(&s); }
  ~CudaStream() { if (s) cudaStreamDestroy(s); }
};
}  // namespace

struct TrtEngine::Impl {
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> ctx;
  std::unique_ptr<CudaStream> stream;
};

TrtEngine::TrtEngine(const std::string &engine_path) : impl_(new Impl) {
  std::ifstream f(engine_path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open engine: " + engine_path);
  std::vector<char> buf((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
  auto runtime = std::unique_ptr<nvinfer1::IRuntime>(
      nvinfer1::createInferRuntime(g_logger));
  impl_->engine.reset(runtime->deserializeCudaEngine(buf.data(), buf.size()));
  if (!impl_->engine)
    throw std::runtime_error("failed to deserialize engine: " + engine_path);
  impl_->ctx.reset(impl_->engine->createExecutionContext());
  impl_->stream.reset(new CudaStream);
}

TrtEngine::~TrtEngine() { delete impl_; }

std::map<std::string, Tensor> TrtEngine::run(
    const std::map<std::string, Tensor> &inputs) {
  std::map<std::string, Tensor> outputs;
  std::vector<std::unique_ptr<void, decltype(&cudaFree)>> dev;
  auto alloc = [&](size_t bytes) {
    void *p = nullptr;
    cudaMalloc(&p, bytes);
    dev.emplace_back(p, &cudaFree);
    return p;
  };

  nvinfer1::IExecutionContext *ctx = impl_->ctx.get();
  // V3 API: name -> device address for all I/O tensors.
  std::vector<nvinfer1::IExecutionContext::IOBindingEntry> bindings;
  for (int i = 0; i < impl_->engine->getNbIOTensors(); ++i) {
    const char *name = impl_->engine->getIOTensorName(i);
    nvinfer1::TensorIOMode mode = impl_->engine->getTensorIOMode(name);
    if (mode == nvinfer1::TensorIOMode::kINPUT) {
      const Tensor &t = inputs.at(name);
      void *d = alloc(t.data.size() * sizeof(float));
      cudaMemcpy(d, t.data.data(), t.data.size() * sizeof(float),
                 cudaMemcpyHostToDevice);
      bindings.push_back({name, d});
    } else {
      nvinfer1::Dims dims = impl_->engine->getTensorShape(name);
      std::vector<int64_t> shape(dims.nbDims);
      for (int d = 0; d < dims.nbDims; ++d) shape[d] = dims.d[d];
      Tensor t(shape);
      void *d = alloc(t.data.size() * sizeof(float));
      outputs[name] = std::move(t);
      bindings.push_back({name, d});
    }
  }
  if (!ctx->enqueueV3(impl_->stream->s, bindings)) {
    cudaStreamSynchronize(impl_->stream->s);
    throw std::runtime_error("engine execute failed");
  }
  cudaStreamSynchronize(impl_->stream->s);

  // copy outputs back
  for (const auto &b : bindings) {
    if (impl_->engine->getTensorIOMode(b.tensorName) ==
        nvinfer1::TensorIOMode::kINPUT)
      continue;
    cudaMemcpy(outputs[b.tensorName].data.data(), b.device,
               outputs[b.tensorName].data.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
  }
  return outputs;
}

}  // namespace dsb
