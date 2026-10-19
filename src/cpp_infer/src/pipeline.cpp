#include "dsb_infer/inference.hpp"
#include "dsb_infer/ops.hpp"

#include <fstream>
#include <stdexcept>

namespace dsb {

using nvinfer1::Dims;

std::vector<std::string> split(const std::string &s, char sep) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : s) {
    if (c == sep) {
      out.push_back(cur);
      cur.clear();
    } else {
      cur.push_back(c);
    }
  }
  out.push_back(cur);
  return out;
}

Tensor make_tensor(const std::vector<int64_t> &shape) {
  Tensor t;
  t.shape = shape;
  size_t n = 1;
  for (int64_t s : shape) n *= (size_t)s;
  t.data.assign(n, 0.0f);
  return t;
}

void fill_tensor(Tensor &t, const std::vector<float> &src) {
  if (src.size() != t.data.size())
    throw std::runtime_error("tensor size mismatch");
  t.data = src;
}

static std::string load_txt(const std::string &path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open: " + path);
  return std::string((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
}

InferencePipeline::InferencePipeline(const Config &cfg,
                                     const std::vector<std::string> &names)
    : cfg_(cfg) {
  for (size_t i = 0; i < names.size(); ++i) {
    engines_.emplace_back(names[i]);
    // expose I/O tensor names in the same order as deploy/graphs.py
    for (auto &n : kInputNames) input_names_.push_back(n);
    for (auto &n : kOutputNames) output_names_.push_back(n);
  }
}

void InferencePipeline::loadFeats(const std::string &feat_dir) {
  feats_ = make_tensor({1, 6, 3, 256, 704});
  fill_tensor(feats_, parse_txt(load_txt(feat_dir + "/feats.txt")));
}

void InferencePipeline::run(const std::vector<Tensor> &inputs,
                            std::vector<Tensor> &outputs) {
  // one engine at a time for now; a single-engine pipeline runs end to end.
  engines_[0].run(inputs, outputs);
}

void InferencePipeline::run_det(const std::vector<Tensor> &inputs,
                                std::vector<Tensor> &preds) {
  std::vector<Tensor> outs(preds.size());
  for (size_t i = 0; i < preds.size(); ++i)
    outs[i] = preds[i];
  engines_[3].run(inputs, outs);
}

void InferencePipeline::run_cam(const std::vector<Tensor> &inputs,
                                std::vector<Tensor> &preds) {
  std::vector<Tensor> outs(preds.size());
  for (size_t i = 0; i < preds.size(); ++i)
    outs[i] = preds[i];
  engines_[0].run(inputs, outs);
}

void InferencePipeline::run_geom(const std::vector<Tensor> &inputs,
                                 std::vector<Tensor> &preds) {
  std::vector<Tensor> outs(preds.size());
  for (size_t i = 0; i < preds.size(); ++i)
    outs[i] = preds[i];
  engines_[1].run(inputs, outs);
}

void InferencePipeline::run_fuser(const std::vector<Tensor> &inputs,
                                  std::vector<Tensor> &preds) {
  std::vector<Tensor> outs(preds.size());
  for (size_t i = 0; i < preds.size(); ++i)
    outs[i] = preds[i];
  engines_[2].run(inputs, outs);
}

DetOutput InferencePipeline::decode_det(const std::vector<Tensor> &preds) {
  return decode_det(cfg_, preds);
}

}  // namespace dsb
