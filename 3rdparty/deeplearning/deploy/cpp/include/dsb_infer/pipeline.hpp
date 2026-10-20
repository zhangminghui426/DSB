#pragma once
// 推理编排：Preprocessor -> Inferencer -> Postprocessor（+ verify / dump）。
// 与 Python 侧 deploy/pipeline.py（TrtPipeline）逐层对应。

#include "dsb_infer/config.hpp"
#include "dsb_infer/inference/inference.hpp"
#include "dsb_infer/postprocess/postprocess.hpp"
#include "dsb_infer/preprocess/preprocess.hpp"

#include <string>

namespace dsb {

class InferencePipeline {
 public:
  struct Result {
    Tensor cam_bev;       // (1, 80, 180, 180)
    Tensor lidar_bev;     // (1, 256, 180, 180)
    Tensor decoder_feat;  // (1, 512, 180, 180)
    DetOutput det;        // 解码后的 boxes / scores / labels
  };

  explicit InferencePipeline(const Config &cfg);

  // 加载输入（preprocess）+ engines / 权重（inference），返回原始输入。
  Preprocessor::Inputs load(const std::string &input_dir,
                            const std::string &engine_dir);

  // 完整链路：infer（含三层）-> 结果。
  Result infer(const Preprocessor::Inputs &in);

  // decoder_feat vs torch 参考（存在 <input_dir>/ref/decoder_feat.bin 时）。
  void verify(const std::string &input_dir, const Result &out) const;

  // 写出 cam/lidar BEV、decoder_feat 与检测结果到 dump_dir。
  void dump(const Result &out, const std::string &dump_dir) const;

 private:
  Config cfg_;
  Preprocessor preprocessor_;
  Inferencer inferencer_;
  Postprocessor postprocessor_;
};

}  // namespace dsb
