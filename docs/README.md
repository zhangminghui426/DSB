# docs/

- [dsb_tech_solution.tex](dsb_tech_solution.tex) / [PDF](dsb_tech_solution.pdf) —
  整体技术方案：离线训练/ONNX 导出/TRT 构建 + 在线 ROS2 时间戳同步/TensorRT 推理。

编译（需要 XeLaTeX 或 tectonic，使用 Noto Sans CJK SC）：

```bash
cd docs
xelatex dsb_tech_solution.tex      # 或 tectonic -X compile dsb_tech_solution.tex
```
