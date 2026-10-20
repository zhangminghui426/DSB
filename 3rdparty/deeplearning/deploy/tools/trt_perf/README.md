# TRT Perf Analysis (NVIDIA TensorRT `.agents/skills/trt-perf-analysis`)

交互式 HTML 性能报告工具，来自 NVIDIA/TensorRT 官方仓库的
`.agents/skills/trt-perf-analysis`（Apache-2.0，本目录随仓库分发以便离线使用）。

## 输入

成对的逐层数据（trtexec `--dumpProfile --dumpLayerInfo` 输出）：

```
layers_<label>.json   层信息（layer 数组，或 TRT 11 inspector 的 {"Layers": [...]}）
profile_<label>.json  逐层时延（根为 list；每项含 name/timeMs/averageMs/medianMs/percentage）
```

## 用法

```bash
# 1) 采集（在 GPU 机器上，需先构建 engine）
cd 3rdparty/deeplearning
bash deploy/launch/perf_trt.sh det   # -> outputs/perf/layers_*.json + profile_*.json
#    pip-only tensorrt 无 trtexec 时：TRTEXEC=/path/to/TensorRT-10.9/bin/trtexec bash deploy/launch/perf_trt.sh det

# 2) 生成报告（首次运行会自动从 NVIDIA/TensorRT 官方仓库拉取 report_template 的
#    app.js / netron-assets.js 大文件，见 run.sh）
bash deploy/tools/trt_perf/run.sh    # -> outputs/perf/report/（analyze/validate/package 一条龙）

# 3) 预览
python -m http.server 8765 --bind 127.0.0.1 --directory 3rdparty/deeplearning/outputs/perf/report
#   浏览器打开 http://127.0.0.1:8765/
```

## 目录

- `scripts/` — analyze_trt_perf.py / validate_analyze_data.py / package_report.py / trt_perf 包
- `schemas/analyze-data.schema.json` — 分析输出 JSON 的结构化校验 schema
- `report_template/` — 报告 HTML 模板（index.html/styles.css/favicon/ThirdPartyNotices；
  app.js 与 netron-assets.js 体积较大未入库，由 run.sh 自动从 NVIDIA 官方 raw 拉取）
- `run.sh` — 本地封装（analyze → validate → package）
