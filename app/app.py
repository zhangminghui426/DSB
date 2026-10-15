"""Fast-Calib 标定 UI 界面。

启动:
  streamlit run app/app.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import streamlit as st

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
FAST_CALIB_DIR = PROJECT_ROOT / "3rdparty" / "fast_calib"
SCRIPTS_DIR = FAST_CALIB_DIR / "scripts"
TMP_DIR = PROJECT_ROOT / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Fast-Calib 标定", layout="wide")
st.title("Fast-Calib 雷达-相机外参标定")

# 全局终端日志 (所有操作的日志都输出到这里)
if "terminal_log" not in st.session_state:
    st.session_state["terminal_log"] = ""

def log_to_terminal(msg: str):
    """向全局终端追加一行日志"""
    st.session_state["terminal_log"] += msg
    if not msg.endswith("\n"):
        st.session_state["terminal_log"] += "\n"

# ==================== 侧边栏 ====================
with st.sidebar:
    st.header("配置")

    # 1. 数据文件
    st.subheader("1. 数据文件")
    uploaded_mcap = st.file_uploader("上传 .mcap 数据", type=["mcap"])
    if uploaded_mcap is not None:
        mcap_path = TMP_DIR / uploaded_mcap.name
        with open(mcap_path, "wb") as f:
            f.write(uploaded_mcap.getbuffer())
        st.session_state["mcap_file"] = str(mcap_path)
        log_to_terminal(f"[上传] mcap 已保存: {mcap_path}")

    # 2. 相机内参
    st.subheader("2. 相机内参")
    uploaded_cam = st.file_uploader("上传 内参.txt", type=["txt"])
    cam_content = ""
    if uploaded_cam is not None:
        cam_content = uploaded_cam.read().decode("utf-8")
        cam_path = TMP_DIR / uploaded_cam.name
        with open(cam_path, "w", encoding="utf-8") as f:
            f.write(cam_content)
        st.text_area("内参内容", cam_content, height=150)
        log_to_terminal(f"[上传] 内参已保存: {cam_path}")

    # 3. 检测器
    st.subheader("3. 检测器")
    detector_type = st.selectbox(
        "标定板检测器",
        options=["手动 ROI", "SECOND 自动检测"],
        index=1,
    )
    if detector_type == "SECOND 自动检测":
        second_score_thr = st.slider("score 阈值", 0.0, 1.0, 0.3, 0.05)
    else:
        second_score_thr = 0.3

    # 4. 手动 ROI
    st.subheader("4. 手动 ROI")
    c1, c2 = st.columns(2)
    with c1:
        x_min = st.number_input("x_min", -10.0, 10.0, 0.5, 0.1)
        x_max = st.number_input("x_max", -10.0, 10.0, 2.5, 0.1)
    with c2:
        y_min = st.number_input("y_min", -10.0, 10.0, -1.0, 0.1)
        y_max = st.number_input("y_max", -10.0, 10.0, 2.0, 0.1)
    c3, c4 = st.columns(2)
    with c3:
        z_min = st.number_input("z_min", -10.0, 10.0, -1.5, 0.1)
    with c4:
        z_max = st.number_input("z_max", -10.0, 10.0, 0.5, 0.1)

    # 5. 算法参数
    st.subheader("5. 算法参数")
    plane_tol = st.number_input("RANSAC 容差 (m)", 0.001, 0.1, 0.01, 0.001)
    n_frames = st.number_input("最大标定帧数", 3, 200, 50, 1)
    min_sample = st.number_input("最小样本数", 3, 20, 6, 1)

    # 终端日志 (所有操作的日志都显示在这里)
    st.subheader("终端日志")
    terminal_box = st.empty()
    terminal_box.code(st.session_state["terminal_log"], language="bash", height=300)

# ==================== 主界面 ====================
tab1, tab2, tab3, tab4 = st.tabs(["标定运行", "可视化质检", "结果输出", "3D点云预览"])

with tab1:
    st.header("运行标定")
    mcap_file = st.session_state.get("mcap_file", "")
    if not mcap_file:
        st.warning("请先在左侧上传 .mcap 数据文件")
    else:
        st.markdown(f"**数据**: `{mcap_file}`  |  **检测器**: {detector_type}")

    if st.button("开始标定", type="primary", disabled=not mcap_file):
        second_model = str("/tmp/best_model.pth")
        roi = f"[[{x_min}, {x_max}], [{y_min}, {y_max}], [{z_min}, {z_max}]]"
        cfg_text = f"""calib_board:
  hole_centers_board:
    [
      [-0.25,  0.20, 0.0],
      [ 0.25,  0.20, 0.0],
      [-0.25, -0.20, 0.0],
      [ 0.25, -0.20, 0.0]
    ]
  marker_size: 0.20
  hole_radius: 0.12
  marker_centers_board:
    1: [-0.55,  0.35, 0.0]
    2: [ 0.55,  0.35, 0.0]
    3: [-0.55, -0.35, 0.0]
    4: [ 0.55, -0.35, 0.0]

hole_palette:
  [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]

point_field_datatype_to_np:
  1: "i1"
  2: "u2"
  3: "i2"
  4: "u2"
  5: "i4"
  6: "u4"
  7: "f4"
  8: "f8"

format_map:
  h264: "h264"
  h265: "hevc"
  hevc: "hevc"
  avc1: "h264"
  x264: "h264"

camera_intrinsic:
  type: fisheye
  K:
    [
      [514.24, 0.0, 960.58],
      [0.0, 513.98, 768.73],
      [0.0, 0.0, 1.0]
    ]
  dist: [0.0964, -0.0128, -0.0076, 0.0017]

calib_algo:
  max_dt_ns: 100000000
  plane_tol: {plane_tol}
  plane_iters: 300
  board_vertical_prior: true
  min_sample: {min_sample}
  n_calib_frames: {n_frames}
  calib_seed: 0
  save_intermediate: true
  save_pointcloud: false
  save_roi_pointcloud: false
  pcl_fmt: "pcd"

default_pc_roi_lim: {roi}

second_detector:
  enabled: {"true" if detector_type == "SECOND 自动检测" else "false"}
  model_path: "{second_model}"
  score_thr: {second_score_thr}

data_config:
  mcap_file: "{mcap_file}"
  cam_topics:
    cam_gmsl_2: "/camera/gmsl_2/image/encoded"
  lidar_topic: "/lidar/rs_h32f70_2/points"

output_dir: "{PROJECT_ROOT / 'outputs'}"
"""
        tmp_cfg = TMP_DIR / "_ui_config.yaml"
        with open(tmp_cfg, "w", encoding="utf-8") as f:
            f.write(cfg_text)

        st.subheader("运行日志")
        log_box = st.empty()
        log_text = ""
        status_box = st.empty()
        status_box.info("标定运行中...")

        proc = subprocess.Popen(
            [sys.executable, str(SCRIPTS_DIR / "calib_lidar_cmaera.py"),
             "--config", str(tmp_cfg)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, cwd=str(PROJECT_ROOT), bufsize=1,
        )
        for line in proc.stdout:
            log_text += line
            # 同时输出到全局终端
            log_to_terminal(line.rstrip())
            # 固定高度, 带滚动条
            log_box.code(log_text, language="bash", height=400)
        proc.wait()

        output_root = PROJECT_ROOT / "outputs"
        result_dir = output_root / "result"

        if proc.returncode == 0:
            status_box.success(f"标定完成! (返回码 {proc.returncode})")
            # 自动列出结果文件
            st.subheader("生成的产物")
            if result_dir.exists():
                files = list(result_dir.glob("*"))
                if files:
                    for f in sorted(files):
                        st.write(f"  - `{f.name}`  ({f.stat().st_size / 1024:.1f} KB)")
                else:
                    status_box.warning(f"result 目录存在但为空，可能有效样本数不足")
            else:
                status_box.error(f"result 目录不存在: {result_dir}")
        else:
            status_box.error(f"标定失败, 返回码 {proc.returncode}")
            st.error("请查看上方日志的错误信息")

with tab2:
    st.header("可视化质检")
    vis_frame = st.number_input("帧号", 0, 1000, 0, 1)
    result_dir = PROJECT_ROOT / "outputs" / "result"
    default_T = str(result_dir / "cam_gmsl_2_lidar2cam_fastcalib.npy")
    vis_T = st.text_input("外参文件路径", value=default_T)
    if st.button("运行投影可视化"):
        log_box = st.empty()
        log_text = ""
        proc = subprocess.Popen(
            [sys.executable, str(SCRIPTS_DIR / "calib_lidar_cmaera.py"),
             "--vis", vis_T, "--frame", str(vis_frame),
             "--config", str(TMP_DIR / "_ui_config.yaml")],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, cwd=str(PROJECT_ROOT), bufsize=1,
        )
        for line in proc.stdout:
            log_text += line
            log_to_terminal(line.rstrip())
            log_box.code(log_text, language="bash", height=400)
        proc.wait()
        if proc.returncode == 0:
            st.success("可视化完成!")
        else:
            st.error(f"可视化失败, 返回码 {proc.returncode}")

with tab3:
    st.header("标定结果")
    if result_dir.exists():
        for f in result_dir.glob("*.npy"):
            st.subheader(f.name)
            T = np.load(f)
            st.write("旋转矩阵 R:")
            st.write(T[:3, :3])
            st.write("平移向量 t:")
            st.write(T[:3, 3])
    else:
        st.info("暂无结果, 请先运行标定")

with tab4:
    st.header("3D点云预览")

    # 支持两种输入：mcap 帧 或 上传 pcd
    input_mode = st.radio("输入方式", ["上传 pcd 文件", "从 mcap 读帧"], horizontal=True)

    if input_mode == "上传 pcd 文件":
        uploaded_pcd = st.file_uploader("上传 .pcd 文件", type=["pcd"])
        if uploaded_pcd is not None:
            pcd_path = TMP_DIR / uploaded_pcd.name
            with open(pcd_path, "wb") as f:
                f.write(uploaded_pcd.getbuffer())
            st.success(f"已保存: {pcd_path}")

            import open3d as o3d
            pcd = o3d.io.read_point_cloud(str(pcd_path))
            pts = np.asarray(pcd.points)
            st.info(f"点云点数: {len(pts)}, 有颜色: {pcd.has_colors()}")

            # 手动 ROI 框选
            st.subheader("ROI 框选")
            c1, c2 = st.columns(2)
            with c1:
                px_min = st.number_input("pcd x_min", -50.0, 50.0, float(x_min), 0.1, key="pcd_xmin")
                px_max = st.number_input("pcd x_max", -50.0, 50.0, float(x_max), 0.1, key="pcd_xmax")
            with c2:
                py_min = st.number_input("pcd y_min", -50.0, 50.0, float(y_min), 0.1, key="pcd_ymin")
                py_max = st.number_input("pcd y_max", -50.0, 50.0, float(y_max), 0.1, key="pcd_ymax")
            c3, c4 = st.columns(2)
            with c3:
                pz_min = st.number_input("pcd z_min", -20.0, 20.0, float(z_min), 0.1, key="pcd_zmin")
            with c4:
                pz_max = st.number_input("pcd z_max", -20.0, 20.0, float(z_max), 0.1, key="pcd_zmax")

            roi_mask = (
                (pts[:, 0] >= px_min) & (pts[:, 0] <= px_max)
                & (pts[:, 1] >= py_min) & (pts[:, 1] <= py_max)
                & (pts[:, 2] >= pz_min) & (pts[:, 2] <= pz_max)
            )
            st.info(f"ROI 内点数: {roi_mask.sum()} / {len(pts)}")

            # plotly 可视化 (ROI内绿色, 外灰色)
            import plotly.graph_objects as go
            step = max(1, len(pts) // 80000)
            p_sub = pts[::step]
            roi_sub = roi_mask[::step]

            fig = go.Figure()
            fig.add_trace(go.Scatter3d(
                x=p_sub[~roi_sub, 0], y=p_sub[~roi_sub, 1], z=p_sub[~roi_sub, 2],
                mode="markers", marker=dict(size=1, color="gray", opacity=0.3),
                name="outside ROI",
            ))
            fig.add_trace(go.Scatter3d(
                x=p_sub[roi_sub, 0], y=p_sub[roi_sub, 1], z=p_sub[roi_sub, 2],
                mode="markers", marker=dict(size=2, color="green", opacity=0.8),
                name="inside ROI",
            ))
            fig.update_layout(
                title=f"{uploaded_pcd.name}: {len(pts)} pts, ROI={roi_mask.sum()} pts",
                scene=dict(aspectmode="data", xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
                width=1000, height=600,
            )
            st.plotly_chart(fig, use_container_width=True)

            # 保存带颜色的 pcd
            vis_dir = PROJECT_ROOT / "outputs" / "vis"
            vis_dir.mkdir(parents=True, exist_ok=True)
            out_pcd = o3d.geometry.PointCloud()
            out_pcd.points = o3d.utility.Vector3dVector(pts)
            colors = np.zeros((len(pts), 3))
            colors[~roi_mask] = [0.5, 0.5, 0.5]
            colors[roi_mask] = [0, 1, 0]
            out_pcd.colors = o3d.utility.Vector3dVector(colors)
            out_path = str(vis_dir / f"roi_{uploaded_pcd.name}")
            o3d.io.write_point_cloud(out_path, out_pcd)
            st.info(f"已保存带颜色的 pcd: {out_path}")
    else:
        mcap_file = st.session_state.get("mcap_file", "")
        if not mcap_file:
            st.warning("请先在左侧上传 .mcap 数据文件")
        else:
            lidar_topic = "/lidar/rs_h32f70_2/points"
            cam_topic = "/camera/gmsl_2/image/encoded"

            # 先统计总帧数
            from mcap.reader import make_reader
            from mcap_ros2.decoder import DecoderFactory
            total_frames = 0
            with open(mcap_file, "rb") as f:
                reader = make_reader(f, decoder_factories=[DecoderFactory()])
                for _ in reader.iter_decoded_messages(topics=[lidar_topic]):
                    total_frames += 1
            st.info(f"mcap 共 {total_frames} 帧点云")

            # 单帧可视化
            st.subheader("单帧可视化")
            frame_idx = st.number_input("帧号", 0, max(total_frames - 1, 0), 0, 1)
            if st.button("渲染该帧"):
                import cv2
                import open3d as o3d
                import plotly.graph_objects as go
                sys.path.insert(0, str(SCRIPTS_DIR))
                from detector import SECONDBoardDetector
                from utils import pointcloud2_to_xyzi

                # datatype 映射
                dtype_map = {
                    1: "i1", 2: "u1", 3: "i2", 4: "u2",
                    5: "i4", 6: "u4", 7: "f4", 8: "f8",
                }

                # 读指定帧的点云 (按字段 offset 正确解析)
                pts = None
                with open(mcap_file, "rb") as f:
                    reader = make_reader(f, decoder_factories=[DecoderFactory()])
                    for i, (schema, channel, msg, ros_msg) in enumerate(
                        reader.iter_decoded_messages(topics=[lidar_topic])
                    ):
                        if i == frame_idx:
                            pts = pointcloud2_to_xyzi(ros_msg, dtype_map)
                            break

                if pts is None:
                    st.error(f"帧 {frame_idx} 超出范围")
                else:
                    st.info(f"有效点云: {len(pts)} pts")

                    # 运行 SECOND 检测器
                    model_path = str("/tmp/best_model.pth")
                    detector = SECONDBoardDetector(model_path=model_path, score_thr=second_score_thr)
                    ok, roi = detector.detect(pts)

                    if ok and roi is not None:
                        (xmin, xmax), (ymin, ymax), (zmin, zmax) = roi
                        st.success(f"SECOND 检测成功: x=[{xmin:.2f},{xmax:.2f}] y=[{ymin:.2f},{ymax:.2f}] z=[{zmin:.2f},{zmax:.2f}]")
                    else:
                        st.warning("SECOND 检测失败, 使用手动 ROI")
                        xmin, xmax = x_min, x_max
                        ymin, ymax = y_min, y_max
                        zmin, zmax = z_min, z_max

                    roi_mask = (
                        (pts[:, 0] >= xmin) & (pts[:, 0] <= xmax)
                        & (pts[:, 1] >= ymin) & (pts[:, 1] <= ymax)
                        & (pts[:, 2] >= zmin) & (pts[:, 2] <= zmax)
                    )

                    # plotly 可视化 (ROI内绿色)
                    step = max(1, len(pts) // 80000)
                    p_sub = pts[::step]
                    roi_sub = roi_mask[::step]

                    fig = go.Figure()
                    fig.add_trace(go.Scatter3d(
                        x=p_sub[~roi_sub, 0], y=p_sub[~roi_sub, 1], z=p_sub[~roi_sub, 2],
                        mode="markers", marker=dict(size=1, color="gray", opacity=0.3),
                        name="outside ROI",
                    ))
                    fig.add_trace(go.Scatter3d(
                        x=p_sub[roi_sub, 0], y=p_sub[roi_sub, 1], z=p_sub[roi_sub, 2],
                        mode="markers", marker=dict(size=2, color="green", opacity=0.8),
                        name="inside ROI",
                    ))
                    fig.update_layout(
                        title=f"Frame {frame_idx}: {len(pts)} pts, ROI={roi_mask.sum()} pts",
                        scene=dict(aspectmode="data", xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
                        width=1000, height=600,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # 保存带颜色 pcd
                    vis_dir = PROJECT_ROOT / "outputs" / "vis"
                    vis_dir.mkdir(parents=True, exist_ok=True)
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
                    colors = np.zeros((len(pts), 3))
                    colors[~roi_mask] = [0.5, 0.5, 0.5]
                    colors[roi_mask] = [0, 1, 0]
                    pcd.colors = o3d.utility.Vector3dVector(colors)
                    pcd_path = str(vis_dir / f"frame_{frame_idx:04d}.pcd")
                    o3d.io.write_point_cloud(pcd_path, pcd)
                    st.info(f"已保存: {pcd_path}")

            # 批量检测所有帧 (subprocess 后台运行, 切 tab 不打断)
            st.subheader("批量检测所有帧")
            if st.button("运行 SECOND 检测全部帧"):
                model_path = str("/tmp/best_model.pth")
                out_dir = PROJECT_ROOT / "outputs" / "vis"

                st.subheader("批量检测日志")
                log_box = st.empty()
                log_text = ""
                st.info("批量检测运行中... 切换 tab 不会打断")

                proc = subprocess.Popen(
                    [sys.executable, str(FAST_CALIB_DIR / "tools" / "batch_detect.py"),
                     "--mcap", mcap_file,
                     "--model", model_path,
                     "--score", str(second_score_thr),
                     "--lidar-topic", lidar_topic,
                     "--out-dir", str(out_dir)],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, cwd=str(PROJECT_ROOT), bufsize=1,
                )
                for line in proc.stdout:
                    log_text += line
                    log_to_terminal(line.rstrip())
                    log_box.code(log_text, language="bash", height=400)
                proc.wait()

                if proc.returncode == 0:
                    st.success(f"批量检测完成! 结果已保存到: {out_dir}")
                    # 列出输出文件
                    files = list(Path(out_dir).glob("*.pcd"))
                    if files:
                        st.write(f"共生成 {len(files)} 个 pcd 文件")
                else:
                    st.error(f"批量检测失败, 返回码 {proc.returncode}")
