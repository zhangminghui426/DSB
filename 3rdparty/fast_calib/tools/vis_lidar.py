"""
雷达点云可视化
用法：
  python vis_lidar.py                # 读calib_intermediate下的bin点云(已ROI裁剪)，每帧存三视图PNG
  python vis_lidar.py --mcap         # 直接从mcap读原始点云(未裁剪)再可视化
  python vis_lidar.py --gui 0        # open3d交互窗口查看第0帧(带坐标轴)
  python vis_lidar.py --gui 0 --mcap # 交互查看mcap原始第0帧
bin格式：KITTI式float32每点4通道(x,y,z,intensity)，见1.py run(pcl_fmt="bin")
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_bin(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def load_mcap(mcap_path, lidar_topic, max_frames=None):
    """直接解析mcap PointCloud2, 返回list[(t_ns, pc(N,5))]"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from mcap_ros2.decoder import DecoderFactory
    from utils import pointcloud2_to_xyzi, iter_decoded_messages_robust

    pftn = {1:"i1",2:"u2",3:"i2",4:"u2",5:"i4",6:"u4",7:"f4",8:"f8"}
    out = []
    for _, chan, msg, ros_msg in iter_decoded_messages_robust(
        mcap_path, topics=[lidar_topic], decoder_factories=[DecoderFactory()]
    ):
        try:
            if type(ros_msg).__name__ != "PointCloud2":
                continue
            out.append((msg.log_time, pointcloud2_to_xyzi(ros_msg, pftn)))
            if max_frames is not None and len(out) >= max_frames:
                break
        except Exception as e:
            print(f"parse err: {e}")
    return out


def save_views(pc, out_png, title=""):
    """顶视(x,y,按z着色) + 前视(x,z) + 侧视(y,z)"""
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    sc = axes[0].scatter(pc[:, 0], pc[:, 1], s=0.3, c=pc[:, 2], cmap="viridis")
    axes[0].set_title(f"{title} top (x,y) color=z")
    axes[0].set_aspect("equal"); axes[0].grid(True)
    axes[1].scatter(pc[:, 0], pc[:, 2], s=0.3)
    axes[1].set_title(f"{title} front (x,z)")
    axes[1].set_aspect("equal"); axes[1].grid(True)
    axes[2].scatter(pc[:, 1], pc[:, 2], s=0.3)
    axes[2].set_title(f"{title} side (y,z)")
    axes[2].set_aspect("equal"); axes[2].grid(True)
    fig.colorbar(sc, ax=axes[0], shrink=0.6, label="z(m)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=90)
    plt.close(fig)
    print(f"saved {out_png}  points={pc.shape[0]}")


def gui(pc, title=""):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc[:, :3].astype(np.float64))
    # 按高度着色
    z = pc[:, 2]
    zn = (z - z.min()) / max(1e-6, (z.max() - z.min()))
    cmap = plt.get_cmap("viridis")
    pcd.colors = o3d.utility.Vector3dVector(np.asarray([cmap(v)[:3] for v in zn]))
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    print(f"[gui] {title} points={pc.shape[0]}  鼠标拖拽旋转/滚轮缩放")
    o3d.visualization.draw_geometries(
        [pcd, frame], window_name=f"lidar {title}",
        width=1280, height=800, point_show_normal=False)


def html_view(pc, out_html, title=""):
    """浏览器交互式3D(plotly)：拖拽旋转/滚轮缩放/按z着色，无需GUI环境"""
    import plotly.graph_objects as go
    step = max(1, len(pc) // 30000)   # 降采样保证浏览器流畅
    p = pc[::step]
    fig = go.Figure(go.Scatter3d(
        x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers",
        marker=dict(size=1.2, color=p[:, 2], colorscale="viridis",
                    colorbar=dict(title="z(m)"))))
    fig.update_layout(title=f"{title} points={len(pc)} (show {len(p)})",
                      scene=dict(aspectmode="data"),
                      width=1200, height=800, margin=dict(l=0, r=0, t=30, b=0))
    fig.write_html(out_html)
    print(f"saved {out_html}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcap", action="store_true", help="从mcap读原始点云")
    ap.add_argument("--mcap_path", default="data/20261012-162756_0.mcap")
    ap.add_argument("--lidar_topic", default="/lidar/rs_h32f70_2/points")
    ap.add_argument("--bin_dir", default="./calib_intermediate/cam_gmsl_4/pointcloud")
    ap.add_argument("--max_frames", type=int, default=None, help="最多可视化前N帧(mcap模式有用)")
    ap.add_argument("--gui", type=int, default=None, metavar="FRAME_IDX", help="open3d交互窗口查看第N帧")
    ap.add_argument("--html", type=int, default=None, metavar="FRAME_IDX", help="导出第N帧为浏览器交互式3D HTML")
    ap.add_argument("--out_dir", default="./lidar_vis")
    args = ap.parse_args()

    if args.mcap:
        frames = [pc for _, pc in load_mcap(args.mcap_path, args.lidar_topic, args.max_frames)]
        tag = "mcap"
    else:
        bins = sorted(glob.glob(os.path.join(args.bin_dir, "*.bin")))
        if not bins:
            raise SystemExit(f"{args.bin_dir} 下无bin文件，先跑1.py或加--mcap")
        frames = [load_bin(b) for b in bins]
        tag = "bin"
    if args.max_frames is not None:
        frames = frames[:args.max_frames]

    os.makedirs(args.out_dir, exist_ok=True)

    if args.gui is not None:
        gui(frames[args.gui], f"{tag} frame{args.gui}")
        return

    if args.html is not None:
        html_view(frames[args.html],
                  os.path.join(args.out_dir, f"{tag}_frame{args.html:04d}.html"),
                  title=f"{tag} f{args.html}")
        return

    for i, pc in enumerate(frames):
        save_views(pc, os.path.join(args.out_dir, f"{tag}_frame{i:04d}.png"), title=f"{tag} f{i}")


if __name__ == "__main__":
    main()
