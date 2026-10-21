#!/usr/bin/env python3
"""批量运行 SECOND 检测所有帧，输出每帧 ROI 结果。

用法:
  python batch_detect.py --mcap <mcap_file> --model <model_path> [--score 0.3]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


def _decoded_iter(mcap_path, topic, factory):
    """不依赖 mcap 库 summary 的解码迭代，兼容截断/未封口文件。"""
    from utils import iter_mcap_messages
    for schema, chan, msg in iter_mcap_messages(mcap_path, topics=[topic]):
        decoder = factory.decoder_for(chan.message_encoding, schema)
        if decoder is None:
            continue
        yield schema, chan, msg, decoder(msg.data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcap", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--score", type=float, default=0.3)
    parser.add_argument("--lidar-topic", default="/lidar/rs_h32f70_2/points")
    parser.add_argument("--out-dir", default="outputs/vis")
    args = parser.parse_args()

    from mcap_ros2.decoder import DecoderFactory
    from detector import SECONDBoardDetector
    from utils import pointcloud2_to_xyzi, iter_mcap_messages
    import open3d as o3d

    # datatype 映射 (与 config.yaml 中 point_field_datatype_to_np 一致)
    dtype_map = {
        1: "i1", 2: "u1", 3: "i2", 4: "u2",
        5: "i4", 6: "u4", 7: "f4", 8: "f8",
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = SECONDBoardDetector(model_path=args.model, score_thr=args.score)

    total = 0
    ok_count = 0
    fail_count = 0

    print(f"开始批量检测: {args.mcap}")
    print(f"模型: {args.model}, score_thr={args.score}")

    factory = DecoderFactory()
    for i, (schema, channel, msg, ros_msg) in enumerate(
        _decoded_iter(args.mcap, args.lidar_topic, factory)
    ):
        total += 1
        # 按字段 offset 正确解析 (与标定主程序一致)
        pts = pointcloud2_to_xyzi(ros_msg, dtype_map)
        print(f"  frame {i}: 原始点云 {len(pts)} pts (含 ring)")

        ok, roi = detector.detect(pts)
        if ok and roi is not None:
            (xmin, xmax), (ymin, ymax), (zmin, zmax) = roi
            ok_count += 1
            print(f"frame {i}: OK  ROI=[{xmin:.2f},{xmax:.2f}] [{ymin:.2f},{ymax:.2f}] [{zmin:.2f},{zmax:.2f}]")

            roi_mask = (
                (pts[:, 0] >= xmin) & (pts[:, 0] <= xmax)
                & (pts[:, 1] >= ymin) & (pts[:, 1] <= ymax)
                & (pts[:, 2] >= zmin) & (pts[:, 2] <= zmax)
            )
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
            colors = np.zeros((len(pts), 3))
            colors[~roi_mask] = [0.5, 0.5, 0.5]
            colors[roi_mask] = [0, 1, 0]
            pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud(str(out_dir / f"batch_frame_{i:04d}.pcd"), pcd)
        else:
            fail_count += 1
            print(f"frame {i}: FAIL")

    print(f"\n检测完成: 共 {total} 帧, 成功 {ok_count} 帧, 失败 {fail_count} 帧")
    print(f"带颜色 pcd 已保存到: {out_dir}")


if __name__ == "__main__":
    main()
