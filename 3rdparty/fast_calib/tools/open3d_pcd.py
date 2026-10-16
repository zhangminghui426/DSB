"""
open3d打开1.py保存的带色pcd（本体浅灰，4孔邻域+孔心按红/绿/蓝/黄逐孔区分）
用法：python open3d_pcd.py [pcd路径]   （默认取calib_intermediate首帧）
窗口内 shift+左键 选点，关闭窗口后打印选中点索引/坐标/颜色
"""
import glob
import sys

import numpy as np
import open3d as o3d

path = sys.argv[1] if len(sys.argv) > 1 else sorted(
    glob.glob("calib_intermediate/cam_gmsl_4/pointcloud/*.pcd"))[0]
pcd = o3d.io.read_point_cloud(path)
print(f"{path}: points={len(pcd.points)} has_colors={pcd.has_colors()}")

vis = o3d.visualization.VisualizerWithEditing()
vis.create_window(window_name="pcd (shift+left click to pick)", width=1280, height=800)
vis.add_geometry(pcd)
vis.run()
vis.destroy_window()

picked = vis.get_picked_points()
pts = np.asarray(pcd.points)
cols = np.asarray(pcd.colors) if pcd.has_colors() else None
print("选中点索引：", picked)
for idx in picked:
    line = f"点{idx}: X={pts[idx, 0]:.3f}, Y={pts[idx, 1]:.3f}, Z={pts[idx, 2]:.3f}"
    if cols is not None and idx < len(cols):
        line += f", RGB=({cols[idx, 0]:.2f}, {cols[idx, 1]:.2f}, {cols[idx, 2]:.2f})"
    print(line)
