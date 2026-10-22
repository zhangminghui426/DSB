"""bf 推理编排（镜像 deploy/cpp/src/bevfusion/bevfusion.cpp forward 数据流）。

  normalize -> camera.backbone(img+depth) -> bevpool(depth_weights x feature)
  -> camera.vtransform -> fuser(camera, lidar) -> head.bbox -> decode

自定义算子（cuda_ops）与 C++ kernel 一一对应；TRT engine 为 4 个 .plan
（build_trt_engine.sh 构建，IO fp16:chw）。
"""
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))  # deeplearning/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))       # deploy/

from . import cuda_ops as ops          # noqa: E402
from .runtime import load_engines       # noqa: E402
from .sparse import build_lidar_backbone, sparse_forward  # noqa: E402

# ------------------------------------------------------------- 模型参数（与 CUDA-BEVFusion main.cpp 一致）
VOXEL_MIN = (-54.0, -54.0, -5.0)
VOXEL_SIZE = (0.075, 0.075, 0.2)
VOXEL_GRID = (1440, 1440, 41)

GEOM_BX = (-53.85, -53.85, 0.0)        # bound.x + bound.z/2
GEOM_DX = (0.3, 0.3, 20.0)
GEOM_NX = (360, 360, 1)
GEOM_DIM = (360, 360, 80)

FEAT_W, FEAT_H, D_DEPTH = 88, 32, 118
IMG_W, IMG_H = 704, 256
NUM_CAMERA = 6

HEAD_CONF_THRESHOLD = 0.12
HEAD_OUT_SIZE_FACTOR = 8
HEAD_PC_RANGE = (-54.0, -54.0)
HEAD_POST_RANGE = (-61.2, -61.2, -10.0, 61.2, 61.2, 10.0)
HEAD_VOXEL_SIZE = (0.075, 0.075)

IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD = (0.229, 0.224, 0.225)


class BevfusionBf:
    """CUDA-BEVFusion 结构 Python 推理（与 C++ main.cpp 同布局）。"""

    def __init__(self, cfg_path: str, ckpt_path: str, model_dir: str):
        self.engines = load_engines(model_dir)
        self.backbone = build_lidar_backbone(cfg_path, ckpt_path)
        self.frustum = ops.make_frustum(NUM_CAMERA, FEAT_W, FEAT_H, D_DEPTH, IMG_W, IMG_H)
        self.batch_size = 1

    # ------------------------------------------------------------ 输入准备
    def normalize(self, imgs) -> torch.Tensor:
        """6 张原始图 (H,W,3) uint8 -> (1,6,3,256,704) half（同 camera-normalization.cu）。

        imgs: (6,H,W,3) uint8 numpy 或 list[6]。
        """
        if not isinstance(imgs, np.ndarray):
            imgs = np.stack([np.asarray(i) for i in imgs])
        resized = []
        for i in range(imgs.shape[0]):
            r = cv2.resize(imgs[i], (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
            resized.append(r)
        t = torch.from_numpy(np.stack(resized)).cuda().float().permute(0, 3, 1, 2) / 255.0
        mean = torch.tensor(IMG_MEAN, dtype=torch.float32, device=t.device).view(1, 3, 1, 1)
        std = torch.tensor(IMG_STD, dtype=torch.float32, device=t.device).view(1, 3, 1, 1)
        t = (t - mean) / std
        return t.unsqueeze(0).half()

    # ------------------------------------------------------------ forward
    @torch.no_grad()
    def forward(self, imgs, points: np.ndarray, camera2lidar: np.ndarray, camera_intrinsics: np.ndarray,
                lidar2image: np.ndarray, img_aug_matrix: np.ndarray) -> list:
        """完整推理（数据流对齐 bevfusion.cpp forward_only）。

        imgs (6,H,W,3) uint8；points (N,5) float16；矩阵均 (6,4,4) float32
        返回检测框列表 [{position:(3,), size:(3,), velocity:(2,), z_rotation, score, id}]
        """
        # --- camera 分支
        normed = self.normalize(imgs)
        points_t = torch.from_numpy(np.ascontiguousarray(points)).cuda().half()
        c2l = torch.from_numpy(np.ascontiguousarray(camera2lidar)).cuda().float()
        k_inv = torch.from_numpy(np.ascontiguousarray(np.linalg.inv(camera_intrinsics))).cuda().float()
        l2i = torch.from_numpy(np.ascontiguousarray(lidar2image)).cuda().float()
        aug = torch.from_numpy(np.ascontiguousarray(img_aug_matrix)).cuda().float()
        aug_inv = torch.from_numpy(np.ascontiguousarray(np.linalg.inv(img_aug_matrix))).cuda().float()

        depth = ops.compute_depth(points_t, l2i, aug)                     # (6,256,704)
        backbone_out = self.engines["camera.backbone"].run({
            "img": normed,
            "depth": depth.reshape(1, NUM_CAMERA, 1, IMG_H, IMG_W),
        })
        cam_feature = backbone_out["camera_feature"]          # (1,6,32,88,80) HWC
        cam_depth_w = backbone_out["camera_depth_weights"]    # (1,6,118,32,88)

        intervals, indices, _, _ = ops.compute_geometry(
            self.frustum, c2l, k_inv, aug_inv,
            GEOM_BX, GEOM_DX, GEOM_NX, GEOM_DIM, NUM_CAMERA)
        farea = FEAT_W * FEAT_H
        bev = ops.bevpool(
            cam_feature.reshape(NUM_CAMERA, farea, -1),
            cam_depth_w.reshape(-1),
            intervals, indices,
            out_h=GEOM_NX[1], out_w=GEOM_NX[0], ndepth=D_DEPTH, farea=farea,
            nchannel=cam_feature.shape[-1])                              # (80,360,360)
        cam_bev = bev.reshape(1, -1, GEOM_NX[1], GEOM_NX[0])
        cam_bev_feat = self.engines["camera.vtransform"].run({"feat_in": cam_bev})["feat_out"]  # (1,80,180,180)

        # --- lidar 分支
        vfeats, vcoors, _ = ops.voxelize(points_t, VOXEL_MIN, VOXEL_SIZE, VOXEL_GRID)
        lidar_feat = sparse_forward(self.backbone, vfeats, vcoors)       # (1,256,180,180)

        # --- 融合 + 检测头
        middle = self.engines["fuser"].run({"camera": cam_bev_feat, "lidar": lidar_feat})["middle"]
        head_out = self.engines["head.bbox"].run({"middle": middle})
        return decode(head_out)

    def forward_no_normalize(self, normed_imgs, points_t, c2l, k_inv, l2i, aug, aug_inv):
        """已归一化输入的前向（供对拍/调试，镜像 C++ forward_no_normalize）。"""
        depth = ops.compute_depth(points_t, l2i, aug)
        backbone_out = self.engines["camera.backbone"].run({
            "img": normed_imgs, "depth": depth.reshape(1, NUM_CAMERA, 1, IMG_H, IMG_W),
        })
        cam_feature = backbone_out["camera_feature"]
        cam_depth_w = backbone_out["camera_depth_weights"]
        intervals, indices, _, _ = ops.compute_geometry(
            self.frustum, c2l, k_inv, aug_inv, GEOM_BX, GEOM_DX, GEOM_NX, GEOM_DIM, NUM_CAMERA)
        farea = FEAT_W * FEAT_H
        bev = ops.bevpool(cam_feature.reshape(NUM_CAMERA, farea, -1), cam_depth_w.reshape(-1),
                          intervals, indices, out_h=GEOM_NX[1], out_w=GEOM_NX[0],
                          ndepth=D_DEPTH, farea=farea, nchannel=cam_feature.shape[-1])
        cam_bev_feat = self.engines["camera.vtransform"].run(
            {"feat_in": bev.reshape(1, -1, GEOM_NX[1], GEOM_NX[0])})["feat_out"]
        vfeats, vcoors, _ = ops.voxelize(points_t, VOXEL_MIN, VOXEL_SIZE, VOXEL_GRID)
        lidar_feat = sparse_forward(self.backbone, vfeats, vcoors)
        middle = self.engines["fuser"].run({"camera": cam_bev_feat, "lidar": lidar_feat})["middle"]
        head_out = self.engines["head.bbox"].run({"middle": middle})
        return decode(head_out)


# ------------------------------------------------------------ head decode
def decode(head_out: dict) -> list:
    """head.bbox 输出 decode（同 C++ head-transbbox.cu decode_kernel）。

    score (1,10,200) rot (1,2,200) dim (1,3,200) reg (1,2,200)
    height (1,1,200) vel (1,2,200)
    """
    score = head_out["score"][0]          # (10,200)
    reg = head_out["reg"][0]              # (2,200)
    dim = head_out["dim"][0].exp()        # (3,200)（同 kernel：dim = exp）
    height = head_out["height"][0, 0]     # (200,)
    rot = head_out["rot"][0]              # (2,200)
    vel = head_out["vel"][0]              # (2,200)

    conf, label = score.max(0)
    xs = reg[0] * HEAD_OUT_SIZE_FACTOR * HEAD_VOXEL_SIZE[0] + HEAD_PC_RANGE[0]
    ys = reg[1] * HEAD_OUT_SIZE_FACTOR * HEAD_VOXEL_SIZE[1] + HEAD_PC_RANGE[1]
    zs = height - dim[2] * 0.5
    keep = (conf >= HEAD_CONF_THRESHOLD) & \
        (xs >= HEAD_POST_RANGE[0]) & (xs <= HEAD_POST_RANGE[3]) & \
        (ys >= HEAD_POST_RANGE[1]) & (ys <= HEAD_POST_RANGE[4]) & \
        (zs >= HEAD_POST_RANGE[2]) & (zs <= HEAD_POST_RANGE[5])
    i = keep.nonzero(as_tuple=False).squeeze(1)
    boxes = []
    for k in i.tolist():
        boxes.append({
            "position": (float(xs[k]), float(ys[k]), float(zs[k])),
            "size": (float(dim[0, k]), float(dim[1, k]), float(dim[2, k])),
            "velocity": (float(vel[0, k]), float(vel[1, k])),
            "z_rotation": float(torch.atan2(rot[0, k], rot[1, k])),
            "score": float(conf[k]),
            "id": int(label[k]),
        })
    return boxes
