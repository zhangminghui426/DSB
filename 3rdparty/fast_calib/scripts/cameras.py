import numpy as np
from typing import Tuple


class BaseCamera:
    def __init__(self, K: np.ndarray, dist: np.ndarray):
        self.K = K.copy()
        self.dist = dist.copy()
        self.fx = K[0,0]
        self.fy = K[1,1]
        self.cx = K[0,2]
        self.cy = K[1,2]

    def project(self, pc_cam: np.ndarray) -> np.ndarray:
        """3D点 (N,3) in camera frame -> 2D像素uv (N,2)"""
        raise NotImplementedError

    def estimate_pose(self, obj_points: np.ndarray, img_points: np.ndarray):
        """solvePnP，子类可重载"""
        raise NotImplementedError


class FisheyeCamera(BaseCamera):
    """Kannala-Brandt KB4鱼眼相机模型"""
    def project(self, pc_cam: np.ndarray) -> np.ndarray:
        Xc, Yc, Zc = pc_cam.T
        r = np.sqrt(Xc ** 2 + Yc ** 2)
        theta = np.arctan2(r, Zc)
        phi = np.arctan2(Yc, Xc)
        k1, k2, k3, k4 = self.dist
        rho = theta + k1 * theta ** 3 + k2 * theta ** 5 + k3 * theta ** 7 + k4 * theta ** 9
        u = self.fx * rho * np.cos(phi) + self.cx
        v = self.fy * rho * np.sin(phi) + self.cy
        return np.column_stack([u, v])

    def estimate_pose(self, obj_points: np.ndarray, img_points: np.ndarray):
        import cv2
        ok, rvec, tvec = cv2.fisheye.solvePnP(
            obj_points, img_points, self.K, self.dist, flags=cv2.SOLVEPNP_IPPE
        )
        return ok, rvec, tvec


class PinholeCamera(BaseCamera):
    def project(self, pc_cam: np.ndarray) -> np.ndarray:
        import cv2
        pts, _ = cv2.projectPoints(pc_cam, np.zeros(3), np.zeros(3), self.K, self.dist)
        return pts.reshape(-1, 2)

    def estimate_pose(self, obj_points: np.ndarray, img_points: np.ndarray):
        import cv2
        ok, rvec, tvec = cv2.solvePnP(
            obj_points, img_points, self.K, self.dist, flags=cv2.SOLVEPNP_IPPE
        )
        return ok, rvec, tvec
