"""
Fast-Calib 鱼眼/针孔通用激光相机外参标定主程序
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import logging
import os
from itertools import combinations
from typing import Optional, Tuple
import cv2
import numpy as np
import av
import open3d as o3d
from scipy.optimize import least_squares
from scipy.spatial.distance import pdist
from scipy.spatial.transform import Rotation
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from utils import (
setup_logger, load_yaml_config, pointcloud2_to_xyzi, msg_stamp_ns,
save_T_yaml, save_point_cloud, colorize_by_holes, circle_fit_2d, residual_3d
)
from detector import CalibrationBoardDetector, SECONDBoardDetector
from cameras import BaseCamera, FisheyeCamera, PinholeCamera


class CompressedVideoDecoder:
    def __init__(self, fmt: str, format_map: dict):
        codec = format_map.get(str(fmt).lower().strip())
        if codec is None:
            raise ValueError(f"unsupported CompressedVideo format: {fmt}")
        self.codec = av.CodecContext.create(codec, "r")
        self.codec.open()
        self._ts_by_pts = {}
        self._idx = 0
        self.n_err = 0

    def decode(self, data, t_ns):
        pts = self._idx
        self._idx += 1
        self._ts_by_pts[pts] = t_ns
        pkt = av.Packet(bytes(data))
        pkt.pts = pkt.dts = pts
        out = []
        try:
            for frame in self.codec.decode(pkt):
                t = self._ts_by_pts.pop(frame.pts, t_ns) if frame.pts is not None else t_ns
                out.append((t, frame.to_ndarray(format="bgr24")))
        except Exception as e:
            self.n_err += 1
            if self.n_err <= 5:
                logging.warning(f"video frame #{pts} decode failed: {e}")
        return out


class FastCalibLidarCamCalibrator:
    def __init__(
        self,
        hole_centers_board: np.ndarray,
        marker_size: float,
        hole_radius: float,
        marker_centers_board: dict,
        max_dt_ns: int,
        plane_tol: float,
        plane_iters: int,
        board_vertical_prior: bool,
        camera: BaseCamera,
        hole_palette: np.ndarray,
        save_intermediate: bool,
        save_pointcloud: bool,
        save_roi_pointcloud: bool,
        output_root: str, # 替换原来两个out_root
        aruco_dict_id=cv2.aruco.DICT_6X6_250,
        board_detector: Optional[CalibrationBoardDetector] = None
    ):
        self.hole_centers_board = hole_centers_board
        self._hole_pair_dists = np.sort(pdist(hole_centers_board))
        ux = np.unique(hole_centers_board[:, 0])
        uy = np.unique(hole_centers_board[:, 1])
        self.delta_width = float(ux[-1] - ux[0])
        self.delta_height = float(uy[-1] - uy[0])
        self.marker_centers_board = marker_centers_board
        self.marker_size = marker_size
        self.hole_radius = hole_radius
        self.plane_tol = plane_tol
        self.plane_iters = plane_iters
        self.board_vertical_prior = board_vertical_prior
        self.max_dt_ns = max_dt_ns
        self.save_intermediate = save_intermediate
        self.save_pointcloud = save_pointcloud
        self.save_roi_pointcloud = save_roi_pointcloud
        # 从总output_root派生目录
        self.out_root_result = os.path.join(output_root, "result")
        self.out_root_intermediate = os.path.join(output_root, "intermediate")
        os.makedirs(self.out_root_result, exist_ok=True)
        self.board_detector = board_detector
        self.camera = camera
        self.hole_palette = hole_palette.copy()
        self.aruco_det = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(aruco_dict_id),
            cv2.aruco.DetectorParameters()
        )

    def _marker_obj_corners(self):
        s = self.marker_size / 2.0
        return np.array([[-s, s, 0.0], [s, s, 0.0], [s, -s, 0.0], [-s, -s, 0.0]], dtype=np.float64)

    def estimate_board_pose(self, corners, ids):
        local = self._marker_obj_corners()
        obj_list, img_list = [], []
        if self.marker_centers_board is not None:
            for i, mid in enumerate(ids.flatten().tolist()):
                center = self.marker_centers_board.get(mid)
                if center is None:
                    continue
                obj_list.append(np.asarray(center, dtype=np.float64) + local)
                img_list.append(corners[i][0].reshape(4, 2))
        if not obj_list:
            k = int(np.argmin(ids.flatten()))
            obj_list, img_list = [local], [corners[k][0].reshape(4, 2)]
        obj_all = np.vstack(obj_list).reshape(1, -1, 3)
        img_all = np.vstack(img_list).reshape(1, -1, 2)
        ok, rvec, tvec = self.camera.estimate_pose(obj_all, img_all)
        return (rvec, tvec) if ok else (None, None)

    def detect_fastcalib_holes_2d(self, img_bgr):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.aruco_det.detectMarkers(gray)
        if ids is None or len(ids) < 4:
            return None, corners, ids, None, None
        rvec, tvec = self.estimate_board_pose(corners, ids)
        if rvec is None:
            return None, corners, ids, None, None
        R_board, _ = cv2.Rodrigues(rvec)
        holes_cam = (R_board @ self.hole_centers_board.T).T + tvec.reshape(3)
        holes_uv = self.camera.project(holes_cam)
        return holes_uv, corners, ids, R_board, tvec.reshape(3)

    def _ring_edge_points(self, pc, plane_model):
        a, b, c, d = plane_model[:4]
        nn = np.sqrt(a * a + b * b + c * c)
        dist = np.abs(pc[:, 0] * a + pc[:, 1] * b + pc[:, 2] * c + d) / nn
        ring = pc[:, 4] if pc.shape[1] >= 5 else np.zeros(len(pc))
        edges = []
        for r in np.unique(ring):
            idx = np.where(ring == r)[0]
            if len(idx) < 10:
                continue
            pts = pc[idx, :3]
            on_plane = dist[idx] < 0.03
            dp = np.linalg.norm(pts - np.roll(pts, 1, axis=0), axis=1)
            dn = np.linalg.norm(pts - np.roll(pts, -1, axis=0), axis=1)
            m = on_plane & ((dp > 0.10) | (dn > 0.10))
            m[0] = m[-1] = False
            if m.any():
                edges.append(pts[m])
        return np.vstack(edges) if edges else np.zeros((0, 3))

    def _square_valid(self, cents4, tol=0.1):
        d = np.sort(pdist(cents4))
        diag = np.hypot(self.delta_width, self.delta_height)
        e = np.sort([self.delta_height, self.delta_height, self.delta_width,
                     self.delta_width, diag, diag])
        return bool(np.all(np.abs(d - e) <= tol * e))

    def detect_hole_centers_lidar(self, pc_roi, max_planes=8):
        xyz = pc_roi[:, :3]
        ring = pc_roi[:, 4] if pc_roi.shape[1] >= 5 else np.zeros(len(pc_roi))
        rest, rest_ring = xyz, ring.copy()
        plane_pts = xyz[:0]
        for _ in range(max_planes):
            if rest.shape[0] < 200:
                break
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(rest.astype(np.float64))
            model, idx = pcd.segment_plane(distance_threshold=self.plane_tol,
                                           ransac_n=3, num_iterations=self.plane_iters)
            idx = np.asarray(idx)
            patch = rest[idx]
            patch_ring = rest_ring[idx]
            rest = np.delete(rest, idx, axis=0)
            rest_ring = np.delete(rest_ring, idx, axis=0)
            if patch.shape[0] < 200:
                continue
            if self.board_vertical_prior and abs(model[2]) > 0.5:
                continue
            plane_pts = patch
            edges = self._ring_edge_points(pc_roi, model)
            if len(edges) < 8:
                continue
            n = model[:3] / np.linalg.norm(model[:3])
            axis = np.cross(n, np.array([0.0, 0.0, 1.0]))
            s = np.linalg.norm(axis)
            if s < 1e-8:
                R_align = np.eye(3)
            else:
                angle = np.arccos(np.clip(n[2], -1.0, 1.0))
                rot_vec = (axis / s) * angle
                R_align = Rotation.from_rotvec(rot_vec).as_matrix()
            al_edge = (R_align @ edges.T).T
            al_patch = (R_align @ patch.T).T
            avg_z = float(al_edge[:, 2].mean())
            exy, pxy = al_edge[:, :2], al_patch[:, :2]
            plane_ring = patch_ring
            gaps = []
            if len(plane_ring) == len(pxy):
                for r in np.unique(plane_ring):
                    msk = plane_ring == r
                    if msk.sum() < 8:
                        continue
                    pp = pxy[msk]
                    w = np.linalg.eigh(np.cov((pp - pp.mean(axis=0)).T))[1][:, 1]
                    p_s = pp[np.argsort(pp @ w)]
                    d = np.linalg.norm(np.diff(p_s, axis=0), axis=1)
                    gaps += [0.5 * (p_s[j] + p_s[j + 1]) for j in np.where((d > 0.15) & (d < 0.35))[0]]
            if len(gaps) < 4:
                continue
            clusters = []
            for g in gaps:
                g2 = g[:2]
                for cl in clusters:
                    if np.linalg.norm(np.mean(cl, axis=0) - g2) < 0.09:
                        cl.append(g2)
                        break
                else:
                    clusters.append([g2])
            coarse = [np.mean(cl, axis=0) for cl in clusters if len(cl) >= 2]
            if len(coarse) < 4:
                continue
            centers = []
            for c0 in np.array(coarse)[:, :2]:
                rim = exy[np.abs(np.linalg.norm(exy - c0, axis=1) - self.hole_radius) < 0.03]
                if len(rim) >= 6:
                    cen, rr = circle_fit_2d(rim)
                    if abs(rr - self.hole_radius) <= 0.03 and np.linalg.norm(cen - c0) < 0.08:
                        c0 = cen
                centers.append(c0)
            if len(centers) < 4:
                continue
            centers = np.array(centers)
            valid = [c4 for c4 in combinations(range(len(centers)), 4) if self._square_valid(centers[list(c4)])]
            if not valid:
                continue
            if len(valid) > 1:
                logging.warning(f"{len(valid)}组圆心满足矩形几何，取第一组")
            c4 = centers[list(valid[0])]
            cents = (R_align.T @ np.column_stack([c4[:, 0], c4[:, 1], np.full(len(c4), avg_z)]).T).T
            return cents, patch
        return None, plane_pts

    def _board_frame_from_centers(self, cents, su):
        m = cents.mean(axis=0)
        ev = np.linalg.eigh(np.cov((cents - m).T))[1]
        u0, n0 = ev[:, 2], ev[:, 0]
        p = 1.0 if (n0 @ m) < 0 else -1.0
        u, n = su * u0, p * n0
        v = np.cross(n, u)
        R_bl = np.stack([u, v, n])
        bc = (R_bl @ (cents - m).T).T
        pi = [int(np.argmin(np.linalg.norm(self.hole_centers_board - q, axis=1))) for q in bc]
        if len(set(pi)) != 4 or max(np.min(np.linalg.norm(self.hole_centers_board - q, axis=1)) for q in bc) > 0.1:
            return None, None
        B_l = np.eye(4)
        B_l[:3, :3] = R_bl
        B_l[:3, 3] = -R_bl @ m
        return B_l, pi

    def _edge_score_maps(self, ed, pc, T):
        if len(pc) == 0:
            return 0.0
        pc_cam = (T[:3, :3] @ pc[:, :3].T).T + T[:3, 3]
        pc_cam = pc_cam[pc_cam[:, 2] > 0.2]
        uv = self.camera.project(pc_cam).astype(int)
        h, w = ed.shape
        m = (uv[:, 0] >= 0) & (uv[:, 0] < w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1)
        uv = uv[m]
        return float(ed[uv[:, 1], uv[:, 0]].sum()) / 255.0 / len(pc) if len(uv) else 0.0

    def calibrate_single_cam(self, lidar_list, img_list, board_poses, edge_frames=None):
        H = self.hole_centers_board
        branches = []
        for su in (1, -1):
            Ls, Cs, Tfs, used, vz = [], [], [], [], []
            for j, (cents, (Rb, tb)) in enumerate(zip(lidar_list, board_poses)):
                B_l, pi = self._board_frame_from_centers(cents, su)
                if B_l is None:
                    continue
                B_c = np.eye(4)
                B_c[:3, :3], B_c[:3, 3] = Rb, tb
                Tfs.append(B_c @ np.linalg.inv(B_l))
                Ls.append(cents[np.argsort(pi)])
                Cs.append((B_c[:3, :3] @ H.T).T + B_c[:3, 3])
                used.append(j)
                vz.append(B_l[:3, :3][1, 2])
            if len(Ls) < 3:
                continue
            vz_mean = float(np.mean(vz))
            L, C = np.vstack(Ls), np.vstack(Cs)
            M = np.mean(Tfs, axis=0)
            U, _, Vt = np.linalg.svd(M[:3, :3])
            R0 = U @ Vt
            if np.linalg.det(R0) < 0:
                U[:, -1] *= -1
                R0 = U @ Vt
            x0 = np.hstack([Rotation.from_matrix(R0).as_rotvec(), M[:3, 3]])
            res = least_squares(residual_3d, x0, args=(L, C), method="trf", loss="huber", f_scale=0.03)
            Rf = Rotation.from_rotvec(res.x[:3]).as_matrix()
            tf = res.x[3:6]
            fr = np.array([np.median(np.linalg.norm((Rf @ Ls[f].T).T + tf - Cs[f], axis=1)) for f in range(len(Ls))])
            keep_f = [f for f in range(len(Ls)) if fr[f] <= max(0.05, 3 * np.median(fr))]
            if 3 <= len(keep_f) < len(Ls):
                L, C = np.vstack([Ls[f] for f in keep_f]), np.vstack([Cs[f] for f in keep_f])
                used = [used[f] for f in keep_f]
                x0 = np.hstack([Rotation.from_matrix(Rf).as_rotvec(), tf])
                res = least_squares(residual_3d, x0, args=(L, C), method="trf", loss="huber", f_scale=0.03)
            T_i = np.eye(4)
            T_i[:3, :3] = Rotation.from_rotvec(res.x[:3]).as_matrix()
            T_i[:3, 3] = res.x[3:6]
            err3d = float(np.median(np.linalg.norm((T_i[:3, :3] @ L.T).T + T_i[:3, 3] - C, axis=1)))
            branches.append((err3d, T_i, Tfs[0], used, vz_mean))
        if not branches:
            raise RuntimeError("no valid board‑registration branch")
        up = [b for b in branches if b[4] > 0.15]
        if len(up) == 1:
            k = branches.index(up[0])
            logging.info(f"branches={len(branches)} upright‑prior vz={[round(b[4],2) for b in branches]} -> pick {k}")
        else:
            per = [[self._edge_score_maps(ed, pc, T_i) for ed, pc in edge_frames] for _, T_i, _, _, _ in branches] if edge_frames else [[b[0]] for b in branches]
            scores = [float(np.mean(pp)) for pp in per]
            k = int(np.argmax(scores))
            logging.info(f"branches={len(branches)} edge‑align={[round(s,3) for s in scores]} -> pick {k}")
        err3d, T_opt, T_init, used, _ = branches[k]
        reps = []
        for j in used:
            pc_cam = (T_opt[:3, :3] @ lidar_list[j].T).T + T_opt[:3, 3]
            uv = self.camera.project(pc_cam)
            cost = np.linalg.norm(uv[:, None, :] - np.asarray(img_list[j])[None, :, :], axis=-1)
            reps.append(cost.min(axis=1).mean())
        logging.info(f"3D‑3D median residual: {err3d*100:.2f} cm ({len(used)} frames used)")
        return T_opt, float(np.mean(reps)), T_init

    def _draw_projection(self, img, pc, T, out_path):
        pc_cam = (T[:3, :3] @ pc[:, :3].T).T + T[:3, 3]
        pc_cam = pc_cam[pc_cam[:, 2] > 0.2]
        if len(pc_cam) == 0:
            return
        uv = self.camera.project(pc_cam).astype(int)
        h, w = img.shape[:2]
        m = (uv[:, 0] >= 0) & (uv[:, 0] < w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1)
        uv, d = uv[m].astype(int), pc_cam[m, 2]
        denom = max(1e-6, d.max() - d.min())
        dn = ((d - d.min()) / denom * 255).astype(np.uint8)
        cmap = cv2.applyColorMap(dn.reshape(-1, 1, 1), cv2.COLORMAP_JET).reshape(-1, 3)
        draw = img.copy()
        step = max(1, len(uv) // 40000)
        for q, col in zip(uv[::step], cmap[::step]):
            cv2.circle(draw, (int(q[0]), int(q[1])), 1, (int(col[0]), int(col[1]), int(col[2])), -1)
        cv2.imwrite(out_path, draw)

    def _scan_mcap_channels(self, mcap_path):
        """扫描 mcap 全部 channel：(topic, schema_name, count)。不依赖解码器。
        优先用 summary（秒级）；无 summary 才全量遍历消息计数。"""
        from collections import Counter
        cnt = Counter()
        names = {}
        with open(mcap_path, "rb") as f:
            reader = make_reader(f)
            summary = reader.get_summary()
            if summary is not None and summary.channels:
                for cid, ch in summary.channels.items():
                    t = ch.topic
                    cnt[t] = summary.statistics.channel_message_counts.get(cid, 0)
                    names[t] = (summary.schemas[ch.schema_id].name
                                if ch.schema_id in summary.schemas else "?")
            else:
                for schema, channel, _ in reader.iter_messages():
                    cnt[channel.topic] += 1
                    names[channel.topic] = schema.name
        return [(t, names[t], cnt[t]) for t in cnt]

    def _resolve_topic(self, mcap_path, cfg_topic, keywords, schema_kw):
        """精确匹配优先；为空时按关键字/schema 扫描候选并自动纠偏。
        返回 (实际 topic, 是否已自动纠偏)。"""
        chans = self._scan_mcap_channels(mcap_path)
        if any(t == cfg_topic for t, _, _ in chans):
            return cfg_topic, False
        cands = [t for t, s, _ in chans
                 if (schema_kw and schema_kw in s) or any(k in t for k in keywords)]
        logging.warning(f"topic '{cfg_topic}' 未匹配到 mcap 实际话题, 可用 channel:")
        for t, s, n in chans:
            logging.warning(f"  {t}  ({s}, {n} msgs)")
        if len(cands) == 1:
            logging.warning(f"自动采用唯一候选 '{cands[0]}'")
            return cands[0], True
        if len(cands) > 1:
            logging.warning(f"候选(按 {keywords}/{schema_kw}): {cands}, 请在 config.yaml 指定正确话题")
        return cfg_topic, False

    def _read_lidar_msgs(self, mcap_path, lidar_topic, point_field_datatype_to_np):
        out = []
        with open(mcap_path, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            for _, chan, msg, ros_msg in reader.iter_decoded_messages(topics=[lidar_topic]):
                if type(ros_msg).__name__ != "PointCloud2":
                    continue
                try:
                    ts = msg_stamp_ns(ros_msg, msg)
                    pc = pointcloud2_to_xyzi(ros_msg, point_field_datatype_to_np)
                    out.append((ts, pc))
                except Exception as e:
                    logging.error(f"lidar parse err:{e}")
        out.sort(key=lambda x: x[0])
        logging.info(f"lidar_msgs count = {len(out)}")
        return out

    def _iter_image_frames(self, mcap_path, img_topic, format_map: dict):
        video_dec = None
        n_img = 0
        with open(mcap_path, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            for _, chan, msg, ros_msg in reader.iter_decoded_messages(topics=[img_topic]):
                t_ns = msg_stamp_ns(ros_msg, msg)
                name = type(ros_msg).__name__
                # 兼容 foxglove_msgs/msg/CompressedVideo 等带前缀的消息类型
                is_compressed_video = "CompressedVideo" in name
                is_compressed_image = "CompressedImage" in name
                is_image = name == "Image"
                try:
                    if is_compressed_video:
                        if video_dec is None:
                            video_dec = CompressedVideoDecoder(ros_msg.format, format_map)
                            logging.info(f"CompressedVideo format={ros_msg.format}")
                        for ft, fimg in video_dec.decode(bytes(ros_msg.data), t_ns):
                            n_img += 1
                            yield ft, fimg
                    elif is_compressed_image:
                        img = cv2.imdecode(np.frombuffer(bytes(ros_msg.data), np.uint8), cv2.IMREAD_COLOR)
                        if img is not None:
                            n_img += 1
                            yield t_ns, img
                    elif is_image:
                        img = np.frombuffer(bytes(ros_msg.data), np.uint8).reshape(ros_msg.height, ros_msg.width, -1)
                        enc = (ros_msg.encoding or "").lower()
                        if img.shape[2] == 1:
                            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                        elif enc.startswith("rgb"):
                            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        n_img += 1
                        yield t_ns, img.copy()
                    else:
                        logging.warning(f"unsupported image msg type: {name}, skip")
                except Exception as e:
                    logging.error(f"decode image err ({name}): {e}")
        extra_err = f", video decode errors={video_dec.n_err}" if video_dec else ""
        logging.info(f"img frames count = {n_img}{extra_err}")

    def read_mcap_sync(self, mcap_path, img_topic, lidar_topic, point_field_datatype_to_np, format_map, max_frames=50, seed=0):
        # 话题解析：精确匹配优先，读不到时按 schema/关键字自动纠偏
        lidar_topic, _ = self._resolve_topic(
            mcap_path, lidar_topic, ["points", "pointcloud", "cloud"], "PointCloud2")
        img_topic, _ = self._resolve_topic(
            mcap_path, img_topic, ["image", "video"], None)
        lidar_msgs = self._read_lidar_msgs(mcap_path, lidar_topic, point_field_datatype_to_np)
        if not lidar_msgs:
            logging.warning("no lidar data, cannot sync")
            return
        # 先把所有图像读出来, 以雷达为基准找最近图像
        img_msgs = list(self._iter_image_frames(mcap_path, img_topic, format_map))
        if len(img_msgs) == 0:
            logging.error("没有读取到任何图像帧, 请检查图像话题和消息类型是否正确!")
            return
        img_ts = np.array([t for t, _ in img_msgs])
        imgs = [img for _, img in img_msgs]
        n_pair = n_drop = 0
        sync_pairs = []
        for t_lidar, pc in lidar_msgs:
            idx = int(np.argmin(np.abs(img_ts - t_lidar)))
            if abs(int(img_ts[idx]) - t_lidar) < self.max_dt_ns:
                sync_pairs.append((imgs[idx], pc))
                n_pair += 1
            else:
                n_drop += 1
        logging.info(f"synced pairs = {n_pair}, dropped = {n_drop}")
        if len(sync_pairs) > max_frames:
            rng = np.random.default_rng(seed)
            sel = rng.choice(len(sync_pairs), max_frames, replace=False)
            sel.sort()
            sync_pairs = [sync_pairs[i] for i in sel]
            logging.info(f"sampled {len(sync_pairs)} / {n_pair} frames seed={seed}")
        for img, pc in sync_pairs:
            yield img, pc

    def run(self, mcap_path, cam_topic_dict, lidar_topic, pc_roi_lim, min_sample, pcl_fmt, n_calib_frames, calib_seed, point_field_datatype_to_np, format_map):
        out_result = {}
        manual_xlim, manual_ylim, manual_zlim = pc_roi_lim
        for cam_name, img_topic in cam_topic_dict.items():
            logging.info(f"\n======== Calibrate {cam_name} ========")
            cam_out_dir = self._make_cam_dir(cam_name) if self.save_intermediate else ""
            pointcloud_dir = os.path.join(cam_out_dir, "pointcloud")
            roi_pointcloud_dir = os.path.join(cam_out_dir, "roi_pointcloud")
            if self.save_pointcloud:
                os.makedirs(pointcloud_dir, exist_ok=True)
                logging.info(f"save_pointcloud is ON, raw full point cloud save to: {pointcloud_dir}")
            if self.save_roi_pointcloud:
                os.makedirs(roi_pointcloud_dir, exist_ok=True)
                logging.info(f"save_roi_pointcloud is ON, roi cropped point cloud save to: {roi_pointcloud_dir}")
            lidar_3d_list, img_2d_list, board_poses = [], [], []
            valid_frames, viz_frames, edge_frames = [], [], []
            n_pairs = n_fail_img = n_fail_lidar = n_fail_roi = 0
            logging.info(f"ROI: x=[{manual_xlim[0]:.2f},{manual_xlim[1]:.2f}] y=[{manual_ylim[0]:.2f},{manual_ylim[1]:.2f}] z=[{manual_zlim[0]:.2f},{manual_zlim[1]:.2f}]")
            logging.info(f"detector: {'SECOND auto' if self.board_detector is not None else 'manual ROI'}")
            for frame_idx, (img, pc) in enumerate(self.read_mcap_sync(mcap_path, img_topic, lidar_topic, point_field_datatype_to_np, format_map, max_frames=n_calib_frames, seed=calib_seed)):
                n_pairs += 1
                logging.info(f"--- Frame {frame_idx}/{n_calib_frames} ---")
                if self.save_pointcloud:
                    pcd_path = os.path.join(pointcloud_dir, f"frame_{frame_idx:04d}.pcd")
                    save_point_cloud(pc, pcd_path)

                # 1. 图像 Aruco 检测
                logging.info(f"  [1/4] 检测图像 Aruco 标记...")
                holes2d, corners, ids, R_board, tvec_board = self.detect_fastcalib_holes_2d(img)
                if holes2d is not None:
                    logging.info(f"        图像检测成功: {len(holes2d)} 个孔中心")
                else:
                    logging.warning(f"        图像 Aruco 检测失败, 跳过该帧")
                    n_fail_img += 1
                    continue

                # 2. ROI 区域检测 (SECOND 或手动)
                logging.info(f"  [2/4] 确定 ROI 区域...")
                if self.board_detector is not None:
                    detect_ok, auto_roi = self.board_detector.detect(pc)
                    if detect_ok and auto_roi is not None:
                        xlim, ylim, zlim = auto_roi
                        logging.info(f"        SECOND 自动检测: x=[{xlim[0]:.2f},{xlim[1]:.2f}] y=[{ylim[0]:.2f},{ylim[1]:.2f}]")
                    else:
                        logging.warning(f"        SECOND 检测失败, 回退到手动 ROI")
                        xlim, ylim, zlim = manual_xlim, manual_ylim, manual_zlim
                else:
                    xlim, ylim, zlim = manual_xlim, manual_ylim, manual_zlim
                    logging.info(f"        手动 ROI: x=[{xlim[0]:.2f},{xlim[1]:.2f}]")

                roi_mask = ((pc[:, 0] >= xlim[0]) & (pc[:, 0] <= xlim[1])
                            & (pc[:, 1] >= ylim[0]) & (pc[:, 1] <= ylim[1])
                            & (pc[:, 2] >= zlim[0]) & (pc[:, 2] <= zlim[1]))
                pc_roi = pc[roi_mask]
                logging.info(f"        ROI 内点数: {pc_roi.shape[0]}")

                if pc_roi.shape[0] < 300:
                    logging.warning(f"        ROI 内点数太少 ({pc_roi.shape[0]} < 300), 跳过该帧")
                    n_fail_roi += 1
                    continue

                # 3. 雷达孔中心检测 (RANSAC 平面 + 圆拟合)
                logging.info(f"  [3/4] 雷达 RANSAC 平面拟合 + 孔中心提取...")
                holes3d, _ = self.detect_hole_centers_lidar(pc_roi)
                if holes3d is not None and len(holes3d) == 4:
                    logging.info(f"        雷达检测成功: 4 个孔中心")
                else:
                    logging.warning(f"        雷达孔中心检测失败, 跳过该帧")
                    n_fail_lidar += 1
                    continue

                # 4. 保存匹配对
                logging.info(f"  [4/4] 保存匹配对 (累计 {len(lidar_3d_list)+1} 对)")
                if self.save_intermediate:
                    img_draw = img.copy()
                    if corners is not None:
                        cv2.aruco.drawDetectedMarkers(img_draw, corners, ids)
                    if holes2d is not None:
                        for uv in holes2d:
                            cv2.circle(img_draw, (int(uv[0]), int(uv[1])), 6, (0, 0, 255), -1)
                    cv2.imwrite(os.path.join(cam_out_dir, "images", f"frame_{frame_idx:04d}.jpg"), img_draw)
                if self.save_roi_pointcloud:
                    pc_base = os.path.join(roi_pointcloud_dir, f"frame_{frame_idx:04d}")
                    if pcl_fmt == "bin":
                        pc_roi[:, :4].astype(np.float32).tofile(pc_base + ".bin")
                    else:
                        cols = colorize_by_holes(pc_roi[:, :3], holes3d, self.hole_radius, self.hole_palette)
                        pts, cls = [pc_roi[:, :3]], [cols]
                        if holes3d is not None:
                            pts.append(holes3d)
                            cls.append(self.hole_palette[:len(holes3d)])
                        pcd_o = o3d.geometry.PointCloud()
                        pcd_o.points = o3d.utility.Vector3dVector(np.vstack(pts).astype(np.float64))
                        pcd_o.colors = o3d.utility.Vector3dVector(np.vstack(cls).astype(np.float64))
                        o3d.io.write_point_cloud(pc_base + (".pcd" if pcl_fmt == "pcd" else ".ply"), pcd_o)

                lidar_3d_list.append(holes3d)
                img_2d_list.append(holes2d)
                board_poses.append((R_board.copy(), tvec_board.copy()))
                valid_frames.append(frame_idx)
                if len(viz_frames) < 6:
                    viz_frames.append((frame_idx, img.copy(), pc.copy()))
                if len(edge_frames) < 8:
                    ed = cv2.dilate(cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 60, 160), np.ones((3, 3), np.uint8))
                    edge_frames.append((ed, pc.copy()))

            logging.info(f"\n==== 帧处理完成 ====")
            logging.info(f"总帧数: {n_pairs}, 有效帧数: {len(lidar_3d_list)}")
            logging.info(f"失败原因: 图像={n_fail_img}, 雷达={n_fail_lidar}, ROI={n_fail_roi}")

            if len(lidar_3d_list) < min_sample:
                logging.warning(f"\n有效样本数 {len(lidar_3d_list)} < 最小要求 {min_sample}, 跳过标定!")
                logging.warning(f"请检查: 1) SECOND 检测器是否正常 2) 图像 Aruco 是否能识别 3) 雷达 ROI 内点数是否足够")
                continue
            if len(lidar_3d_list) >= 2:
                cent = np.array([h.mean(axis=0) for h in lidar_3d_list])
                spread = float(np.max(np.linalg.norm(cent - cent.mean(axis=0), axis=1)))
                logging.info(f"标定板移动范围: {spread:.3f}m (建议 > 0.2m)")
                if spread < 0.2:
                    logging.warning(f"board movement {spread:.3f}m, recommend multi‑pose capture")

            logging.info(f"\n==== 开始 LM 优化 ====")
            logging.info(f"输入: {len(lidar_3d_list)} 组 3D-2D 匹配点")
            T_opt, mean_err, _ = self.calibrate_single_cam(lidar_3d_list, img_2d_list, board_poses, edge_frames=edge_frames or None)
            logging.info(f"LM 优化完成, 平均重投影误差: {mean_err:.3f} px")
            npy_out = os.path.join(self.out_root_result, f"{cam_name}_lidar2cam_fastcalib.npy")
            yaml_out = os.path.join(self.out_root_result, f"{cam_name}_lidar2cam_fastcalib.yaml")
            np.save(npy_out, T_opt)
            save_T_yaml(yaml_out, T_opt, {"mean_reproj_err_px": float(mean_err)})
            logging.info(f"saved {npy_out} / {yaml_out}")
            if self.save_intermediate:
                for fi, h3, h2 in zip(valid_frames, lidar_3d_list, img_2d_list):
                    imgd = cv2.imread(os.path.join(cam_out_dir, "images", f"frame_{fi:04d}.jpg"))
                    if imgd is None:
                        continue
                    pc_cam = (T_opt[:3, :3] @ h3.T).T + T_opt[:3, 3]
                    uv = self.camera.project(pc_cam)
                    for q in uv:
                        p1 = (int(q[0]), int(q[1]))
                        j = int(np.argmin(np.linalg.norm(h2 - q, axis=1)))
                        p2 = (int(h2[j][0]), int(h2[j][1]))
                        cv2.circle(imgd, p1, 10, (0, 255, 0), 2)
                        cv2.line(imgd, p1, p2, (0, 255, 0), 1)
                    cv2.imwrite(os.path.join(cam_out_dir, "reproj", f"frame_{fi:04d}_reproj.jpg"), imgd)
                for fi, imgv, pcv in viz_frames:
                    self._draw_projection(imgv, pcv, T_opt, os.path.join(cam_out_dir, "proj", f"frame_{fi:04d}_proj.jpg"))
            logging.info(f"saved overlays: {len(valid_frames)} reproj + {len(viz_frames)} proj")
            out_result[cam_name] = {"T_lidar2cam": T_opt, "mean_reproj_err": mean_err}
        return out_result

    def _make_cam_dir(self, cam_name: str):
        cam_dir = os.path.join(self.out_root_intermediate, cam_name)
        os.makedirs(os.path.join(cam_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(cam_dir, "proj"), exist_ok=True)
        os.makedirs(os.path.join(cam_dir, "reproj"), exist_ok=True)
        os.makedirs(os.path.join(cam_dir, "pointcloud"), exist_ok=True)
        os.makedirs(os.path.join(cam_dir, "roi_pointcloud"), exist_ok=True)
        return cam_dir

    def visualize_with_T(self, mcap_path, img_topic, lidar_topic, pc_roi_lim, T, frame_idx, point_field_datatype_to_np, format_map, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        xlim, ylim, zlim = pc_roi_lim
        for i, (img, pc) in enumerate(self.read_mcap_sync(mcap_path, img_topic, lidar_topic, point_field_datatype_to_np, format_map)):
            if i != frame_idx:
                continue
            out = os.path.join(out_dir, f"proj_T_frame{i:04d}.jpg")
            self._draw_projection(img, pc, T, out)
            logging.info(f"saved {out}")
            h2d, _, _, _, _ = self.detect_fastcalib_holes_2d(img)
            roi = pc[(pc[:, 0] >= xlim[0]) & (pc[:, 0] <= xlim[1])
                     & (pc[:, 1] >= ylim[0]) & (pc[:, 1] <= ylim[1])
                     & (pc[:, 2] >= zlim[0]) & (pc[:, 2] <= zlim[1])]
            cents, _ = self.detect_hole_centers_lidar(roi)
            if h2d is not None and cents is not None and len(cents) == 4:
                pc_cam = (T[:3, :3] @ cents.T).T + T[:3, 3]
                uv = self.camera.project(pc_cam)
                cost = np.linalg.norm(uv[:, None] - h2d[None, :], axis=-1)
                logging.info(f"frame {i}: hole reproj err = {cost.min(axis=1).round(1)} px")
            return
        logging.warning(f"frame {frame_idx} not found")


def load_T(path):
    meta = {}
    if path.endswith(".npy"):
        T = np.load(path)
        yaml_path = path.replace(".npy", ".yaml")
        if os.path.exists(yaml_path):
            import yaml
            with open(yaml_path, "r", encoding="utf-8") as f:
                meta = yaml.safe_load(f)
    else:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        T = np.eye(4)
        T[:3, :3] = np.array(d["Rcl"], dtype=np.float64).reshape(3, 3)
        T[:3, 3] = np.array(d["Pcl"], dtype=np.float64).reshape(3)
        meta = d
    return T, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vis", default=None, help="外参npy/yaml路径，质检投影")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--config", type=str, default="config/config.yaml", help="配置yaml路径")
    args = ap.parse_args()
    cfg = load_yaml_config(args.config)
    output_root = cfg["output_dir"]
    log_file = os.path.join(output_root, "calib.log")
    os.makedirs(output_root, exist_ok=True)
    setup_logger(log_file)

    calib_board = cfg["calib_board"]
    hole_centers_board = np.array(calib_board["hole_centers_board"], dtype=np.float64)
    marker_size = calib_board["marker_size"]
    hole_radius = calib_board["hole_radius"]
    marker_centers_board = calib_board["marker_centers_board"]
    hole_palette = np.array(cfg["hole_palette"], dtype=np.float64)
    point_field_datatype_to_np = cfg["point_field_datatype_to_np"]
    format_map = cfg["format_map"]
    cam_int = cfg["camera_intrinsic"]
    K = np.array(cam_int["K"], dtype=np.float64)
    dist = np.array(cam_int["dist"], dtype=np.float64)
    cam_type = cam_int.get("type", "fisheye")
    # 根据配置实例化相机
    if cam_type == "fisheye":
        camera = FisheyeCamera(K, dist)
    elif cam_type == "pinhole":
        camera = PinholeCamera(K, dist)
    else:
        raise ValueError(f"unsupported camera type {cam_type}")
    algo_cfg = cfg["calib_algo"]
    default_pc_roi_lim = tuple(tuple(x) for x in cfg["default_pc_roi_lim"])
    data_cfg = cfg["data_config"]
    mcap_file = data_cfg["mcap_file"]
    cam_topics = data_cfg["cam_topics"]
    lidar_topic = data_cfg["lidar_topic"]

    # SECOND 检测器 (可选)
    board_detector = None
    second_cfg = cfg.get("second_detector", {})
    if second_cfg.get("enabled", False) and SECONDBoardDetector is not None:
        logging.info(f"loading SECOND detector: {second_cfg['model_path']}")
        board_detector = SECONDBoardDetector(
            second_cfg["model_path"],
            score_thr=second_cfg.get("score_thr", 0.3),
        )

    calibrator = FastCalibLidarCamCalibrator(
        hole_centers_board=hole_centers_board,
        marker_size=marker_size,
        hole_radius=hole_radius,
        marker_centers_board=marker_centers_board,
        max_dt_ns=algo_cfg["max_dt_ns"],
        plane_tol=algo_cfg["plane_tol"],
        plane_iters=algo_cfg["plane_iters"],
        board_vertical_prior=algo_cfg["board_vertical_prior"],
        camera=camera,
        hole_palette=hole_palette,
        save_intermediate=algo_cfg["save_intermediate"],
        save_pointcloud=algo_cfg["save_pointcloud"],
        save_roi_pointcloud=algo_cfg["save_roi_pointcloud"],
        output_root=output_root,
        aruco_dict_id=cv2.aruco.DICT_6X6_250,
        board_detector=board_detector
    )
    if args.vis:
        T, meta_info = load_T(args.vis)
        logging.info("==== Load calib yaml meta info ====")
        logging.info(meta_info)
        vis_out_dir = os.path.join(output_root, "vis_overlays")
        cam_topic = list(cam_topics.values())[0]
        calibrator.visualize_with_T(
            mcap_file,
            cam_topic,
            lidar_topic,
            default_pc_roi_lim,
            T,
            args.frame,
            point_field_datatype_to_np,
            format_map,
            out_dir=vis_out_dir
        )
    else:
        calibrator.run(
            mcap_path=mcap_file,
            cam_topic_dict=cam_topics,
            lidar_topic=lidar_topic,
            pc_roi_lim=default_pc_roi_lim,
            min_sample=algo_cfg["min_sample"],
            pcl_fmt=algo_cfg["pcl_fmt"],
            n_calib_frames=algo_cfg["n_calib_frames"],
            calib_seed=algo_cfg["calib_seed"],
            point_field_datatype_to_np=point_field_datatype_to_np,
            format_map=format_map
        )
    logging.info("\n===== Calibration done =====")
