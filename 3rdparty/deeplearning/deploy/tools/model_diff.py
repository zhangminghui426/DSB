#!/usr/bin/env python
"""三端模型对比：torch（训练接口）<-> onnxruntime <-> TensorRT，逐模块对拍。

目标：定位 C++ 部署（TRT plan）与训练模型（pth）的差异环节。链：
  img+points+矩阵 -> camera.backbone(feature+depth_weights) -> bev_pool(bev360)
  -> downsample(cam_bev) + sparse(lidar_bev) -> fuser(middle) -> head.bbox(score/rot/...)
  -> decode(3D 框)

后端：
  torch  训练接口全链（build_dataset + collate_fn，sample 与 example-data 同帧），dump 每模块张量
  onnx   5 个 onnx（CUDA-BEVFusion 结构）用 onnxruntime 逐模块跑（输入用 torch 上游 dump）
  trt    4 个 plan（C++ 部署引擎）用 tensorrt python 逐模块跑（输入同 onnx）
  diff   三端逐模块 max-abs / rel 对比 + 框对比

用法（3rdparty/deeplearning 下，先跑 torch 再 onnx/trt/diff；或 --backend all）：
  python deploy/tools/model_diff.py configs/det_convfuser.py ckpts/bevfusion-det-bf.pth \
      --backend all --onnx-dir outputs/onnx_bf --engine-dir deploy/cpp/model/resnet50/build \
      --dump-dir outputs/model_diff --sample-index 0
"""
import argparse
import os

import numpy as np
import torch


def load_tensor(p):
    with open(p, "rb") as f:
        h = np.fromfile(f, dtype="<i4", count=3)
        ndim, dtypei = int(h[1]), int(h[2])
        dims = tuple(int(v) for v in np.fromfile(f, dtype="<i4", count=ndim))
        dt = {1: "<i4", 2: "<f2", 3: "<f4", 4: "<i8",
              5: "<u8", 6: "<u4", 7: "<i1", 8: "<u1"}[dtypei]
        return np.fromfile(f, dtype=dt).reshape(dims)


def dump(name, arr, dump_dir, sub):
    d = os.path.join(dump_dir, sub)
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, name + ".npy"), np.asarray(arr))
    print(f"[dump] {sub}/{name}.npy {np.asarray(arr).shape}")


def load_dump(name, dump_dir, sub="torch"):
    return np.load(os.path.join(dump_dir, sub, name + ".npy"))


# ---------------------------------------------------------------- torch 后端
def project_depth(batch, vt):
    """BaseDepthTransform 前段：点云 -> 稀疏深度图 [B,N,1,H,W]（与训练推理一致）。"""
    img = batch["img"]
    B, N = img.shape[0], img.shape[1]
    H, W = vt.image_size
    depth = torch.zeros(B, N, 1, H, W, device=img.device)
    for b in range(B):
        cur = batch["points"][b][:, :3].float().clone()
        aug = batch["lidar_aug_matrix"][b].float()
        l2i = batch["lidar2image"][b].float()
        iaug = batch["img_aug_matrix"][b].float()
        cur -= aug[:3, 3]
        cur = torch.inverse(aug[:3, :3]).matmul(cur.transpose(1, 0))
        cur = l2i[:, :3, :3].matmul(cur)
        cur += l2i[:, :3, 3].reshape(-1, 3, 1)
        dist = cur[:, 2, :]
        cur[:, 2, :] = torch.clamp(cur[:, 2, :], 1e-5, 1e5)
        cur[:, :2, :] /= cur[:, 2:3, :]
        cur = iaug[:, :3, :3].matmul(cur)
        cur += iaug[:, :3, 3].reshape(-1, 3, 1)
        cur = cur[:, :2, :].transpose(1, 2)[..., [1, 0]]
        on_img = ((cur[..., 0] < H) & (cur[..., 0] >= 0) &
                  (cur[..., 1] < W) & (cur[..., 1] >= 0))
        for c in range(N):
            m = on_img[c]
            depth[b, c, 0, cur[c, m, 0].long(), cur[c, m, 1].long()] = dist[c, m]
    return depth


def backend_torch(cfg, ckpt, dump_dir, sample_index=0):
    from scripts.config import load_config
    from models import build_model
    from scripts.trainer import load_model_checkpoint, to_device
    from datasets import build_dataset, collate_fn
    from deploy.export_bf import CamBackboneBf, DownsampleBf, HeadBboxBf
    from deploy.export import CamVtransformDet, FuserDecoder

    cfg = load_config(cfg)
    model = build_model(cfg.model).cuda().eval()
    msg, _ = load_model_checkpoint(model, ckpt, strict=True)
    print("loaded %s missing=%d unexpected=%d" % (ckpt, len(msg.missing_keys), len(msg.unexpected_keys)))

    ds = build_dataset(cfg.data.val)
    batch = to_device(collate_fn([ds[sample_index]]), "cuda")

    vt = model.encoders["camera"]["vtransform"]
    img = batch["img"]
    B, N, C, H, W = img.shape
    half = lambda t: t.detach().cpu().half().numpy()

    # 1. depth 投影 + camera.backbone（feature HWC + depth_weights）
    depth = project_depth(batch, vt)
    dump("img", half(img), dump_dir, "torch")
    dump("depth", half(depth), dump_dir, "torch")
    cam = CamBackboneBf(model)
    feature, depth_weights = cam(img, depth)  # (B,N,fH,fW,C) / (B,N,D,fH,fW)
    dump("camera_feature", half(feature), dump_dir, "torch")
    dump("camera_depth_weights", half(depth_weights), dump_dir, "torch")

    # 2. bev_pool -> bev360
    bb = model.encoders["camera"]["backbone"]
    neck = model.encoders["camera"]["neck"]
    x = bb(img.view(B * N, C, H, W))
    x = neck(x)
    x = x[0] if not torch.is_tensor(x) else x
    feats = x.view(B, N, *x.shape[1:])
    depthnet_out = CamVtransformDet(model)(feats, depth)  # (B*N, D+C, fH, fW)
    D, CC = vt.D, vt.C
    dw = depthnet_out[:, :D].softmax(1).view(B, N, D, *feats.shape[3:])
    fc = depthnet_out[:, D:].view(B, N, CC, *feats.shape[3:])
    xpool = (dw.unsqueeze(2) * fc.unsqueeze(3)).permute(0, 1, 3, 4, 5, 2).contiguous()
    geom = vt.get_geometry(
        batch["camera2lidar"][..., :3, :3], batch["camera2lidar"][..., :3, 3],
        batch["camera_intrinsics"][..., :3, :3],
        batch["img_aug_matrix"][..., :3, :3], batch["img_aug_matrix"][..., :3, 3],
        extra_rots=batch["lidar_aug_matrix"][..., :3, :3],
        extra_trans=batch["lidar_aug_matrix"][..., :3, 3],
    )
    bev360 = vt.bev_pool(geom, xpool)  # (B, C, 360, 360)
    dump("bev360", half(bev360), dump_dir, "torch")
    cam_bev = DownsampleBf(model)(bev360)  # (B, C, 180, 180)
    dump("cam_bev", half(cam_bev), dump_dir, "torch")

    # 3. lidar sparse
    feats_l, coords, sizes = model.voxelize(batch["points"], "lidar")
    batch_size = coords[-1, 0] + 1
    lidar_bev = model.encoders["lidar"]["backbone"](feats_l, coords, batch_size, sizes=sizes)
    dump("lidar_bev", half(lidar_bev), dump_dir, "torch")

    # 4. fuser + decoder
    middle = FuserDecoder(model)(cam_bev, lidar_bev)  # (B,512,180,180)
    dump("middle", half(middle), dump_dir, "torch")

    # 5. head.bbox 输出（decode 留在 C++/get_bboxes）
    head = model.heads["object"]
    head.onnx_export = True
    score, rot, dim, reg, height, vel = HeadBboxBf(model)(middle)
    for name, t in [("score", score), ("rot", rot), ("dim", dim),
                    ("reg", reg), ("height", height), ("vel", vel)]:
        dump(name, half(t), dump_dir, "torch")

    # 6. decode 3D 框（torch 训练侧 decode，与 C++ head-transbbox 对比）
    pred_dicts = head([middle], [{}])
    bboxes = head.get_bboxes(pred_dicts, batch["metas"])
    b = bboxes[0][0].tensor.detach().cpu().numpy()
    s = bboxes[0][1].detach().cpu().numpy()
    l = bboxes[0][2].detach().cpu().numpy()
    print(f"[torch] {len(b)} boxes")
    for i in range(len(b)):
        print("  box[%d] id=%d score=%.3f pos=(%.2f,%.2f,%.2f) size=(%.2f,%.2f,%.2f) rot=%.3f"
              % (i, int(l[i]), s[i], b[i, 0], b[i, 1], b[i, 2],
                 b[i, 3], b[i, 4], b[i, 5], b[i, 6]))
    with open(os.path.join(dump_dir, "torch", "boxes.txt"), "w") as f:
        for i in range(len(b)):
            f.write("box[%d] id=%d score=%.3f pos=(%.2f,%.2f,%.2f) size=(%.2f,%.2f,%.2f) rot=%.3f\n"
                    % (i, int(l[i]), s[i], b[i, 0], b[i, 1], b[i, 2],
                       b[i, 3], b[i, 4], b[i, 5], b[i, 6]))


# ---------------------------------------------------------------- onnx 后端
def backend_onnx(onnx_dir, dump_dir):
    import onnxruntime as ort
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    feeds = {
        "camera.backbone": {"img": "img", "depth": "depth"},
        "camera.vtransform": {"feat_in": "bev360"},
        "fuser": {"camera": "cam_bev", "lidar": "lidar_bev"},
        "head.bbox": {"middle": "middle"},
    }
    outs = {
        "camera.backbone": ["camera_feature", "camera_depth_weights"],
        "camera.vtransform": ["cam_bev"],
        "fuser": ["middle"],
        "head.bbox": ["score", "rot", "dim", "reg", "height", "vel"],
    }
    for graph, feed_map in feeds.items():
        path = os.path.join(onnx_dir, graph + ".onnx")
        if not os.path.exists(path):
            print(f"[onnx] SKIP {graph}: {path} missing")
            continue
        sess = ort.InferenceSession(path, providers=providers)
        in_names = {i.name for i in sess.get_inputs()}
        feed = {k: load_dump(v, dump_dir).astype(np.float16) for k, v in feed_map.items()}
        res = sess.run(None, feed)
        names = [o.name for o in sess.get_outputs()]
        print(f"[onnx] {graph} in={list(feed)} out={names}")
        for name, arr in zip(names, res):
            out_key = outs[graph][0] if name in outs[graph] else None
            # 按 onnx 输出名->我们命名约定（部分图输出顺序即约定顺序）
        for j, arr in enumerate(res):
            key = outs[graph][j] if j < len(outs[graph]) else names[j]
            dump(key, arr, dump_dir, "onnx")


# ---------------------------------------------------------------- trt 后端
def _trt_session(engine_dir, graph):
    import tensorrt as trt
    rt = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    with open(os.path.join(engine_dir, graph + ".plan"), "rb") as f:
        eng = rt.deserialize_cuda_engine(f.read())
    ctx = eng.create_execution_context()
    names = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors)]
    in_names = [n for n in names if eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
    out_names = [n for n in names if eng.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
    return ctx, in_names, out_names


def backend_trt(engine_dir, dump_dir):
    import tensorrt as trt
    graphs = ["camera.backbone", "camera.vtransform", "fuser", "head.bbox"]
    feed_map = {
        "camera.backbone": {"img": "img", "depth": "depth"},
        "camera.vtransform": {"feat_in": "bev360"},
        "fuser": {"camera": "cam_bev", "lidar": "lidar_bev"},
        "head.bbox": {"middle": "middle"},
    }
    out_map = {
        "camera.backbone": ["camera_feature", "camera_depth_weights"],
        "camera.vtransform": ["cam_bev"],
        "fuser": ["middle"],
        "head.bbox": ["score", "rot", "dim", "reg", "height", "vel"],
    }
    stream = torch.cuda.current_stream().cuda_stream
    for graph in graphs:
        path = os.path.join(engine_dir, graph + ".plan")
        if not os.path.exists(path):
            print(f"[trt] SKIP {graph}: {path} missing")
            continue
        ctx, in_names, out_names = _trt_session(engine_dir, graph)
        buffers = {}
        for name in in_names:
            feed_key = next(k for k, v in feed_map[graph].items() if v == name) \
                if name in feed_map[graph].values() else name
            src = feed_map[graph].get(feed_key, feed_key)
            arr = load_dump(src, dump_dir).astype(np.float16)
            t = torch.from_numpy(np.ascontiguousarray(arr)).cuda()
            buffers[name] = t
            ctx.set_tensor_address(name, t.data_ptr())
        for name in out_names:
            shape = tuple(ctx.get_tensor_shape(name))
            t = torch.empty(shape, dtype=torch.float16, device="cuda")
            buffers[name] = t
            ctx.set_tensor_address(name, t.data_ptr())
        ok = ctx.execute_async_v3(stream)
        torch.cuda.synchronize()
        if not ok:
            print(f"[trt] {graph} execute failed"); continue
        print(f"[trt] {graph} in={in_names} out={out_names}")
        for j, name in enumerate(out_names):
            key = out_map[graph][j] if j < len(out_map[graph]) else name
            dump(key, buffers[name].cpu().numpy(), dump_dir, "trt")


# ---------------------------------------------------------------- diff
def _stats(a, b, name):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    if a.shape != b.shape:
        return f"  {name}: shape {a.shape} vs {b.shape} MISMATCH"
    diff = np.abs(a - b)
    rel = diff / (np.abs(a) + 1e-6)
    return ("  %s: max_abs=%.6g mean_abs=%.6g max_rel=%.6g"
            % (name, diff.max(), diff.mean(), rel.max()))


def backend_diff(dump_dir):
    tensors = ["camera_feature", "camera_depth_weights", "cam_bev",
               "middle", "score", "rot", "dim", "reg", "height", "vel"]
    print("== torch vs onnx（onnx 转换损失）==")
    for t in tensors:
        try:
            a = load_dump(t, dump_dir, "torch"); b = load_dump(t, dump_dir, "onnx")
            print(_stats(a, b, t))
        except FileNotFoundError:
            print(f"  {t}: onnx dump 缺失")
    print("== torch vs trt（总部署损失）==")
    for t in tensors:
        try:
            a = load_dump(t, dump_dir, "torch"); b = load_dump(t, dump_dir, "trt")
            print(_stats(a, b, t))
        except FileNotFoundError:
            print(f"  {t}: trt dump 缺失")
    print("== onnx vs trt（TRT 构建/执行损失）==")
    for t in tensors:
        try:
            a = load_dump(t, dump_dir, "onnx"); b = load_dump(t, dump_dir, "trt")
            print(_stats(a, b, t))
        except FileNotFoundError:
            print(f"  {t}: onnx/trt dump 缺失")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", nargs="?")
    ap.add_argument("ckpt", nargs="?")
    ap.add_argument("--backend", default="all", choices=["torch", "onnx", "trt", "diff", "all"])
    ap.add_argument("--onnx-dir", default="outputs/onnx_bf")
    ap.add_argument("--engine-dir", default="deploy/cpp/model/resnet50/build")
    ap.add_argument("--dump-dir", default="outputs/model_diff")
    ap.add_argument("--sample-index", type=int, default=0)
    args = ap.parse_args()

    if args.backend in ("torch", "all"):
        assert args.config and args.ckpt, "torch 后端需要 config + ckpt"
        backend_torch(args.config, args.ckpt, args.dump_dir, args.sample_index)
    if args.backend in ("onnx", "all"):
        backend_onnx(args.onnx_dir, args.dump_dir)
    if args.backend in ("trt", "all"):
        backend_trt(args.engine_dir, args.dump_dir)
    if args.backend in ("diff", "all"):
        backend_diff(args.dump_dir)


if __name__ == "__main__":
    main()
