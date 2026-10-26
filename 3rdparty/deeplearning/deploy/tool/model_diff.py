"""torch / onnxruntime / TensorRT 三端对比：逐模块精度、延时、可视化。

用法（3rdparty/deeplearning 下）：
  python deploy/tool/model_diff.py configs/det_convfuser.py ckpts/bevfusion-det-bf.pth \
      --backend all --dump-dir outputs/model_diff

  --backend torch   训练接口全链，落盘各模块张量/框
  --backend onnx    onnxruntime 跑 4 张图（输入用 torch dump）
  --backend trt     TensorRT 跑 4 个 plan（输入同上）
  --backend bench   torch/onnx/trt 逐模块延时 -> benchmark.json
  --backend diff    打印精度对比表
  --backend vis     框叠加图（BEV + 6 相机）
  --backend report  延时/精度对比图 -> report.png

产物（--dump-dir）：{torch,onnx,trt}/*.npy、benchmark.json、report.png、
                    onnx_vis.jpg、trt_vis.jpg
"""

import argparse
import json
import os
import os.path as osp
import sys
import time

import numpy as np
import torch

_HERE = osp.dirname(osp.abspath(__file__))
_DL_ROOT = osp.dirname(osp.dirname(_HERE))  # 3rdparty/deeplearning
for path in (_DL_ROOT, _HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

from common import load_tensor, parse_cpp_boxes, save_tensor  # noqa: E402
from deploy.tool.onnx_stage import (  # noqa: E402
    CamBackboneBf, CamVtransformDet, DownsampleBf, FuserDecoder, HeadBboxBf)

# 4 个可部署图的输入/输出 dump 名（与 export.py 的 BF_IO 对应）
FEEDS = {
    "camera.backbone": {"img": "img", "depth": "depth"},
    "camera.vtransform": {"feat_in": "bev360"},
    "fuser": {"camera": "cam_bev", "lidar": "lidar_bev"},
    "head.bbox": {"middle": "middle"},
}
OUTS = {
    "camera.backbone": ["camera_feature", "camera_depth_weights"],
    "camera.vtransform": ["cam_bev"],
    "fuser": ["middle"],
    "head.bbox": ["score", "rot", "dim", "reg", "height", "vel"],
}
GRAPHS = list(FEEDS)
STAGE_TENSORS = {
    "camera.backbone": ["camera_feature", "camera_depth_weights"],
    "camera.vtransform": ["cam_bev"],
    "fuser": ["middle"],
    "head.bbox": ["score", "rot", "dim", "reg", "height", "vel"],
}
CAM_NAMES = ["FRONT", "FRONT_RIGHT", "FRONT_LEFT", "BACK", "BACK_LEFT", "BACK_RIGHT"]


def dump(name, arr, dump_dir, sub):
    os.makedirs(osp.join(dump_dir, sub), exist_ok=True)
    np.save(osp.join(dump_dir, sub, name + ".npy"), np.asarray(arr))


def load_dump(name, dump_dir, sub="torch"):
    return np.load(osp.join(dump_dir, sub, name + ".npy"))


def load_model_only(cfg, ckpt):
    from models import build_model
    from scripts.config import load_config
    from scripts.trainer import load_model_checkpoint
    cfg = load_config(cfg)
    model = build_model(cfg.model).cuda().eval()
    load_model_checkpoint(model, ckpt, strict=True)
    return model


# ---------------------------------------------------------------- torch
def project_depth(batch, vt):
    """点云 -> 稀疏深度图 (B,N,1,H,W)。"""
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


def backend_torch(cfg, ckpt, dump_dir, sample_index=0, cpp_dir="outputs/model_diff/cpp"):
    from datasets import build_dataset, collate_fn
    from models import build_model
    from scripts.config import load_config
    from scripts.trainer import load_model_checkpoint, to_device

    cfg = load_config(cfg)
    model = build_model(cfg.model).cuda().eval()
    load_model_checkpoint(model, ckpt, strict=True)

    batch = to_device(collate_fn([build_dataset(cfg.data.val)[sample_index]]), "cuda")
    vt = model.encoders["camera"]["vtransform"]
    img = batch["img"]
    B, N, C, H, W = img.shape
    np_dump = lambda t: t.detach().cpu().numpy()

    # camera.backbone -> bev_pool/downsample -> lidar sparse -> fuser -> head
    depth = project_depth(batch, vt)
    dump("img", np_dump(img), dump_dir, "torch")
    dump("depth", np_dump(depth), dump_dir, "torch")
    feature, depth_weights = CamBackboneBf(model)(img, depth)
    dump("camera_feature", np_dump(feature), dump_dir, "torch")
    dump("camera_depth_weights", np_dump(depth_weights), dump_dir, "torch")

    bb, neck = model.encoders["camera"]["backbone"], model.encoders["camera"]["neck"]
    x = neck(bb(img.view(B * N, C, H, W)))
    x = x[0] if not torch.is_tensor(x) else x
    feats = x.view(B, N, *x.shape[1:])
    depthnet_out = CamVtransformDet(model)(feats, depth)
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
    bev360 = vt.bev_pool(geom, xpool)
    dump("bev360", np_dump(bev360), dump_dir, "torch")
    cam_bev = DownsampleBf(model)(bev360)
    dump("cam_bev", np_dump(cam_bev), dump_dir, "torch")

    feats_l, coords, sizes = model.voxelize(batch["points"], "lidar")
    lidar_bev = model.encoders["lidar"]["backbone"](
        feats_l, coords, coords[-1, 0] + 1, sizes=sizes)
    dump("lidar_bev", np_dump(lidar_bev), dump_dir, "torch")
    save_tensor(coords.detach().cpu().numpy().astype(np.int32),
                osp.join(cpp_dir, "torch_lidar_coords"))
    save_tensor(feats_l.detach().cpu().numpy().astype(np.float16),
                osp.join(cpp_dir, "torch_lidar_feats"))

    middle = FuserDecoder(model)(cam_bev, lidar_bev)
    dump("middle", np_dump(middle), dump_dir, "torch")

    head = model.heads["object"]
    head.onnx_export = True
    for name, t in zip(["score", "rot", "dim", "reg", "height", "vel"],
                       HeadBboxBf(model)(middle)):
        dump(name, np_dump(t), dump_dir, "torch")

    pred_dicts = head([middle], [{}])
    bboxes = head.get_bboxes(pred_dicts, batch["metas"])
    boxes = bboxes[0][0].tensor.detach().cpu().numpy()
    scores = bboxes[0][1].detach().cpu().numpy()
    labels = bboxes[0][2].detach().cpu().numpy()
    dump("query_labels", head.query_labels.detach().cpu().numpy(), dump_dir, "torch")
    dump("boxes", np.concatenate([boxes, scores[:, None], labels[:, None]], axis=1),
         dump_dir, "torch")
    print(f"[torch] dumps -> {dump_dir}/torch, {len(scores)} boxes")
    return model


# ---------------------------------------------------------------- onnx / trt
def backend_onnx(onnx_dir, dump_dir, model=None):
    import onnxruntime as ort

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    for graph in GRAPHS:
        path = osp.join(onnx_dir, graph + ".onnx")
        if not osp.exists(path):
            print(f"[onnx] skip {graph}: {path} missing")
            continue
        sess = ort.InferenceSession(path, providers=providers)
        feed = {k: load_dump(v, dump_dir).astype(np.float32) for k, v in FEEDS[graph].items()}
        outs = sess.run(None, feed)
        for key, arr in zip(OUTS[graph], outs):
            dump(key, np.asarray(arr).astype(np.float32), dump_dir, "onnx")
        print(f"[onnx] {graph} ok")
    _decode_boxes(dump_dir, "onnx", model)


def _trt_plan_path(engine_dir, graph, head_plan=None):
    if graph != "head.bbox":
        return osp.join(engine_dir, graph + ".plan")
    if head_plan:
        return osp.join(engine_dir, head_plan)
    for name in ("head.bbox.fp32.plan", "head.bbox.layernormplugin.plan", "head.bbox.plan"):
        path = osp.join(engine_dir, name)
        if osp.exists(path):
            return path
    return osp.join(engine_dir, "head.bbox.plan")


def _trt_session(engine_dir, graph, head_plan=None):
    import tensorrt as trt

    path = graph if osp.isabs(graph) or "/" in graph else _trt_plan_path(engine_dir, graph, head_plan)
    rt = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    with open(path, "rb") as f:
        eng = rt.deserialize_cuda_engine(f.read())
    ctx = eng.create_execution_context()
    names = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors)]
    ins = [n for n in names if eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
    outs = [n for n in names if eng.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
    return ctx, ins, outs


def _bind_trt(ctx, in_names, out_names, graph, dump_dir):
    """按 dump 分配 TRT IO buffer，返回 {name: tensor}。"""
    buffers = {}
    for name in in_names:
        arr = load_dump(FEEDS[graph][name], dump_dir).astype(np.float16)
        t = torch.from_numpy(np.ascontiguousarray(arr)).cuda()
        buffers[name] = t
        ctx.set_tensor_address(name, t.data_ptr())
    for name in out_names:
        t = torch.empty(tuple(ctx.get_tensor_shape(name)), dtype=torch.float16, device="cuda")
        buffers[name] = t
        ctx.set_tensor_address(name, t.data_ptr())
    return buffers


def backend_trt(engine_dir, dump_dir, model=None, head_plan=None):
    stream = torch.cuda.current_stream().cuda_stream
    for graph in GRAPHS:
        path = _trt_plan_path(engine_dir, graph, head_plan)
        if not osp.exists(path):
            print(f"[trt] skip {graph}: {path} missing")
            continue
        ctx, in_names, out_names = _trt_session(engine_dir, path)
        buffers = _bind_trt(ctx, in_names, out_names, graph, dump_dir)
        if not ctx.execute_async_v3(stream):
            print(f"[trt] {graph} execute failed")
            continue
        torch.cuda.synchronize()
        for key, name in zip(OUTS[graph], out_names):
            dump(key, buffers[name].cpu().numpy(), dump_dir, "trt")
        print(f"[trt] {graph} ok")
    _decode_boxes(dump_dir, "trt", model)


def _decode_boxes(dump_dir, sub, model=None):
    """head 输出 -> 3D 框（filter=False + score>0.12，与 C++ 对齐）。"""
    if model is None or not osp.exists(osp.join(dump_dir, sub, "score.npy")):
        return
    head = model.heads["object"]
    t = lambda k: torch.from_numpy(load_dump(k, dump_dir, sub)).float().cuda()
    preds = head.bbox_coder.decode(t("score"), t("rot"), t("dim"), t("reg"),
                                   t("height"), t("vel"), filter=False)
    boxes = preds[0]["bboxes"].detach().cpu().numpy()
    scores = preds[0]["scores"].detach().cpu().numpy()
    labels = preds[0]["labels"].detach().cpu().numpy()
    mask = scores > 0.12
    dump("boxes", np.concatenate([boxes[mask], scores[mask, None], labels[mask, None]], axis=1),
         dump_dir, sub)
    print(f"[{sub}] {int(mask.sum())} boxes (score>0.12)")


# ---------------------------------------------------------------- bench
def _time_cuda(fn, warmup, iters):
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def bench_torch(model, dump_dir, iters, warmup):
    """4 个部署模块的 fp32 延时（ms）。"""
    tensors = {k: torch.from_numpy(load_dump(k, dump_dir, "torch")).cuda()
               for k in ("img", "depth", "bev360", "cam_bev", "lidar_bev", "middle")}
    cam = CamBackboneBf(model)
    down = DownsampleBf(model)
    fuser = FuserDecoder(model)
    head = HeadBboxBf(model)
    calls = {
        "camera.backbone": lambda: cam(tensors["img"], tensors["depth"]),
        "camera.vtransform": lambda: down(tensors["bev360"]),
        "fuser": lambda: fuser(tensors["cam_bev"], tensors["lidar_bev"]),
        "head.bbox": lambda: head(tensors["middle"]),
    }
    return {g: _time_cuda(calls[g], warmup, iters) for g in GRAPHS}


def bench_onnx(onnx_dir, dump_dir, iters, warmup):
    import onnxruntime as ort

    result = {}
    for graph in GRAPHS:
        path = osp.join(onnx_dir, graph + ".onnx")
        if not osp.exists(path):
            continue
        sess = ort.InferenceSession(path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        feed = {k: np.ascontiguousarray(load_dump(v, dump_dir).astype(np.float32))
                for k, v in FEEDS[graph].items()}
        for _ in range(warmup):
            sess.run(None, feed)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            sess.run(None, feed)
        torch.cuda.synchronize()
        result[graph] = (time.perf_counter() - t0) * 1000.0 / iters
    return result


def bench_trt(engine_dir, dump_dir, iters, warmup, head_plan=None):
    stream = torch.cuda.current_stream().cuda_stream
    result = {}
    for graph in GRAPHS:
        path = _trt_plan_path(engine_dir, graph, head_plan)
        if not osp.exists(path):
            continue
        ctx, in_names, out_names = _trt_session(engine_dir, path)
        _bind_trt(ctx, in_names, out_names, graph, dump_dir)
        run = lambda: ctx.execute_async_v3(stream)
        for _ in range(warmup):
            run()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            run()
        end.record()
        torch.cuda.synchronize()
        result[graph] = start.elapsed_time(end) / iters
    return result


def backend_bench(args, model):
    """torch/onnx/trt 逐模块延时，写 benchmark.json 并打印表格。"""
    data = {"iters": args.bench_iters, "warmup": args.bench_warmup, "graphs": GRAPHS,
            "backends": {}}
    data["backends"]["torch"] = bench_torch(model, args.dump_dir, args.bench_iters, args.bench_warmup)
    data["backends"]["onnx"] = bench_onnx(args.onnx_dir, args.dump_dir,
                                          args.bench_iters, args.bench_warmup)
    data["backends"]["trt"] = bench_trt(args.engine_dir, args.dump_dir,
                                        args.bench_iters, args.bench_warmup, args.head_plan)
    data["e2e"] = {k: sum(v.values()) for k, v in data["backends"].items() if v}

    with open(osp.join(args.dump_dir, "benchmark.json"), "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n[bench] ms (iters={data['iters']}, warmup={data['warmup']})")
    print(f"{'module':<18}{'torch':>10}{'onnx':>10}{'trt':>10}")
    for g in GRAPHS:
        row = [data["backends"][b].get(g) for b in ("torch", "onnx", "trt")]
        print(f"{g:<18}" + "".join(f"{v:>10.2f}" if v else f"{'-':>10}" for v in row))
    print(f"{'end2end':<18}" + "".join(
        f"{data['e2e'].get(b, 0):>10.2f}" for b in ("torch", "onnx", "trt")))
    return data


# ---------------------------------------------------------------- acc / vis
def _stage_stats(dump_dir, backend, ref="torch"):
    """每个部署阶段 vs torch 的 max/mean abs 误差条。"""
    stats = {}
    for stage, names in STAGE_TENSORS.items():
        max_abs = 0.0
        total, count = 0.0, 0
        for name in names:
            pa, pb = osp.join(dump_dir, ref, name + ".npy"), osp.join(dump_dir, backend, name + ".npy")
            if not (osp.exists(pa) and osp.exists(pb)):
                continue
            a = np.load(pa).astype(np.float32)
            b = np.load(pb).astype(np.float32)
            if a.shape != b.shape:
                continue
            d = np.abs(a - b)
            max_abs = max(max_abs, float(d.max()))
            total += float(d.sum())
            count += d.size
        if count:
            stats[stage] = {"max_abs": max_abs, "mean_abs": total / count}
    return stats


def backend_diff(dump_dir):
    """打印精度对比表（逐张量）。"""
    tensors = ["camera_feature", "camera_depth_weights", "cam_bev", "middle",
               "score", "rot", "dim", "reg", "height", "vel"]
    for ref, cur, title in [("torch", "onnx", "torch vs onnx"), ("torch", "trt", "torch vs trt")]:
        print(f"== {title} ==")
        for t in tensors:
            pa, pb = osp.join(dump_dir, ref, t + ".npy"), osp.join(dump_dir, cur, t + ".npy")
            if not (osp.exists(pa) and osp.exists(pb)):
                continue
            a, b = np.load(pa).astype(np.float32), np.load(pb).astype(np.float32)
            if a.shape != b.shape:
                print(f"  {t}: shape {a.shape} vs {b.shape} MISMATCH")
                continue
            d = np.abs(a - b)
            print(f"  {t}: max_abs={d.max():.4g} mean_abs={d.mean():.4g}")


def _box_stats(dump_dir):
    """各后端检测框数量 + 到 torch 最近框距离（score>=0.12）。"""
    def load(sub):
        path = osp.join(dump_dir, sub, "boxes.npy")
        if not osp.exists(path):
            return None
        b = np.load(path)
        if b.shape[1] >= 11:
            return b[:, :3], b[:, 9]
        return b[:, :3], np.load(osp.join(dump_dir, sub, "scores.npy"))

    ref = load("torch")
    if ref is None:
        return {}
    rp, rs = ref
    rp = rp[rs > 0.12]
    out = {"torch": {"count": int(len(rp))}}
    for sub in ("onnx", "trt"):
        cur = load(sub)
        if cur is None:
            continue
        p, s = cur
        p = p[s > 0.12]
        if len(p) == 0 or len(rp) == 0:
            out[sub] = {"count": int(len(p))}
            continue
        d = np.linalg.norm(p[:, None, :] - rp[None, :, :], axis=2).min(axis=1)
        out[sub] = {"count": int(len(p)), "mean_m": float(d.mean()), "max_m": float(d.max())}
    return out


def _visible(boxes, thr=0.12):
    """boxes.npy 第 9 列为 score 时按阈值过滤，避免画出全部 query。"""
    return boxes[boxes[:, 9] >= thr] if boxes.shape[1] >= 11 else boxes


def _plot_vis(out, dump_dir, data_dir, subs, extra_sets=()):
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sets = [(arr, color, label) for arr, color, label in extra_sets]
    for sub, color, label in subs:
        path = osp.join(dump_dir, sub, "boxes.npy")
        if osp.exists(path):
            sets.append((_visible(np.load(path)), color, label))

    l2i = load_tensor(osp.join(data_dir, "lidar2image.tensor"))
    pts = load_tensor(osp.join(data_dir, "points.tensor"))

    fig, axes = plt.subplots(3, 3, figsize=(26, 15))
    ax = axes[0][0]
    if pts.ndim >= 2 and pts.shape[1] >= 3:
        sel = pts[::5, :3]
        ax.scatter(sel[:, 0], -sel[:, 1], s=0.3, c="lightgray", alpha=0.4)
    for arr, color, _ in sets:
        for b in arr:
            x, y, _, w, l, _, rot = b[:7]
            c, s = np.cos(rot), np.sin(rot)
            rect = np.array([[-w / 2, -l / 2], [w / 2, -l / 2], [w / 2, l / 2],
                             [-w / 2, l / 2], [-w / 2, -l / 2]]) @ np.array([[c, -s], [s, c]]).T
            rect += np.array([x, y])
            ax.plot(rect[:, 0], -rect[:, 1], color=color, lw=1.2)
    ax.set_xlim(-54, 54)
    ax.set_ylim(-54, 54)
    ax.set_aspect("equal")
    ax.set_title("BEV " + " / ".join(f"{label}({color})" for _, color, label in sets))
    ax.grid(True, alpha=0.3)

    cam_axes = [axes[0][1], axes[0][2], axes[1][0], axes[1][1], axes[1][2], axes[2][0]]
    for c, axc in enumerate(cam_axes):
        img = cv2.imread(osp.join(data_dir, f"{c}-{CAM_NAMES[c]}.jpg"))
        if img is None:
            continue
        height, width = img.shape[:2]
        axc.imshow(img[:, :, ::-1])
        for arr, color, _ in sets:
            for b in arr:
                x, y, z, w, l, h, rot = b[:7]  # z 为框中心（nuScenes/预测同约定）
                cc, ss = np.cos(rot), np.sin(rot)
                base = np.array([[w / 2, l / 2], [w / 2, -l / 2],
                                 [-w / 2, -l / 2], [-w / 2, l / 2]]) @ np.array([[cc, -ss], [ss, cc]]).T
                base += np.array([x, y])
                corners = np.concatenate([np.c_[base, np.full(4, z - h / 2)],
                                          np.c_[base, np.full(4, z + h / 2)]], axis=0)
                uv = np.c_[corners, np.ones(8)] @ l2i[c].T
                if np.any(uv[:, 2] <= 0.1):  # 在相机后方，跳过
                    continue
                u, v = uv[:, 0] / uv[:, 2], uv[:, 1] / uv[:, 2]
                for i, j in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
                             (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
                    axc.plot([u[i], u[j]], [v[i], v[j]], color=color, lw=1)
        axc.set_xlim(0, width)
        axc.set_ylim(height, 0)
        axc.set_title(CAM_NAMES[c])
        axc.axis("off")

    axes[2][1].axis("off")
    axl = axes[2][2]
    axl.axis("off")
    for arr, color, label in sets:
        axl.plot([], [], color=color, lw=2, label=f"{label} ({len(arr)})")
    axl.legend(loc="center", fontsize=14)

    plt.savefig(out, dpi=100, bbox_inches="tight")
    print(f"[vis] {out}")


def backend_vis(dump_dir, data_dir, cpp_boxes_txt=None):
    cpp_sets = ()
    if cpp_boxes_txt and osp.exists(cpp_boxes_txt):
        cpp_sets = ((parse_cpp_boxes(cpp_boxes_txt), "blue", "cpp"),)
    for sub, extra in [("onnx", ()), ("trt", cpp_sets)]:
        subs = [("torch", "limegreen", "torch"), (sub, "red", sub)]
        _plot_vis(osp.join(dump_dir, f"{sub}_vis.jpg"), dump_dir, data_dir, subs, extra)


# ---------------------------------------------------------------- report
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"
COLORS = {"torch": "#2a78d6", "onnx": "#eb6834", "trt": "#1baf7a"}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d5d4cf")
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def backend_report(dump_dir):
    """延时 + 精度对比图 -> report.png。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bench = None
    path = osp.join(dump_dir, "benchmark.json")
    if osp.exists(path):
        bench = json.load(open(path))
    acc = {sub: _stage_stats(dump_dir, sub) for sub in ("onnx", "trt")}
    boxes = _box_stats(dump_dir)

    fig = plt.figure(figsize=(13, 9), facecolor=SURFACE, constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.10])
    fig.suptitle("BEVFusion deployment parity: latency and accuracy",
                 color=INK, fontsize=13, x=0.01, ha="left")

    # 1. per-module latency
    ax = fig.add_subplot(gs[0, 0])
    _style(ax)
    if bench:
        backends = ["torch", "onnx", "trt"]
        y = np.arange(len(GRAPHS))
        h = 0.26
        for i, b in enumerate(backends):
            vals = [bench["backends"].get(b, {}).get(g, 0) for g in GRAPHS]
            ax.barh(y + (i - 1) * h, vals, height=h * 0.92,
                    color=COLORS[b], label=b, zorder=3)
        ax.set_yticks(y, GRAPHS, color=INK, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("latency (ms)", color=INK2, fontsize=9)
        ax.legend(frameon=False, fontsize=8.5, loc="lower right")
        ax.set_title("Per-module latency", color=INK, fontsize=11, loc="left")
    else:
        ax.text(0.5, 0.5, "benchmark.json missing\n(run --backend bench)",
                ha="center", va="center", color=INK2, transform=ax.transAxes)
        ax.set_axis_off()

    # 2. end-to-end latency
    ax = fig.add_subplot(gs[0, 1])
    _style(ax)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    if bench:
        backends = ["torch", "onnx", "trt"]
        vals = [bench["e2e"].get(b, 0) for b in backends]
        bars = ax.bar(backends, vals, color=[COLORS[b] for b in backends],
                      width=0.55, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.1f} ms",
                    ha="center", va="bottom", color=INK, fontsize=9)
        ax.set_ylim(0, max(vals) * 1.25)
        ax.set_ylabel("latency (ms)", color=INK2, fontsize=9)
        ax.set_title("End-to-end (4 modules, 1 frame)", color=INK, fontsize=11, loc="left")
        ax.text(0.0, -0.22, f"iters={bench['iters']}, warmup={bench['warmup']}; "
                            "torch/onnx fp32, trt fp16",
                transform=ax.transAxes, color=INK2, fontsize=8)
    else:
        ax.set_axis_off()

    # 3/4. accuracy vs torch
    for idx, metric in enumerate(("mean_abs", "max_abs")):
        ax = fig.add_subplot(gs[1, idx])
        _style(ax)
        stages = list(STAGE_TENSORS)
        y = np.arange(len(stages))
        h = 0.34
        for i, sub in enumerate(("onnx", "trt")):
            vals = [max(acc[sub].get(s, {}).get(metric, 0), 1e-9) for s in stages]
            ax.barh(y + (i - 0.5) * h, vals, height=h * 0.92,
                    color=COLORS[sub], label=sub, zorder=3)
        ax.set_xscale("log")
        ax.set_yticks(y, stages, color=INK, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel(f"{metric.replace('_', ' ')} (log)", color=INK2, fontsize=9)
        ax.legend(frameon=False, fontsize=8.5, loc="upper right")
        ax.set_title(("Mean absolute error vs torch" if metric == "mean_abs"
                      else "Max absolute error vs torch"),
                     color=INK, fontsize=11, loc="left")

    # footer: boxes + dumps
    axf = fig.add_subplot(gs[2, :])
    axf.axis("off")
    lines = []
    if boxes:
        parts = []
        for sub in ("torch", "onnx", "trt"):
            if sub in boxes:
                b = boxes[sub]
                extra = f", mean nn {b['mean_m'] * 100:.1f}cm" if "mean_m" in b else ""
                parts.append(f"{sub} {b['count']}{extra}")
        lines.append("boxes (score>=0.12): " + " | ".join(parts))
    lines.append(f"dump: {dump_dir}")
    axf.text(0.0, 0.5, "    ".join(lines), color=INK2, fontsize=8,
             va="center", transform=axf.transAxes)

    out = osp.join(dump_dir, "report.png")
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    print(f"[report] {out}")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?")
    ap.add_argument("ckpt", nargs="?")
    ap.add_argument("--backend", default="all",
                    choices=["torch", "onnx", "trt", "bench", "diff",
                             "trt-full", "vis", "report", "all"])
    ap.add_argument("--onnx-dir", default="outputs/onnx")
    ap.add_argument("--engine-dir", default="deploy/model/resnet50/build")
    ap.add_argument("--dump-dir", default="outputs/model_diff")
    ap.add_argument("--sample-index", type=int, default=0)
    ap.add_argument("--data-dir", default="outputs/example-data")
    ap.add_argument("--cpp-boxes", default=None)
    ap.add_argument("--head-plan", default=None)
    ap.add_argument("--cpp-dir", default="outputs/model_diff/cpp")
    ap.add_argument("--bench-iters", type=int, default=20)
    ap.add_argument("--bench-warmup", type=int, default=5)
    args = ap.parse_args()

    model = None
    need_model = args.backend in ("torch", "onnx", "trt", "bench", "all")
    if need_model:
        assert args.config and args.ckpt, f"{args.backend} 需要 config + ckpt"
    if args.backend in ("torch", "all"):
        model = backend_torch(args.config, args.ckpt, args.dump_dir,
                              args.sample_index, args.cpp_dir)
    elif need_model:
        model = load_model_only(args.config, args.ckpt)

    if args.backend in ("onnx", "all"):
        backend_onnx(args.onnx_dir, args.dump_dir, model)
    if args.backend in ("trt", "all"):
        backend_trt(args.engine_dir, args.dump_dir, model, args.head_plan)
    if args.backend in ("bench", "all"):
        backend_bench(args, model)
    if args.backend in ("diff", "all"):
        backend_diff(args.dump_dir)
    if args.backend in ("trt-full", "all"):
        _diff_cpp(args.dump_dir, args.cpp_dir)
    if args.backend in ("vis", "all"):
        backend_vis(args.dump_dir, args.data_dir, args.cpp_boxes)
    if args.backend in ("report", "all"):
        backend_report(args.dump_dir)


def _diff_cpp(dump_dir, cpp_dir):
    print("== torch vs C++ 全链 ==")
    for t in ["lidar_bev", "cam_bev", "middle"]:
        try:
            a = load_dump(t, dump_dir, "torch").astype(np.float32)
            b = load_tensor(osp.join(cpp_dir, f"trt_{t}")).astype(np.float32)
            d = np.abs(a - b)
            print(f"  {t}: max_abs={d.max():.4g} mean_abs={d.mean():.4g}")
        except FileNotFoundError:
            print(f"  {t}: C++ dump missing (BEVFUSION_DUMP_DIR)")


if __name__ == "__main__":
    main()
