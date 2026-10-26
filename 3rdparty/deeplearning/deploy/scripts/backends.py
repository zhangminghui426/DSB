"""三端推理后端：torch / onnxruntime / TensorRT 跑图并落盘中间张量。"""

import os
import os.path as osp

import numpy as np
import torch

from deploy.scripts.common import FEEDS, GRAPHS, OUTS, dump, load_dump, save_tensor
from deploy.scripts.export import (
    CamBackboneBf, CamVtransformDet, DownsampleBf, FuserDecoder, HeadBboxBf)


def load_model_only(config, checkpoint):
    from models import build_model
    from scripts.config import load_config
    from scripts.trainer import load_model_checkpoint
    cfg = load_config(config)
    model = build_model(cfg.model).cuda().eval()
    load_model_checkpoint(model, checkpoint, strict=True)
    return model


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


class TorchBackend:
    """训练接口全链，落盘各模块张量/框。"""

    def __init__(self, config, checkpoint, dump_dir, sample_index=0,
                 cpp_dir="outputs/model_diff/cpp"):
        self.config = config
        self.checkpoint = checkpoint
        self.dump_dir = dump_dir
        self.sample_index = sample_index
        self.cpp_dir = cpp_dir

    def run(self):
        from datasets import build_dataset, collate_fn
        from models import build_model
        from scripts.config import load_config
        from scripts.trainer import load_model_checkpoint, to_device

        cfg = load_config(self.config)
        model = build_model(cfg.model).cuda().eval()
        load_model_checkpoint(model, self.checkpoint, strict=True)
        batch = to_device(collate_fn([build_dataset(cfg.data.val)[self.sample_index]]), "cuda")

        vt = model.encoders["camera"]["vtransform"]
        img = batch["img"]
        B, N, C, H, W = img.shape
        np_dump = lambda t: t.detach().cpu().numpy()

        depth = project_depth(batch, vt)
        dump("img", np_dump(img), self.dump_dir, "torch")
        dump("depth", np_dump(depth), self.dump_dir, "torch")
        feature, depth_weights = CamBackboneBf(model)(img, depth)
        dump("camera_feature", np_dump(feature), self.dump_dir, "torch")
        dump("camera_depth_weights", np_dump(depth_weights), self.dump_dir, "torch")

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
        dump("bev360", np_dump(bev360), self.dump_dir, "torch")
        cam_bev = DownsampleBf(model)(bev360)
        dump("cam_bev", np_dump(cam_bev), self.dump_dir, "torch")

        feats_l, coords, sizes = model.voxelize(batch["points"], "lidar")
        lidar_bev = model.encoders["lidar"]["backbone"](
            feats_l, coords, coords[-1, 0] + 1, sizes=sizes)
        dump("lidar_bev", np_dump(lidar_bev), self.dump_dir, "torch")
        save_tensor(coords.detach().cpu().numpy().astype(np.int32),
                    osp.join(self.cpp_dir, "torch_lidar_coords"))
        save_tensor(feats_l.detach().cpu().numpy().astype(np.float16),
                    osp.join(self.cpp_dir, "torch_lidar_feats"))

        middle = FuserDecoder(model)(cam_bev, lidar_bev)
        dump("middle", np_dump(middle), self.dump_dir, "torch")

        head = model.heads["object"]
        head.onnx_export = True
        for name, t in zip(["score", "rot", "dim", "reg", "height", "vel"],
                           HeadBboxBf(model)(middle)):
            dump(name, np_dump(t), self.dump_dir, "torch")

        pred_dicts = head([middle], [{}])
        bboxes = head.get_bboxes(pred_dicts, batch["metas"])
        boxes = bboxes[0][0].tensor.detach().cpu().numpy()
        scores = bboxes[0][1].detach().cpu().numpy()
        labels = bboxes[0][2].detach().cpu().numpy()
        dump("query_labels", head.query_labels.detach().cpu().numpy(), self.dump_dir, "torch")
        dump("boxes", np.concatenate([boxes, scores[:, None], labels[:, None]], axis=1),
             self.dump_dir, "torch")
        print(f"[torch] dumps -> {self.dump_dir}/torch, {len(scores)} boxes")
        return model


class OnnxBackend:
    """onnxruntime 跑 4 张图（输入取 torch dump）。"""

    def __init__(self, onnx_dir, dump_dir):
        self.onnx_dir = onnx_dir
        self.dump_dir = dump_dir

    def run(self, model=None):
        import onnxruntime as ort

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        for graph in GRAPHS:
            path = osp.join(self.onnx_dir, graph + ".onnx")
            if not osp.exists(path):
                print(f"[onnx] skip {graph}: {path} missing")
                continue
            sess = ort.InferenceSession(path, providers=providers)
            feed = {k: load_dump(v, self.dump_dir).astype(np.float32)
                    for k, v in FEEDS[graph].items()}
            for key, arr in zip(OUTS[graph], sess.run(None, feed)):
                dump(key, np.asarray(arr).astype(np.float32), self.dump_dir, "onnx")
            print(f"[onnx] {graph} ok")
        decode_boxes(self.dump_dir, "onnx", model)


def trt_plan_path(engine_dir, graph, head_plan=None):
    if graph != "head.bbox":
        return osp.join(engine_dir, graph + ".plan")
    if head_plan:
        return osp.join(engine_dir, head_plan)
    for name in ("head.bbox.fp32.plan", "head.bbox.layernormplugin.plan", "head.bbox.plan"):
        path = osp.join(engine_dir, name)
        if osp.exists(path):
            return path
    return osp.join(engine_dir, "head.bbox.plan")


def trt_session(engine_dir, graph, head_plan=None):
    import tensorrt as trt

    path = graph if osp.isabs(graph) or "/" in graph else trt_plan_path(engine_dir, graph, head_plan)
    rt = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    with open(path, "rb") as f:
        eng = rt.deserialize_cuda_engine(f.read())
    ctx = eng.create_execution_context()
    names = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors)]
    ins = [n for n in names if eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
    outs = [n for n in names if eng.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
    return ctx, ins, outs


def bind_trt(ctx, in_names, out_names, graph, dump_dir):
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


class TrtBackend:
    """TensorRT 跑 4 个 plan（输入取 torch dump）。"""

    def __init__(self, engine_dir, dump_dir, head_plan=None):
        self.engine_dir = engine_dir
        self.dump_dir = dump_dir
        self.head_plan = head_plan

    def run(self, model=None):
        stream = torch.cuda.current_stream().cuda_stream
        for graph in GRAPHS:
            path = trt_plan_path(self.engine_dir, graph, self.head_plan)
            if not osp.exists(path):
                print(f"[trt] skip {graph}: {path} missing")
                continue
            ctx, in_names, out_names = trt_session(self.engine_dir, path)
            buffers = bind_trt(ctx, in_names, out_names, graph, self.dump_dir)
            if not ctx.execute_async_v3(stream):
                print(f"[trt] {graph} execute failed")
                continue
            torch.cuda.synchronize()
            for key, name in zip(OUTS[graph], out_names):
                dump(key, buffers[name].cpu().numpy(), self.dump_dir, "trt")
            print(f"[trt] {graph} ok")
        decode_boxes(self.dump_dir, "trt", model)


def decode_boxes(dump_dir, sub, model=None):
    """head 输出 -> 3D 框（filter=False + score>0.12，与 C++ 对齐）。"""
    if model is None or not osp.exists(osp.join(dump_dir, sub, "score.npy")):
        return
    head = model.heads["object"]
    get = lambda k: torch.from_numpy(load_dump(k, dump_dir, sub)).float().cuda()
    preds = head.bbox_coder.decode(get("score"), get("rot"), get("dim"), get("reg"),
                                   get("height"), get("vel"), filter=False)
    boxes = preds[0]["bboxes"].detach().cpu().numpy()
    scores = preds[0]["scores"].detach().cpu().numpy()
    labels = preds[0]["labels"].detach().cpu().numpy()
    mask = scores > 0.12
    dump("boxes", np.concatenate([boxes[mask], scores[mask, None], labels[mask, None]], axis=1),
         dump_dir, sub)
    print(f"[{sub}] {int(mask.sum())} boxes (score>0.12)")
