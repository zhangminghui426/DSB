"""Export det inference inputs / weights / reference for the C++ runtime.

Produces src/cpp_infer consumes:
  <out>/meta.json            geometry + structure + decode parameters
  <out>/inputs/img.bin       (6,3,256,704) fp32 normalized image (eval protocol)
  <out>/inputs/points.bin    int32 n + (n,5) fp32 lidar points (eval protocol)
  <out>/inputs/matrices.bin  19x(4x4) float32 row-major: 6 camera2lidar +
                             6 camera_intrinsics + 6 img_aug + 1 lidar_aug
  <out>/weights/sparse_encoder.bin  conv/bn params, concatenated (meta gives layout)
  <out>/weights/downsample.bin      vtransform downsample conv/bn params
  <out>/ref/decoder_feat.bin        (1,512,180,180) fp32 torch reference
  <out>/ref/boxes.json              decoded boxes/scores/labels reference

Run on the GPU box (same env as training):
  python tools/export_cpp_input.py configs/det_convfuser.py ckpts/bevfusion-det-bf.pth \
      --out-dir deploy_inputs --sample-index 0

Then build & run src/cpp_infer with --input-dir <out>.
"""

import argparse
import json
import os
import os.path as osp
import sys

# deepleraing holds datasets/models/scripts/configs; allow running this script
# from the repo root with bare relative paths like configs/det_convfuser.py.
_DL_ROOT = osp.join(osp.dirname(osp.abspath(__file__)), "..", "3rdparty",
                    "deepleraing")
sys.path.insert(0, _DL_ROOT)
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), ".."))

import numpy as np
import torch

from datasets import build_dataset, collate_fn
from models import build_model
from scripts.config import load_config
from scripts.trainer import load_model_checkpoint, to_device


def dump_bin(path, arr):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    arr.tofile(path)


def bn_params(m):
    """(gamma, beta, running_mean, running_var, eps) as np arrays."""
    return {
        "w": m.weight.detach().cpu().numpy(),
        "b": m.bias.detach().cpu().numpy(),
        "mean": m.running_mean.detach().cpu().numpy(),
        "var": m.running_var.detach().cpu().numpy(),
        "eps": float(m.eps),
    }


def conv_params(m):
    w = m.weight.detach().cpu().numpy()  # spconv 2.x: (Cout, K, D, H, Cin)
    b = m.bias.detach().cpu().numpy() if m.bias is not None else None
    return w, b


def collect_sparse_layers(encoder):
    """Enumerate SparseEncoder conv layers in execution order.

    Returns list of layer dicts + concatenated float32 weight blob.
    Layer dict: type=subm|strided, in/out channels, kernel (kx,ky,kz),
    stride, padding, offsets (z,y,x-outer kernel enumeration, K entries),
    w_off/w_len/b_off in the blob.
    """
    layers, blobs = [], []

    def add(conv, bn, ktype, kernel, stride, pad):
        w, b = conv_params(conv)
        # spconv 2.x conv.weight: (Cout, K, D, H, Cin) -> flatten to (Cout, K, Cin)
        Cout, K, D, H, Cin = w.shape
        assert (K, D, H) == tuple(kernel), (w.shape, kernel)
        wf = w.reshape(Cout, K * D * H, Cin)
        # kernel enumeration: z, y, x outer (spconv convention)
        offsets = []
        for dz in range(kernel[2]):
            for dy in range(kernel[1]):
                for dx in range(kernel[0]):
                    offsets.append(
                        [dx - pad[0] if ktype == "subm" else dx,
                         dy - pad[1] if ktype == "subm" else dy,
                         dz - pad[2] if ktype == "subm" else dz]
                    )
        g, bta, mean, var, eps = (
            bn_params(bn)["w"], bn_params(bn)["b"],
            bn_params(bn)["mean"], bn_params(bn)["var"], bn_params(bn)["eps"],
        )
        off = sum(x.size for x in blobs)
        blobs.extend([wf, b if b is not None else np.zeros(Cout, np.float32),
                      g, bta, mean, var])
        layers.append(dict(
            type=ktype, in_c=Cin, out_c=Cout, kernel=kernel,
            stride=stride, pad=pad, offsets=offsets,
            w_off=off,
            w_len=int(wf.size), b_len=Cout,
            bn_eps=eps,
        ))

    def add_subm_block(conv1, bn1, conv2, bn2):
        add(conv1, bn1, "subm", (3, 3, 3), (1, 1, 1), (1, 1, 1))
        add(conv2, bn2, "subm", (3, 3, 3), (1, 1, 1), (1, 1, 1))

    # conv_input
    add(encoder.conv_input[0], encoder.conv_input[1], "subm", (3, 3, 3), (1, 1, 1), (1, 1, 1))
    for i, layer in enumerate(encoder.encoder_layers):
        for j, m in enumerate(layer):
            if m.__class__.__name__ == "SparseBasicBlock":
                add_subm_block(m.conv1, m.bn1, m.conv2, m.bn2)
            else:  # SparseSequential(conv, bn, relu)
                add(m[0], m[1], "strided", (3, 3, 3), (2, 2, 2), (1, 1, 1))
    # conv_out: SparseConv3d kernel (1,1,3) stride (1,1,2) pad 0 (x,y,z order)
    add(encoder.conv_out[0], encoder.conv_out[1], "strided", (1, 1, 3), (1, 1, 2), (0, 0, 0))
    return layers, np.concatenate(blobs)


def collect_downsample(vt):
    """DepthLSSTransform.downsample: 3x (conv, bn, relu); mid conv stride 2."""
    layers, blobs = [], []
    names = []
    for m in vt.downsample:
        if isinstance(m, torch.nn.Conv2d):
            names.append(m)
        elif isinstance(m, torch.nn.BatchNorm2d):
            conv = names[-1]
            w = conv.weight.detach().cpu().numpy()
            b = conv.bias.detach().cpu().numpy() if conv.bias is not None else None
            g, bta, mean, var = (bn_params(m)["w"], bn_params(m)["b"],
                                 bn_params(m)["mean"], bn_params(m)["var"])
            off = sum(x.size for x in blobs)
            blobs.extend([w, b if b is not None else np.zeros(w.shape[0], np.float32),
                          g, bta, mean, var])
            layers.append(dict(
                in_c=w.shape[1], out_c=w.shape[0], k=conv.kernel_size[0],
                stride=conv.stride[0], pad=conv.padding[0],
                w_off=off, w_len=int(w.size), b_len=w.shape[0],
                bn_eps=float(m.eps),
            ))
    return layers, np.concatenate(blobs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("--out-dir", default="deploy_inputs")
    ap.add_argument("--sample-index", type=int, default=0)
    args = ap.parse_args()

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    # resolve bare repo-root relative paths against 3rdparty/deepleraing
    def resolve(p):
        return p if osp.exists(p) else osp.join(_DL_ROOT, p)

    cfg = load_config(resolve(args.config))

    # rebase bare relative data paths onto 3rdparty/deepleraing
    def rebase(p):
        return p if (not p or osp.isabs(p) or osp.exists(p)) else \
            osp.join(_DL_ROOT, p)

    for phase in ("train", "val", "test"):
        d = cfg.data.get(phase)
        if not d:
            continue
        if d.get("dataset_root"):
            d["dataset_root"] = rebase(d["dataset_root"])
        if d.get("ann_file"):
            d["ann_file"] = rebase(d["ann_file"])

    model = build_model(cfg.model).cuda().eval()
    load_model_checkpoint(model, resolve(args.checkpoint), strict=True)

    ds = build_dataset(cfg.data.val)
    sample = ds[args.sample_index]
    batch = to_device(collate_fn([sample]), "cuda")

    out = args.out_dir
    for sub in ("inputs", "weights", "ref"):
        os.makedirs(osp.join(out, sub), exist_ok=True)

    # ---- inputs -------------------------------------------------------
    img = batch["img"][0].detach().cpu().numpy()  # (6,3,256,704)
    dump_bin(osp.join(out, "inputs/img.bin"), img)
    pts = batch["points"][0].detach().cpu().numpy().astype(np.float32)  # (N,5)
    with open(osp.join(out, "inputs/points.bin"), "wb") as f:
        f.write(np.int32(len(pts)).tobytes())
        f.write(pts.tobytes())
    cam2lidar = batch["camera2lidar"][0].detach().cpu().numpy()   # (N,4,4)
    intrins = batch["camera_intrinsics"][0].detach().cpu().numpy()  # (N,4,4)
    img_aug = batch["img_aug_matrix"][0].detach().cpu().numpy()   # (N,4,4)
    lidar_aug = batch["lidar_aug_matrix"][0].detach().cpu().numpy()  # (4,4)
    if lidar_aug.ndim == 2:
        lidar_aug = lidar_aug[None]  # -> (1,4,4)
    mats = np.concatenate([cam2lidar, intrins, img_aug, lidar_aug],
                          axis=0).astype(np.float32)  # (19,4,4)
    dump_bin(osp.join(out, "inputs/matrices.bin"), mats)

    # ---- weights -----------------------------------------------------
    enc = model.encoders["lidar"]["backbone"]
    sparse_layers, sparse_blob = collect_sparse_layers(enc)
    dump_bin(osp.join(out, "weights/sparse_encoder.bin"), sparse_blob)
    vt = model.encoders["camera"]["vtransform"]
    ds_layers, ds_blob = collect_downsample(vt)
    dump_bin(osp.join(out, "weights/downsample.bin"), ds_blob)

    # ---- reference outputs -------------------------------------------
    with torch.no_grad():
        bev = model.bev_branch(
            batch["img"], batch["points"], None, batch["camera2ego"],
            batch["lidar2ego"], batch["lidar2camera"], batch["lidar2image"],
            batch["camera_intrinsics"], batch["camera2lidar"],
            batch["img_aug_matrix"], batch["lidar_aug_matrix"],
            batch["metas"], depths=None, reverse=True,
        )[0]
        dump_bin(osp.join(out, "ref/decoder_feat.bin"), bev.detach().cpu().numpy())
        out_d = model(**batch)[0]
        boxes = out_d["boxes_3d"].tensor.detach().cpu().numpy()
        scores = out_d["scores_3d"].detach().cpu().numpy()
        labels = out_d["labels_3d"].detach().cpu().numpy()
    with open(osp.join(out, "ref/boxes.json"), "w") as f:
        json.dump({"boxes": boxes.tolist(), "scores": scores.tolist(),
                   "labels": labels.tolist()}, f)

    head = model.heads["object"]
    bbox_coder = head.bbox_coder
    meta = dict(
        B=1, N=6,
        image_size=[256, 704], feature_size=[32, 88],
        xbound=[-54.0, 54.0, 0.3], ybound=[-54.0, 54.0, 0.3],
        zbound=[-10.0, 10.0, 20.0], dbound=[1.0, 60.0, 0.5],
        lidar=dict(
            point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
            voxel_size=[0.075, 0.075, 0.2],
            grid_size=[1440, 1440, 41],
            max_points_per_voxel=10, max_voxels=160000,
            sparse_shape_out=[180, 180, 2],  # (X, Y, Z) after conv_out
            layers=sparse_layers,
        ),
        vtransform=dict(
            out_channels=80,
            downsample_layers=ds_layers,
        ),
        head=dict(
            num_classes=10, num_proposals=200, nms_kernel_size=3,
            num_proposals_bev=180 * 180,
            pc_range=[-54.0, -54.0],
            out_size_factor=8, voxel_size=[0.075, 0.075],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            score_threshold=0.0,
        ),
        sample_index=args.sample_index,
    )
    with open(osp.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)

    # ---- C++ key=value meta (meta.txt) -------------------------------
    lines = ["# dsb cpp inference meta (key = space-separated values)"]
    lines.append(f"B = {meta['B']}")
    lines.append(f"N = {meta['N']}")
    lines.append(f"image_h = {meta['image_size'][0]}")
    lines.append(f"image_w = {meta['image_size'][1]}")
    lines.append(f"feat_h = {meta['feature_size'][0]}")
    lines.append(f"feat_w = {meta['feature_size'][1]}")
    for k in ("xbound", "ybound", "zbound", "dbound"):
        lines.append(f"{k} = {' '.join(map(str, meta[k]))}")
    nx = [int((meta['xbound'][1] - meta['xbound'][0]) / meta['xbound'][2]),
          int((meta['ybound'][1] - meta['ybound'][0]) / meta['ybound'][2]),
          int((meta['zbound'][1] - meta['zbound'][0]) / meta['zbound'][2])]
    lines.append(f"nx = {' '.join(map(str, nx))}")
    lc = meta["lidar"]
    lines.append(f"lidar_pcr = {' '.join(map(str, lc['point_cloud_range']))}")
    lines.append(f"lidar_voxel = {' '.join(map(str, lc['voxel_size']))}")
    lines.append(f"lidar_grid = {' '.join(map(str, lc['grid_size']))}")
    lines.append(f"lidar_max_pts = {lc['max_points_per_voxel']}")
    lines.append(f"lidar_max_voxels = {lc['max_voxels']}")
    lines.append(f"sparse_out = {' '.join(map(str, lc['sparse_shape_out']))}")
    lines.append(f"vt_out_c = {meta['vtransform']['out_channels']}")
    hd = meta["head"]
    lines.append(f"head_nclass = {hd['num_classes']}")
    lines.append(f"head_nprop = {hd['num_proposals']}")
    lines.append(f"head_nms_k = {hd['nms_kernel_size']}")
    lines.append(f"head_bev = {hd['num_proposals_bev']}")
    lines.append(f"head_pc = {' '.join(map(str, hd['pc_range']))}")
    lines.append(f"head_osf = {hd['out_size_factor']}")
    lines.append(f"head_voxel = {' '.join(map(str, hd['voxel_size']))}")
    lines.append(f"head_post = {' '.join(map(str, hd['post_center_range']))}")
    lines.append(f"head_score_thr = {hd['score_threshold']}")
    lines.append(f"sparse_layer_N = {len(lc['layers'])}")
    for ly in lc["layers"]:
        k = ly["kernel"]
        s = ly["stride"]
        p = ly["pad"]
        lines.append(
            f"{ly['type']} {ly['in_c']} {ly['out_c']} "
            f"{k[0]} {k[1]} {k[2]} {s[0]} {s[1]} {s[2]} {p[0]} {p[1]} {p[2]} "
            f"{len(ly['offsets'])} {ly['w_off']} {ly['w_len']} {ly['b_len']} {ly['bn_eps']}"
        )
        for off in ly["offsets"]:
            lines.append(f"  {off[0]} {off[1]} {off[2]}")
    dsl = meta["vtransform"]["downsample_layers"]
    lines.append(f"downsample_N = {len(dsl)}")
    for ly in dsl:
        lines.append(
            f"{ly['in_c']} {ly['out_c']} {ly['k']} {ly['stride']} {ly['pad']} "
            f"{ly['w_off']} {ly['w_len']} {ly['b_len']} {ly['bn_eps']}"
        )
    with open(osp.join(out, "meta.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[export] wrote {out} (sample {args.sample_index})")
    print(f"[export] points: {pts.shape}, img: {img.shape}")
    print(f"[export] ref boxes: {len(scores)}, top score: {scores[:3].tolist() if len(scores) else []}")


if __name__ == "__main__":
    main()
