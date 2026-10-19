"""Camera BEV assembly at deploy time: depthnet output -> BEV feature.

The exported ONNX graphs stop at the depth network (dtransform/depthnet);
geometry + bev_pool stay on the torch/CUDA runtime side because bev_pool is
a custom CUDA op. This module rebuilds the camera BEV from a depthnet output
with torch geometry + the CUDA kernel (and can run the pure-torch reference
for engine-free verification).
"""

import torch

__all__ = ["camera_bev_from_depthnet"]


def camera_bev_from_depthnet(model, batch, depthnet, task="det", feats=None):
    """depthnet logits (B*N, D+C, fH, fW) -> camera BEV (B, C*Dz, H, W).

    Mirrors DepthLSSTransform/LSSTransform forward after the depth network:
    softmax over depth bins, fold depth into features, lift with the camera
    geometry and run bev_pool (CUDA op at runtime, or the onnx reference if
    the vtransform has onnx_export enabled).

    Args:
        model: BevPerception with a camera encoder.
        batch: collated sample dict (img + camera matrices on the model device).
        depthnet: (B*N, D+C, fH, fW) depth-network logits.
        task: "det" or "seg" (informational; both share this code path).
        feats: optional (B*N, C, fH, fW) camera backbone output. When given
            (e.g. from a TRT engine) the backbone/neck are not re-run.

    Returns:
        (B, C*Dz, H, W) camera BEV feature (Z collapsed, per base bev_pool).
    """
    vt = model.encoders["camera"]["vtransform"]
    img = batch["img"]
    B, N = img.shape[0], img.shape[1]
    fH, fW = vt.feature_size
    D, C = vt.D, vt.C

    # camera backbone + neck (torch runtime, not part of the exported graphs)
    if feats is None:
        x = img.view(B * N, *img.shape[2:])
        x = model.encoders["camera"]["backbone"](x)
        x = model.encoders["camera"]["neck"](x)
        if not torch.is_tensor(x):
            x = x[0]
        feats = x
    x = feats.view(B, N, C, fH, fW)

    # fold depth probability into features:
    # (B,N,D,fH,fW) * (B,N,C,fH,fW) -> (B,N,C,D,fH,fW) -> (B,N,D,fH,fW,C)
    depth = depthnet[:, :D].softmax(dim=1).view(B, N, D, fH, fW)
    feat = depthnet[:, D:].view(B, N, C, fH, fW)
    x = (depth.unsqueeze(2) * feat.unsqueeze(3))
    x = x.permute(0, 1, 3, 4, 5, 2).contiguous()

    geom = vt.get_geometry(
        batch["camera2lidar"][..., :3, :3],
        batch["camera2lidar"][..., :3, 3],
        batch["camera_intrinsics"][..., :3, :3],
        batch["img_aug_matrix"][..., :3, :3],
        batch["img_aug_matrix"][..., :3, 3],
        extra_rots=batch["lidar_aug_matrix"][..., :3, :3],
        extra_trans=batch["lidar_aug_matrix"][..., :3, 3],
    )
    bev = vt.bev_pool(geom, x)  # (B, C, Dz, H, W) -> collapse Z -> (B, C*Dz, H, W)
    if hasattr(vt, "downsample") and not isinstance(vt.downsample, torch.nn.Identity):
        bev = vt.downsample(bev)
    return bev
