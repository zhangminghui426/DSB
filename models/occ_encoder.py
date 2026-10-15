"""Occupancy image->BEV encoder chains (the occ branch of BevPerception).

OccEncoder: ResNet|Swin backbone + CustomFPN neck + BEVPoolv2 LSS view
transform + CustomResNet/FPN_LSS BEV encoder (port of BEVDetOCC minus head).
OccEncoderStereo4D: adds the temporal loop (adjacent frames under no_grad,
extra stereo reference frame, cost volume) and optional test-time BEV
feature alignment (port of BEVStereo4DOCC minus head).

Attribute names match the official FlashOCC checkpoints exactly
(img_backbone / img_neck / img_view_transformer / img_bev_encoder_backbone /
img_bev_encoder_neck [/ pre_process_net]) so tools/remap_occ.py is a single
prefix remap. forward returns (bev_feature, aux_losses) — the occ head lives
in BevPerception.heads["occ"].
"""

import torch
import torch.nn.functional as F
from torch import nn

from models.backbones.resnet import ResNetBackbone
from models.backbones.swin import SwinTransformer
from models.bev_encoder import CustomResNet, FPN_LSS
from models.necks.custom_fpn import CustomFPN
from models.vtransforms.occ_lss import LSSViewTransformer

__all__ = ["OccEncoder", "OccEncoderStereo4D"]


def drop_type(cfg):
    """Copy a config dict without its 'type' key (dispatch happens here)."""
    cfg = dict(cfg)
    cfg.pop("type", None)
    return cfg


class OccEncoder(nn.Module):
    def __init__(self, img_backbone, img_neck, img_view_transformer,
                 img_bev_encoder_backbone, img_bev_encoder_neck,
                 upsample=False):
        super().__init__()
        bb_cfg = dict(img_backbone)
        bb_type = bb_cfg.pop("type")
        if bb_type == "ResNetBackbone":
            self.img_backbone = ResNetBackbone(**bb_cfg)
        elif bb_type == "SwinTransformer":
            self.img_backbone = SwinTransformer(**bb_cfg)
        else:
            raise ValueError(f"unsupported img_backbone: {bb_type}")

        neck_cfg = dict(img_neck)
        neck_type = neck_cfg.pop("type")
        if neck_type == "CustomFPN":
            self.img_neck = CustomFPN(**neck_cfg)
        elif neck_type == "FPN_LSS":
            self.img_neck = FPN_LSS(**neck_cfg)
        else:
            raise ValueError(f"unsupported img_neck: {neck_type}")

        vt_cfg = dict(img_view_transformer)
        vt_type = vt_cfg.pop("type")
        if vt_type == "LSSViewTransformer":
            self.img_view_transformer = LSSViewTransformer(**vt_cfg)
        elif vt_type in ("LSSViewTransformerBEVStereo", "LSSViewTransformerBEVDepth"):
            from models.vtransforms.occ_lss_depth import LSSViewTransformerBEVStereo

            self.img_view_transformer = LSSViewTransformerBEVStereo(**vt_cfg)
        else:
            raise ValueError(f"unsupported vtransform: {vt_type}")

        self.img_bev_encoder_backbone = CustomResNet(**drop_type(img_bev_encoder_backbone))
        self.img_bev_encoder_neck = FPN_LSS(**drop_type(img_bev_encoder_neck))
        self.upsample = upsample

    def prepare_inputs(self, inputs):
        """7-tuple; sensor->key-ego transforms computed in float64 (numeric
        parity with the original), then cast back to float32."""
        assert len(inputs) == 7
        B, N, C, H, W = inputs[0].shape
        imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = inputs

        sensor2egos = sensor2egos.view(B, N, 4, 4)
        ego2globals = ego2globals.view(B, N, 4, 4)
        keyego2global = ego2globals[:, 0, ...].unsqueeze(1)
        global2keyego = torch.inverse(keyego2global.double())
        sensor2keyegos = (global2keyego @ ego2globals.double() @ sensor2egos.double()).float()
        return [imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda]

    def image_encoder(self, img):
        """(B, N, 3, H, W) -> (B, N, C', fH, fW)"""
        B, N, C, imH, imW = img.shape
        x = self.img_backbone(img.view(B * N, C, imH, imW))
        x = self.img_neck(x)
        if isinstance(x, (list, tuple)):
            x = x[0]
        _, output_dim, outH, outW = x.shape
        return x.view(B, N, output_dim, outH, outW)

    def bev_encoder(self, x):
        x = self.img_bev_encoder_backbone(x)
        x = self.img_bev_encoder_neck(x)
        if isinstance(x, (list, tuple)):
            x = x[0]
        return x

    def forward(self, img_inputs, gt_depth=None, **kwargs):
        """img_inputs 7-tuple -> (BEV feature (B, C, H, W), aux losses dict)."""
        img_inputs = self.prepare_inputs(img_inputs)
        x = self.image_encoder(img_inputs[0])
        x, depth = self.img_view_transformer([x] + img_inputs[1:7])
        bev_feat = self.bev_encoder(x)
        if self.upsample:
            bev_feat = F.interpolate(
                bev_feat, scale_factor=2, mode="bilinear", align_corners=True
            )
        return bev_feat, {}


class OccEncoderStereo4D(OccEncoder):
    """Temporal flow per forward (num_frame = key + num_adj + extra_ref,
    processed fid = num_frame-1 .. 0):
    - extra ref frame: stereo feature only (backbone /4 stride), no BEV
    - adjacent frames: full path under torch.no_grad(), cost volume vs the
      previous frame's stereo feature
    - key frame: with gradients; BEV features concatenated as [prev, key]
    With align_after_view_transfromation (official test protocol), adjacent
    BEV features are re-aligned to the key ego frame via grid_sample.
    """

    def __init__(self, pre_process=None, align_after_view_transfromation=False,
                 num_adj=1, with_prev=True, **kwargs):
        super().__init__(**kwargs)
        self.pre_process = pre_process is not None
        if self.pre_process:
            self.pre_process_net = CustomResNet(**drop_type(pre_process))
        self.align_after_view_transfromation = align_after_view_transfromation
        self.num_frame = num_adj + 1
        self.with_prev = with_prev
        self.grid = None
        # stereo: one extra reference frame for the cost volume
        self.extra_ref_frames = 1
        self.temporal_frame = self.num_frame
        self.num_frame += self.extra_ref_frames

    # ---------------------------------------------------------------- inputs
    def prepare_inputs(self, img_inputs, stereo=False):
        """Split the sequential-frame batch into per-frame tensors.

        imgs are cam-major (B, N_views*num_frame, ...), poses frame-major
        (B, num_frame*N_views, ...). Returns per-frame lists plus (for stereo)
        curr2adjsensor transforms. sensor2keyego computed in float64.
        """
        B, N, C, H, W = img_inputs[0].shape
        N = N // self.num_frame
        imgs = img_inputs[0].view(B, N, self.num_frame, C, H, W)
        imgs = [t.squeeze(2) for t in torch.split(imgs, 1, 2)]
        sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = img_inputs[1:7]

        sensor2egos = sensor2egos.view(B, self.num_frame, N, 4, 4)
        ego2globals = ego2globals.view(B, self.num_frame, N, 4, 4)

        keyego2global = ego2globals[:, 0, 0, ...].unsqueeze(1).unsqueeze(1)
        global2keyego = torch.inverse(keyego2global.double())
        sensor2keyegos = (global2keyego @ ego2globals.double() @ sensor2egos.double()).float()

        curr2adjsensor = None
        if stereo:
            sensor2egos_curr = sensor2egos[:, :self.temporal_frame, ...].double()
            ego2globals_curr = ego2globals[:, :self.temporal_frame, ...].double()
            sensor2egos_adj = sensor2egos[:, 1:self.temporal_frame + 1, ...].double()
            ego2globals_adj = ego2globals[:, 1:self.temporal_frame + 1, ...].double()
            # curr_sensor -> curr_ego -> global -> prev_ego -> prev_sensor
            curr2adjsensor = (
                torch.inverse(ego2globals_adj @ sensor2egos_adj)
                @ ego2globals_curr @ sensor2egos_curr
            ).float()
            curr2adjsensor = [p.squeeze(1) for p in torch.split(curr2adjsensor, 1, 1)]
            curr2adjsensor.extend([None for _ in range(self.extra_ref_frames)])
            assert len(curr2adjsensor) == self.num_frame

        extra = [
            sensor2keyegos,
            ego2globals,
            intrins.view(B, self.num_frame, N, 3, 3),
            post_rots.view(B, self.num_frame, N, 3, 3),
            post_trans.view(B, self.num_frame, N, 3),
        ]
        extra = [[p.squeeze(1) for p in torch.split(t, 1, 1)] for t in extra]
        sensor2keyegos, ego2globals, intrins, post_rots, post_trans = extra
        return (imgs, sensor2keyegos, ego2globals, intrins, post_rots,
                post_trans, bda, curr2adjsensor)

    # ------------------------------------------------------------- alignment
    def gen_grid(self, input, sensor2keyegos, bda, bda_adj=None):
        """Sampling grid mapping the key BEV cells into the adjacent frame."""
        B, C, H, W = input.shape
        if self.grid is None:
            xs = torch.linspace(0, W - 1, W, dtype=input.dtype, device=input.device).view(1, W).expand(H, W)
            ys = torch.linspace(0, H - 1, H, dtype=input.dtype, device=input.device).view(H, 1).expand(H, W)
            self.grid = torch.stack((xs, ys, torch.ones_like(xs)), -1)
        grid = self.grid
        grid = grid.view(1, H, W, 3).expand(B, H, W, 3).view(B, H, W, 3, 1)

        curr_sensor2keyego = sensor2keyegos[0][:, 0:1, :, :]
        prev_sensor2keyego = sensor2keyegos[1][:, 0:1, :, :]

        bda_ = torch.zeros((B, 1, 4, 4), dtype=grid.dtype).to(grid)
        bda_[:, :, :3, :3] = bda.unsqueeze(1)
        bda_[:, :, 3, 3] = 1
        curr_sensor2keyego = bda_.matmul(curr_sensor2keyego)
        if bda_adj is not None:
            bda_ = torch.zeros((B, 1, 4, 4), dtype=grid.dtype).to(grid)
            bda_[:, :, :3, :3] = bda_adj.unsqueeze(1)
            bda_[:, :, 3, 3] = 1
        prev_sensor2keyego = bda_.matmul(prev_sensor2keyego)

        # key_ego -> prev_ego (via the front-camera sensor frames)
        keyego2adjego = curr_sensor2keyego.matmul(torch.inverse(prev_sensor2keyego))
        keyego2adjego = keyego2adjego.unsqueeze(dim=1)
        keyego2adjego = keyego2adjego[..., [True, True, False, True], :][..., [True, True, False, True]]

        vt = self.img_view_transformer
        feat2bev = torch.zeros((3, 3), dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = vt.grid_interval[0]
        feat2bev[1, 1] = vt.grid_interval[1]
        feat2bev[0, 2] = vt.grid_lower_bound[0]
        feat2bev[1, 2] = vt.grid_lower_bound[1]
        feat2bev[2, 2] = 1
        feat2bev = feat2bev.view(1, 3, 3)

        tf = torch.inverse(feat2bev).matmul(keyego2adjego).matmul(feat2bev)
        grid = tf.matmul(grid)
        normalize_factor = torch.tensor([W - 1.0, H - 1.0], dtype=input.dtype, device=input.device)
        return grid[:, :, :, :2, 0] / normalize_factor.view(1, 1, 1, 2) * 2.0 - 1.0

    def shift_feature(self, input, sensor2keyegos, bda, bda_adj=None):
        grid = self.gen_grid(input, sensor2keyegos, bda, bda_adj=bda_adj)
        return F.grid_sample(input, grid.to(input.dtype), align_corners=True)

    # -------------------------------------------------------------- encoders
    def image_encoder(self, img, stereo=False):
        """(B, N, 3, H, W) -> ((B, N, C, fH, fW), stereo_feat or None).

        stereo_feat is the backbone's first output (out_indices[0], /4 stride).
        """
        B, N, C, imH, imW = img.shape
        x = self.img_backbone(img.view(B * N, C, imH, imW))
        stereo_feat = None
        if stereo:
            stereo_feat = x[0]
            x = x[1:]
        x = self.img_neck(x)
        if isinstance(x, (list, tuple)):
            x = x[0]
        _, output_dim, outH, outW = x.shape
        return x.view(B, N, output_dim, outH, outW), stereo_feat

    def extract_stereo_ref_feat(self, x):
        """/4-stride reference feature for the cost volume.

        ResNet: stem + layer1. Swin (out_indices=[0,2,3]): stage-0 output.
        """
        B, N, C, imH, imW = x.shape
        x = x.view(B * N, C, imH, imW)
        bb = self.img_backbone
        if isinstance(bb, SwinTransformer):
            return bb(x)[0]
        x = bb.conv1(x)
        x = bb.bn1(x)
        x = bb.relu(x)
        x = bb.maxpool(x)
        x = bb.layer1(x)
        return x

    def prepare_bev_feat(self, img, sensor2keyego, ego2global, intrin, post_rot,
                         post_tran, bda, mlp_input, feat_prev_iv, k2s_sensor,
                         extra_ref_frame):
        if extra_ref_frame:
            return None, None, self.extract_stereo_ref_feat(img)

        x, stereo_feat = self.image_encoder(img, stereo=True)
        metas = dict(
            k2s_sensor=k2s_sensor,
            intrins=intrin,
            post_rots=post_rot,
            post_trans=post_tran,
            frustum=self.img_view_transformer.cv_frustum.to(x),
            cv_downsample=4,
            downsample=self.img_view_transformer.downsample,
            grid_config=self.img_view_transformer.grid_config,
            cv_feat_list=[feat_prev_iv, stereo_feat],
        )
        bev_feat, depth = self.img_view_transformer(
            [x, sensor2keyego, ego2global, intrin, post_rot, post_tran, bda, mlp_input],
            metas,
        )
        if self.pre_process:
            bev_feat = self.pre_process_net(bev_feat)[0]
        return bev_feat, depth, stereo_feat

    def extract_img_feat(self, img_inputs):
        (imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans,
         bda, curr2adjsensor) = self.prepare_inputs(img_inputs, stereo=True)

        bev_feat_list = []
        depth_key_frame = None
        feat_prev_iv = None
        for fid in range(self.num_frame - 1, -1, -1):
            img = imgs[fid]
            key_frame = fid == 0
            extra_ref_frame = fid == self.num_frame - self.extra_ref_frames
            if key_frame or self.with_prev:
                sensor2keyego, ego2global = sensor2keyegos[fid], ego2globals[fid]
                if self.align_after_view_transfromation:
                    sensor2keyego, ego2global = sensor2keyegos[0], ego2globals[0]
                mlp_input = self.img_view_transformer.get_mlp_input(
                    sensor2keyegos[0], ego2globals[0], intrins[fid],
                    post_rots[fid], post_trans[fid], bda,
                )
                inputs_curr = (img, sensor2keyego, ego2global, intrins[fid],
                               post_rots[fid], post_trans[fid], bda, mlp_input,
                               feat_prev_iv, curr2adjsensor[fid], extra_ref_frame)
                if key_frame:
                    bev_feat, depth, feat_curr_iv = self.prepare_bev_feat(*inputs_curr)
                    depth_key_frame = depth
                else:
                    with torch.no_grad():
                        bev_feat, depth, feat_curr_iv = self.prepare_bev_feat(*inputs_curr)

                if not extra_ref_frame:
                    bev_feat_list.append(bev_feat)
                if not key_frame:
                    feat_prev_iv = feat_curr_iv

        if not self.with_prev:
            bev_feat_key = bev_feat_list[0]
            b, c, h, w = bev_feat_key.shape
            zeros = torch.zeros(
                [b, c * (self.num_frame - self.extra_ref_frames - 1), h, w]
            ).to(bev_feat_key)
            bev_feat_list = [zeros, bev_feat_key]

        if self.align_after_view_transfromation:
            for adj_id in range(self.num_frame - 2):
                bev_feat_list[adj_id] = self.shift_feature(
                    bev_feat_list[adj_id],
                    [sensor2keyegos[0], sensor2keyegos[self.num_frame - 2 - adj_id]],
                    bda,
                )

        # loop order (fid high->low) => bev_feat_list == [prev, key]
        bev_feat = torch.cat(bev_feat_list, dim=1)
        return self.bev_encoder(bev_feat), depth_key_frame

    def forward(self, img_inputs, gt_depth=None, **kwargs):
        """Sequential-frame img_inputs -> (BEV feature, aux losses).

        aux losses carry loss_depth (BCE depth supervision) when gt_depth is
        provided (training); empty otherwise.
        """
        bev_feat, depth = self.extract_img_feat(img_inputs)
        if self.upsample:
            bev_feat = F.interpolate(
                bev_feat, scale_factor=2, mode="bilinear", align_corners=True
            )
        losses = {}
        if gt_depth is not None:
            losses["loss_depth"] = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        return bev_feat, losses
