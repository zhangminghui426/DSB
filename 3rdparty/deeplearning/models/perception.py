"""BevPerception — unified top-level model (det / seg / occ multi-head).

Direct composition, no registries. Encoders are optional branches:
- camera: SwinTransformer + GeneralizedLSSFPN + LSS/DepthLSS vtransform (det/seg)
- lidar:  Voxelization + SparseEncoder (det/seg)
- occ:    OccEncoder | OccEncoderStereo4D (occupancy image->BEV chain)
Heads select the task subset: object (TransFusionHead) and map
(BEVSegmentationHead) consume the fused/decoded BEV feature; occ
(BEVOCCHead2D) consumes the occ encoder's BEV feature. fuser/decoder may be
None (occ-only configs).

The parameter tree (encoders/fuser/decoder/heads) matches the official
checkpoints exactly for the det/seg branches; the occ branch remaps through
tools/remap_occ.py (single prefix). Losses are returned as a flat
{loss/<head>/<name>} dict; reduction/logging lives in scripts/trainer.py.
"""

import torch
from torch import nn
from torch.nn import functional as F

from models.backbones.resnet import ResNetBackbone
from models.backbones.second import SECOND
from models.backbones.sparse_encoder import SparseEncoder
from models.backbones.swin import SwinTransformer
from models.fusers.conv import ConvFuser
from models.heads.occ_head import BEVOCCHead2D
from models.heads.segm_vanilla import BEVSegmentationHead
from models.heads.transfusion import TransFusionHead
from models.necks.generalized_lss import GeneralizedLSSFPN
from models.necks.second import SECONDFPN
from models.occ_encoder import OccEncoder, OccEncoderStereo4D
from models.vtransforms.depth_lss import DepthLSSTransform
from models.vtransforms.lss import LSSTransform
from ops import Voxelization

__all__ = ["BevPerception", "build_model"]


def drop_type(cfg):
    """Copy a config dict without its 'type' key (dispatch happens here)."""
    cfg = dict(cfg)
    cfg.pop("type", None)
    return cfg


class BevPerception(nn.Module):
    def __init__(self, encoders, fuser=None, decoder=None, heads=None,
                 loss_scale=None):
        super().__init__()

        self.encoders = nn.ModuleDict()
        if encoders.get("camera") is not None:
            camera = encoders["camera"]
            vtransform_cfg = dict(camera["vtransform"])
            vtransform_type = vtransform_cfg.pop("type")
            if vtransform_type == "DepthLSSTransform":
                vtransform = DepthLSSTransform(**vtransform_cfg)
            elif vtransform_type == "LSSTransform":
                vtransform = LSSTransform(**vtransform_cfg)
            else:
                raise ValueError(f"unsupported vtransform: {vtransform_type}")
            backbone_cfg = dict(camera["backbone"])
            backbone_type = backbone_cfg.pop("type", "SwinTransformer")
            if backbone_type == "SwinTransformer":
                backbone = SwinTransformer(**backbone_cfg)
            elif backbone_type == "ResNetBackbone":
                backbone = ResNetBackbone(**backbone_cfg)
            else:
                raise ValueError(f"unsupported camera backbone: {backbone_type}")
            self.encoders["camera"] = nn.ModuleDict(
                {
                    "backbone": backbone,
                    "neck": GeneralizedLSSFPN(**drop_type(camera["neck"])),
                    "vtransform": vtransform,
                }
            )
        if encoders.get("lidar") is not None:
            lidar = encoders["lidar"]
            voxelize_cfg = lidar["voxelize"]
            if voxelize_cfg.get("max_num_points", -1) <= 0:
                raise NotImplementedError(
                    "DynamicScatter (max_num_points<=0) is not supported; "
                    "use a hard-voxelization config (max_num_points>0)."
                )
            self.encoders["lidar"] = nn.ModuleDict(
                {
                    "voxelize": Voxelization(**voxelize_cfg),
                    "backbone": SparseEncoder(**drop_type(lidar["backbone"])),
                }
            )
            self.voxelize_reduce = lidar.get("voxelize_reduce", True)
        if encoders.get("radar") is not None:
            raise NotImplementedError("radar encoder is not supported")
        if encoders.get("occ") is not None:
            occ_cfg = dict(encoders["occ"])
            occ_type = occ_cfg.pop("type", "OccEncoder")
            if occ_type == "OccEncoder":
                self.encoders["occ"] = OccEncoder(**occ_cfg)
            elif occ_type == "OccEncoderStereo4D":
                self.encoders["occ"] = OccEncoderStereo4D(**occ_cfg)
            else:
                raise ValueError(f"unsupported occ encoder: {occ_type}")

        self.fuser = ConvFuser(**drop_type(fuser)) if fuser is not None else None

        self.decoder = None
        if decoder is not None:
            self.decoder = nn.ModuleDict(
                {
                    "backbone": SECOND(**drop_type(decoder["backbone"])),
                    "neck": SECONDFPN(**drop_type(decoder["neck"])),
                }
            )

        self.heads = nn.ModuleDict()
        for name, head_cfg in (heads or {}).items():
            if head_cfg is None:
                continue
            if name == "object":
                self.heads["object"] = TransFusionHead(**drop_type(head_cfg))
            elif name == "map":
                self.heads["map"] = BEVSegmentationHead(**drop_type(head_cfg))
            elif name == "occ":
                self.heads["occ"] = BEVOCCHead2D(**drop_type(head_cfg))
            else:
                raise ValueError(f"unsupported head: {name}")
        assert len(self.heads) > 0, "at least one head must be enabled"

        self.loss_scale = (
            dict(loss_scale) if loss_scale is not None
            else {name: 1.0 for name in self.heads}
        )
        self.init_weights()

    def init_weights(self):
        """Scratch init lives in each module; only Swin needs an explicit call."""
        if "camera" in self.encoders:
            backbone = self.encoders["camera"]["backbone"]
            if hasattr(backbone, "init_weights"):
                backbone.init_weights()

    def extract_camera_features(
        self,
        x,
        points,
        radar_points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
        gt_depths=None,
    ):
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)

        x = self.encoders["camera"]["backbone"](x)
        x = self.encoders["camera"]["neck"](x)
        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        return self.encoders["camera"]["vtransform"](
            x,
            points,
            radar_points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            img_metas,
            depth_loss=False,
            gt_depths=gt_depths,
        )

    def extract_features(self, x, sensor):
        feats, coords, sizes = self.voxelize(x, sensor)
        batch_size = coords[-1, 0] + 1
        return self.encoders[sensor]["backbone"](feats, coords, batch_size, sizes=sizes)

    @torch.no_grad()
    def voxelize(self, points, sensor):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.encoders[sensor]["voxelize"](res.float())
            if len(ret) == 3:
                f, c, n = ret  # hard voxelize
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
                feats = feats.contiguous()
        return feats, coords, sizes

    def bev_branch(self, img, points, radar, camera2ego, lidar2ego, lidar2camera,
                   lidar2image, camera_intrinsics, camera2lidar, img_aug_matrix,
                   lidar_aug_matrix, metas, depths=None, reverse=False):
        """camera/lidar encoders -> fuser -> decoder -> shared BEV feature."""
        sensors = [s for s in self.encoders.keys() if s != "occ"]
        if reverse:
            sensors = sensors[::-1]  # avoid OOM at eval
        features = []
        for sensor in sensors:
            if sensor == "camera":
                feature = self.extract_camera_features(
                    img, points, radar, camera2ego, lidar2ego, lidar2camera,
                    lidar2image, camera_intrinsics, camera2lidar,
                    img_aug_matrix, lidar_aug_matrix, metas, gt_depths=depths,
                )
            elif sensor == "lidar":
                feature = self.extract_features(points, sensor)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")
            features.append(feature)
        if reverse:
            features = features[::-1]
        if self.fuser is not None:
            x = self.fuser(features)
        else:
            assert len(features) == 1, features
            x = features[0]
        x = self.decoder["backbone"](x)
        return self.decoder["neck"](x)

    def forward(
        self,
        img=None,
        points=None,
        camera2ego=None,
        lidar2ego=None,
        lidar2camera=None,
        lidar2image=None,
        camera_intrinsics=None,
        camera2lidar=None,
        img_aug_matrix=None,
        lidar_aug_matrix=None,
        metas=None,
        depths=None,
        radar=None,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        img_inputs=None,
        voxel_semantics=None,
        mask_camera=None,
        mask_lidar=None,
        gt_depth=None,
        **kwargs,
    ):
        """Train: flat {loss/<head>/<name>, stats/...} dict.
        Eval: list[B] of per-sample dicts (boxes_3d / masks_bev / pred_occ)."""
        has_bev = self.decoder is not None
        has_occ = "occ" in self.encoders

        if self.training:
            outputs = {}
            x = None
            if has_bev:
                x = self.bev_branch(
                    img, points, radar, camera2ego, lidar2ego, lidar2camera,
                    lidar2image, camera_intrinsics, camera2lidar,
                    img_aug_matrix, lidar_aug_matrix, metas, depths=depths,
                )
            for head_name, head in self.heads.items():
                if head_name == "occ":
                    occ_bev, aux = self.encoders["occ"](img_inputs, gt_depth=gt_depth)
                    losses = head.loss(head(occ_bev), voxel_semantics, mask_camera)
                    losses.update(aux)
                elif head_name == "object":
                    pred_dict = head(x, metas)
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                else:  # map
                    losses = head(x, gt_masks_bev)
                for name, val in losses.items():
                    if isinstance(val, torch.Tensor) and val.requires_grad:
                        outputs[f"loss/{head_name}/{name}"] = val * self.loss_scale[head_name]
                    else:
                        outputs[f"stats/{head_name}/{name}"] = val
            return outputs

        batch_size = (img.shape[0] if img is not None else img_inputs[0].shape[0])
        outputs = [{} for _ in range(batch_size)]
        if has_bev:
            x = self.bev_branch(
                img, points, radar, camera2ego, lidar2ego, lidar2camera,
                lidar2image, camera_intrinsics, camera2lidar,
                img_aug_matrix, lidar_aug_matrix, metas, depths=depths,
                reverse=True,
            )
            for head_name, head in self.heads.items():
                if head_name == "object":
                    pred_dict = head(x, metas)
                    bboxes = head.get_bboxes(pred_dict, metas)
                    for k, (boxes, scores, labels) in enumerate(bboxes):
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                            }
                        )
                elif head_name == "map":
                    logits = head(x)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                            }
                        )
        if has_occ and "occ" in self.heads:
            occ_bev, _ = self.encoders["occ"](img_inputs)
            preds = self.heads["occ"].get_occ(self.heads["occ"](occ_bev))
            for k, occ in enumerate(preds):
                outputs[k]["pred_occ"] = torch.from_numpy(occ)
        return outputs


def build_model(cfg):
    """Single model entry point: config dict -> BevPerception."""
    cfg = dict(cfg)
    model_type = cfg.pop("type", "BevPerception")
    if model_type != "BevPerception":
        raise ValueError(f"unsupported model type: {model_type}")
    return BevPerception(**cfg)
