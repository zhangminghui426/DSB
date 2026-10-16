"""TransFusionHead ported from mmdet3d/models/heads/bbox/transfusion.py.

Pure PyTorch — no mmcv/mmdet/mmdet3d imports.
"""

import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models.layers import ConvBNReLU
from datasets.geometry.boxes import xywhr2xyxyr
from models.core.bbox_coder import TransFusionBBoxCoder
from models.core.assigner import build_bbox_assigner
from models.core.sampler import AssignResult, PseudoSampler
from models.core.gaussian import draw_heatmap_gaussian, gaussian_radius
from models.core.post_processing import circle_nms
from models.utils import FFN, PositionEmbeddingLearned, TransformerDecoderLayer
from models.losses import FocalLoss, GaussianFocalLoss, L1Loss

__all__ = ["TransFusionHead"]


def create_loss(cfg):
    """Instantiate a loss module from its config dict; None passes through."""
    if cfg is None:
        return None
    cfg = dict(cfg)
    loss_type = cfg.pop("type")
    if loss_type == "FocalLoss":
        return FocalLoss(**cfg)
    if loss_type == "GaussianFocalLoss":
        return GaussianFocalLoss(**cfg)
    if loss_type == "L1Loss":
        return L1Loss(**cfg)
    raise ValueError(f"unknown loss type: {loss_type}")


def multi_apply(func, *args, **kwargs):
    """Apply func elementwise across iterables; returns a tuple of lists."""
    return tuple(map(list, zip(*[func(*a, **kwargs) for a in zip(*args)])))


def clip_sigmoid(x, eps=1e-4):
    y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
    return y


class TransFusionHead(nn.Module):
    def __init__(
        self,
        num_proposals=128,
        auxiliary=True,
        in_channels=128 * 3,
        hidden_channel=128,
        num_classes=4,
        # config for Transformer
        num_decoder_layers=3,
        num_heads=8,
        nms_kernel_size=1,
        ffn_channel=256,
        dropout=0.1,
        bn_momentum=0.1,
        activation="relu",
        # config for FFN
        common_heads=dict(),
        # loss
        loss_cls=dict(type="GaussianFocalLoss", reduction="mean"),
        loss_iou=None,
        loss_bbox=dict(type="L1Loss", reduction="mean"),
        loss_heatmap=dict(type="GaussianFocalLoss", reduction="mean"),
        # others
        train_cfg=None,
        test_cfg=None,
        bbox_coder=None,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_proposals = num_proposals
        self.auxiliary = auxiliary
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.bn_momentum = bn_momentum
        self.nms_kernel_size = nms_kernel_size
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.use_sigmoid_cls = loss_cls.get("use_sigmoid", False)
        if not self.use_sigmoid_cls:
            self.num_classes += 1
        self.loss_cls = create_loss(loss_cls)
        self.loss_bbox = create_loss(loss_bbox)
        self.loss_iou = create_loss(loss_iou)
        self.loss_heatmap = create_loss(loss_heatmap)

        # Build bbox coder
        if bbox_coder is not None:
            coder_cfg = dict(bbox_coder)
            coder_cfg.pop("type", None)
            self.bbox_coder = TransFusionBBoxCoder(**coder_cfg)
        else:
            self.bbox_coder = TransFusionBBoxCoder(
                pc_range=[-54, -54],
                out_size_factor=8,
                voxel_size=[0.075, 0.075],
            )

        self.sampling = False

        # a shared convolution
        self.shared_conv = nn.Conv2d(
            in_channels, hidden_channel, kernel_size=3, padding=1, bias=True
        )

        self.heatmap_head = nn.Sequential(
            ConvBNReLU(hidden_channel, hidden_channel, 3, padding=1),
            nn.Conv2d(hidden_channel, num_classes, 3, padding=1, bias=True),
        )
        self.class_encoding = nn.Conv1d(num_classes, hidden_channel, 1)

        # transformer decoder layers for object query with LiDAR feature
        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(
                TransformerDecoderLayer(
                    hidden_channel,
                    num_heads,
                    ffn_channel,
                    dropout,
                    activation,
                    self_posembed=PositionEmbeddingLearned(2, hidden_channel),
                    cross_posembed=PositionEmbeddingLearned(2, hidden_channel),
                )
            )

        # Prediction Head
        self.prediction_heads = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            heads = copy.deepcopy(common_heads)
            heads.update(dict(heatmap=(self.num_classes, 2)))
            self.prediction_heads.append(FFN(hidden_channel, heads))

        self.init_weights()
        self.init_assigner_sampler()

        # Position Embedding for Cross-Attention, which is re-used during training
        x_size = self.test_cfg["grid_size"][0] // self.test_cfg["out_size_factor"]
        y_size = self.test_cfg["grid_size"][1] // self.test_cfg["out_size_factor"]
        self.bev_pos = self.create_2D_grid(x_size, y_size)

        self.img_feat_pos = None
        self.img_feat_collapsed_pos = None

    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        # NOTE: modified
        batch_x, batch_y = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid],
            indexing="ij",
        )
        batch_x = batch_x + 0.5
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
        return coord_base

    def init_weights(self):
        # initialize transformer
        for m in self.decoder.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)
        self.init_bn_momentum()

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def init_assigner_sampler(self):
        """Initialize the target assigner and sampler of the head."""
        if self.train_cfg is None:
            return

        if self.sampling:
            raise NotImplementedError("sampling not ported")
        else:
            self.bbox_sampler = PseudoSampler()
        assigner_cfg = self.train_cfg["assigner"] if isinstance(self.train_cfg, dict) else self.train_cfg.assigner
        if isinstance(assigner_cfg, dict):
            self.bbox_assigner = build_bbox_assigner(assigner_cfg)
        elif isinstance(assigner_cfg, list):
            self.bbox_assigner = [
                build_bbox_assigner(res) for res in assigner_cfg
            ]

    def pool(self, x, k):
        """max_pool2d(k, stride=1, padding=0); ONNX-export friendly variant
        (unfold + amax; k=1 is identity)."""
        if getattr(self, "onnx_export", False):
            if k <= 1:
                return x
            return x.unfold(2, k, 1).unfold(3, k, 1).amax(dim=(-2, -1))
        return F.max_pool2d(x, kernel_size=k, stride=1, padding=0)

    def forward_single(self, inputs, img_inputs, metas):
        """Forward function for CenterPoint.
        Args:
            inputs (torch.Tensor): Input feature map with the shape of
                [B, 512, 128(H), 128(W)].
        Returns:
            list[dict]: Output results for tasks.
        """
        batch_size = inputs.shape[0]
        lidar_feat = self.shared_conv(inputs)

        #################################
        # image to BEV
        #################################
        lidar_feat_flatten = lidar_feat.view(
            batch_size, lidar_feat.shape[1], -1
        )  # [BS, C, H*W]
        bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(lidar_feat.device)

        #################################
        # image guided query initialization
        #################################
        dense_heatmap = self.heatmap_head(lidar_feat)
        dense_heatmap_img = None
        heatmap = dense_heatmap.detach().sigmoid()
        padding = self.nms_kernel_size // 2
        local_max = torch.zeros_like(heatmap)
        # equals to nms radius = voxel_size * out_size_factor * kenel_size
        local_max_inner = self.pool(heatmap, self.nms_kernel_size)
        local_max[:, :, padding:(-padding), padding:(-padding)] = local_max_inner
        ## for Pedestrian & Traffic_cone in nuScenes
        if self.test_cfg["dataset"] == "nuScenes":
            local_max[
                :,
                8,
            ] = self.pool(heatmap[:, 8], 1)
            local_max[
                :,
                9,
            ] = self.pool(heatmap[:, 9], 1)
        elif self.test_cfg["dataset"] == "Waymo":  # for Pedestrian & Cyclist in Waymo
            local_max[
                :,
                1,
            ] = self.pool(heatmap[:, 1], 1)
            local_max[
                :,
                2,
            ] = self.pool(heatmap[:, 2], 1)
        heatmap = heatmap * (heatmap == local_max)
        heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)

        # top #num_proposals among all classes
        top_proposals = heatmap.view(batch_size, -1).argsort(dim=-1, descending=True)[
            ..., : self.num_proposals
        ]
        top_proposals_class = top_proposals // heatmap.shape[-1]
        top_proposals_index = top_proposals % heatmap.shape[-1]
        query_feat = lidar_feat_flatten.gather(
            index=top_proposals_index[:, None, :].expand(
                -1, lidar_feat_flatten.shape[1], -1
            ),
            dim=-1,
        )
        self.query_labels = top_proposals_class

        # add category embedding
        one_hot = F.one_hot(top_proposals_class, num_classes=self.num_classes).permute(
            0, 2, 1
        )
        query_cat_encoding = self.class_encoding(one_hot.float())
        query_feat += query_cat_encoding

        query_pos = bev_pos.gather(
            index=top_proposals_index[:, None, :]
            .permute(0, 2, 1)
            .expand(-1, -1, bev_pos.shape[-1]),
            dim=1,
        )

        #################################
        # transformer decoder layer (LiDAR feature as K,V)
        #################################
        ret_dicts = []
        for i in range(self.num_decoder_layers):
            prefix = "last_" if (i == self.num_decoder_layers - 1) else f"{i}head_"

            # Transformer Decoder Layer
            # :param query: B C Pq    :param query_pos: B Pq 3/6
            query_feat = self.decoder[i](
                query_feat, lidar_feat_flatten, query_pos, bev_pos
            )

            # Prediction
            res_layer = self.prediction_heads[i](query_feat)
            res_layer["center"] = res_layer["center"] + query_pos.permute(0, 2, 1)
            first_res_layer = res_layer
            ret_dicts.append(res_layer)

            # for next level positional embedding
            query_pos = res_layer["center"].detach().clone().permute(0, 2, 1)

        #################################
        # transformer decoder layer (img feature as K,V)
        #################################
        ret_dicts[0]["query_heatmap_score"] = heatmap.gather(
            index=top_proposals_index[:, None, :].expand(-1, self.num_classes, -1),
            dim=-1,
        )  # [bs, num_classes, num_proposals]
        ret_dicts[0]["dense_heatmap"] = dense_heatmap

        if self.auxiliary is False:
            # only return the results of last decoder layer
            return [ret_dicts[-1]]

        # return all the layer's results for auxiliary superivison
        new_res = {}
        for key in ret_dicts[0].keys():
            if key not in ["dense_heatmap", "dense_heatmap_old", "query_heatmap_score"]:
                new_res[key] = torch.cat(
                    [ret_dict[key] for ret_dict in ret_dicts], dim=-1
                )
            else:
                new_res[key] = ret_dicts[0][key]
        return [new_res]

    def forward(self, feats, metas=None):
        """Forward pass.
        Args:
            feats (list[torch.Tensor] | torch.Tensor): Multi-level features, e.g.,
                features produced by FPN.
            metas: Optional metadata (unused in forward_single).
        Returns:
            tuple(list[dict]): Output results. first index by level, second index by layer
        """
        if isinstance(feats, torch.Tensor):
            feats = [feats]
        res = multi_apply(self.forward_single, feats, [None], [metas])
        assert len(res) == 1, "only support one level features."
        return res

    def get_targets(self, gt_bboxes_3d, gt_labels_3d, preds_dict):
        """Generate training targets.
        Args:
            gt_bboxes_3d: List of LiDARInstance3DBoxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dict: prediction dicts.
        Returns:
            tuple[torch.Tensor]
        """
        # change preds_dict into list of dict (index by batch_id)
        list_of_pred_dict = []
        for batch_idx in range(len(gt_bboxes_3d)):
            pred_dict = {}
            for key in preds_dict[0].keys():
                pred_dict[key] = preds_dict[0][key][batch_idx: batch_idx + 1]
            list_of_pred_dict.append(pred_dict)

        assert len(gt_bboxes_3d) == len(list_of_pred_dict)

        res_tuple = multi_apply(
            self.get_targets_single,
            gt_bboxes_3d,
            gt_labels_3d,
            list_of_pred_dict,
            np.arange(len(gt_labels_3d)),
        )
        labels = torch.cat(res_tuple[0], dim=0)
        label_weights = torch.cat(res_tuple[1], dim=0)
        bbox_targets = torch.cat(res_tuple[2], dim=0)
        bbox_weights = torch.cat(res_tuple[3], dim=0)
        ious = torch.cat(res_tuple[4], dim=0)
        num_pos = np.sum(res_tuple[5])
        matched_ious = np.mean(res_tuple[6])
        heatmap = torch.cat(res_tuple[7], dim=0)
        return (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            ious,
            num_pos,
            matched_ious,
            heatmap,
        )

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d, preds_dict, batch_idx):
        """Generate training targets for a single sample."""
        num_proposals = preds_dict["center"].shape[-1]

        # get pred boxes, carefully! don't change the network outputs
        score = copy.deepcopy(preds_dict["heatmap"].detach())
        center = copy.deepcopy(preds_dict["center"].detach())
        height = copy.deepcopy(preds_dict["height"].detach())
        dim = copy.deepcopy(preds_dict["dim"].detach())
        rot = copy.deepcopy(preds_dict["rot"].detach())
        if "vel" in preds_dict.keys():
            vel = copy.deepcopy(preds_dict["vel"].detach())
        else:
            vel = None

        boxes_dict = self.bbox_coder.decode(
            score, rot, dim, center, height, vel
        )  # decode the prediction to real world metric bbox
        bboxes_tensor = boxes_dict[0]["bboxes"]
        gt_bboxes_tensor = gt_bboxes_3d.tensor.to(score.device)
        # each layer should do label assign separately.
        if self.auxiliary:
            num_layer = self.num_decoder_layers
        else:
            num_layer = 1

        assign_result_list = []
        for idx_layer in range(num_layer):
            bboxes_tensor_layer = bboxes_tensor[
                self.num_proposals * idx_layer: self.num_proposals * (idx_layer + 1), :
            ]
            score_layer = score[
                ...,
                self.num_proposals * idx_layer: self.num_proposals * (idx_layer + 1),
            ]

            assigner_type = self.train_cfg["assigner"]["type"] if isinstance(self.train_cfg, dict) else self.train_cfg.assigner.type
            if assigner_type == "HungarianAssigner3D":
                assign_result = self.bbox_assigner.assign(
                    bboxes_tensor_layer,
                    gt_bboxes_tensor,
                    gt_labels_3d,
                    score_layer,
                    self.train_cfg,
                )
            elif assigner_type == "HeuristicAssigner":
                raise NotImplementedError("HeuristicAssigner not ported")
            else:
                raise NotImplementedError
            assign_result_list.append(assign_result)

        # combine assign result of each layer
        assign_result_ensemble = AssignResult(
            num_gts=sum([res.num_gts for res in assign_result_list]),
            gt_inds=torch.cat([res.gt_inds for res in assign_result_list]),
            max_overlaps=torch.cat([res.max_overlaps for res in assign_result_list]),
            labels=torch.cat([res.labels for res in assign_result_list]),
        )
        sampling_result = self.bbox_sampler.sample(
            assign_result_ensemble, bboxes_tensor, gt_bboxes_tensor
        )
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        assert len(pos_inds) + len(neg_inds) == num_proposals

        # create target for loss computation
        bbox_targets = torch.zeros([num_proposals, self.bbox_coder.code_size]).to(
            center.device
        )
        bbox_weights = torch.zeros([num_proposals, self.bbox_coder.code_size]).to(
            center.device
        )
        ious = assign_result_ensemble.max_overlaps
        ious = torch.clamp(ious, min=0.0, max=1.0)
        labels = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)
        label_weights = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)

        if gt_labels_3d is not None:  # default label is -1
            labels += self.num_classes

        # both pos and neg have classification loss, only pos has regression and iou loss
        if len(pos_inds) > 0:
            pos_bbox_targets = self.bbox_coder.encode(sampling_result.pos_gt_bboxes)

            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            if gt_labels_3d is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels_3d[sampling_result.pos_assigned_gt_inds]
            _pw = self.train_cfg["pos_weight"] if isinstance(self.train_cfg, dict) else self.train_cfg.pos_weight
            if _pw <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = _pw

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # compute dense heatmap targets
        device = labels.device
        gt_bboxes_3d = torch.cat(
            [gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]], dim=1
        ).to(device)
        grid_size = torch.tensor(self.train_cfg["grid_size"])
        pc_range = torch.tensor(self.train_cfg["point_cloud_range"])
        voxel_size = torch.tensor(self.train_cfg["voxel_size"])
        feature_map_size = (
            grid_size[:2] // self.train_cfg["out_size_factor"]
        )  # [x_len, y_len]
        heatmap = gt_bboxes_3d.new_zeros(
            self.num_classes, feature_map_size[1], feature_map_size[0]
        )
        for idx in range(len(gt_bboxes_3d)):
            width = gt_bboxes_3d[idx][3]
            length = gt_bboxes_3d[idx][4]
            width = width / voxel_size[0] / self.train_cfg["out_size_factor"]
            length = length / voxel_size[1] / self.train_cfg["out_size_factor"]
            if width > 0 and length > 0:
                radius = gaussian_radius(
                    (length, width), min_overlap=self.train_cfg["gaussian_overlap"]
                )
                radius = max(self.train_cfg["min_radius"], int(radius))
                x, y = gt_bboxes_3d[idx][0], gt_bboxes_3d[idx][1]

                coor_x = (
                    (x - pc_range[0])
                    / voxel_size[0]
                    / self.train_cfg["out_size_factor"]
                )
                coor_y = (
                    (y - pc_range[1])
                    / voxel_size[1]
                    / self.train_cfg["out_size_factor"]
                )

                center = torch.tensor(
                    [coor_x, coor_y], dtype=torch.float32, device=device
                )
                center_int = center.to(torch.int32)

                # original
                # draw_heatmap_gaussian(heatmap[gt_labels_3d[idx]], center_int, radius)
                # NOTE: fix
                draw_heatmap_gaussian(
                    heatmap[gt_labels_3d[idx]], center_int[[1, 0]], radius
                )

        mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
        return (
            labels[None],
            label_weights[None],
            bbox_targets[None],
            bbox_weights[None],
            ious[None],
            int(pos_inds.shape[0]),
            float(mean_iou),
            heatmap[None],
        )

    def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, **kwargs):
        """Loss function for CenterHead."""
        (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            ious,
            num_pos,
            matched_ious,
            heatmap,
        ) = self.get_targets(gt_bboxes_3d, gt_labels_3d, preds_dicts[0])
        if hasattr(self, "on_the_image_mask"):
            label_weights = label_weights * self.on_the_image_mask
            bbox_weights = bbox_weights * self.on_the_image_mask[:, :, None]
            num_pos = bbox_weights.max(-1).values.sum()
        preds_dict = preds_dicts[0][0]
        loss_dict = dict()

        # compute heatmap loss
        loss_heatmap = self.loss_heatmap(
            clip_sigmoid(preds_dict["dense_heatmap"].float()),
            heatmap,
            avg_factor=max(heatmap.eq(1).float().sum().item(), 1),
        )
        loss_dict["loss_heatmap"] = loss_heatmap

        # compute loss for each layer
        for idx_layer in range(self.num_decoder_layers if self.auxiliary else 1):
            if idx_layer == self.num_decoder_layers - 1 or (
                idx_layer == 0 and self.auxiliary is False
            ):
                prefix = "layer_-1"
            else:
                prefix = f"layer_{idx_layer}"

            layer_labels = labels[
                ...,
                idx_layer * self.num_proposals: (idx_layer + 1) * self.num_proposals,
            ].reshape(-1)
            layer_label_weights = label_weights[
                ...,
                idx_layer * self.num_proposals: (idx_layer + 1) * self.num_proposals,
            ].reshape(-1)
            layer_score = preds_dict["heatmap"][
                ...,
                idx_layer * self.num_proposals: (idx_layer + 1) * self.num_proposals,
            ]
            layer_cls_score = layer_score.permute(0, 2, 1).reshape(-1, self.num_classes)
            layer_loss_cls = self.loss_cls(
                layer_cls_score.float(),
                layer_labels,
                layer_label_weights,
                avg_factor=max(num_pos, 1),
            )

            layer_center = preds_dict["center"][
                ...,
                idx_layer * self.num_proposals: (idx_layer + 1) * self.num_proposals,
            ]
            layer_height = preds_dict["height"][
                ...,
                idx_layer * self.num_proposals: (idx_layer + 1) * self.num_proposals,
            ]
            layer_rot = preds_dict["rot"][
                ...,
                idx_layer * self.num_proposals: (idx_layer + 1) * self.num_proposals,
            ]
            layer_dim = preds_dict["dim"][
                ...,
                idx_layer * self.num_proposals: (idx_layer + 1) * self.num_proposals,
            ]
            preds = torch.cat(
                [layer_center, layer_height, layer_dim, layer_rot], dim=1
            ).permute(
                0, 2, 1
            )  # [BS, num_proposals, code_size]
            if "vel" in preds_dict.keys():
                layer_vel = preds_dict["vel"][
                    ...,
                    idx_layer
                    * self.num_proposals: (idx_layer + 1)
                    * self.num_proposals,
                ]
                preds = torch.cat(
                    [layer_center, layer_height, layer_dim, layer_rot, layer_vel], dim=1
                ).permute(
                    0, 2, 1
                )  # [BS, num_proposals, code_size]
            code_weights = self.train_cfg.get("code_weights", None)
            layer_bbox_weights = bbox_weights[
                :,
                idx_layer * self.num_proposals: (idx_layer + 1) * self.num_proposals,
                :,
            ]
            layer_reg_weights = layer_bbox_weights * layer_bbox_weights.new_tensor(
                code_weights
            )
            layer_bbox_targets = bbox_targets[
                :,
                idx_layer * self.num_proposals: (idx_layer + 1) * self.num_proposals,
                :,
            ]
            layer_loss_bbox = self.loss_bbox(
                preds.float(), layer_bbox_targets, layer_reg_weights, avg_factor=max(num_pos, 1)
            )

            loss_dict[f"{prefix}_loss_cls"] = layer_loss_cls
            loss_dict[f"{prefix}_loss_bbox"] = layer_loss_bbox

        loss_dict[f"matched_ious"] = layer_loss_cls.new_tensor(matched_ious)

        return loss_dict

    def get_bboxes(self, preds_dicts, metas, img=None, rescale=False, for_roi=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
        Returns:
            list[list[dict]]: Decoded bbox, scores and labels for each layer & each batch
        """
        rets = []
        for layer_id, preds_dict in enumerate(preds_dicts):
            batch_size = preds_dict[0]["heatmap"].shape[0]
            batch_score = preds_dict[0]["heatmap"][..., -self.num_proposals:].sigmoid().float()
            one_hot = F.one_hot(
                self.query_labels, num_classes=self.num_classes
            ).permute(0, 2, 1)
            batch_score = batch_score * preds_dict[0]["query_heatmap_score"] * one_hot

            batch_center = preds_dict[0]["center"][..., -self.num_proposals:].float()
            batch_height = preds_dict[0]["height"][..., -self.num_proposals:].float()
            batch_dim = preds_dict[0]["dim"][..., -self.num_proposals:].float()
            batch_rot = preds_dict[0]["rot"][..., -self.num_proposals:].float()
            batch_vel = None
            if "vel" in preds_dict[0]:
                batch_vel = preds_dict[0]["vel"][..., -self.num_proposals:].float()

            temp = self.bbox_coder.decode(
                batch_score,
                batch_rot,
                batch_dim,
                batch_center,
                batch_height,
                batch_vel,
                filter=True,
            )

            if self.test_cfg["dataset"] == "nuScenes":
                self.tasks = [
                    dict(
                        num_class=8,
                        class_names=[],
                        indices=[0, 1, 2, 3, 4, 5, 6, 7],
                        radius=-1,
                    ),
                    dict(
                        num_class=1,
                        class_names=["pedestrian"],
                        indices=[8],
                        radius=0.175,
                    ),
                    dict(
                        num_class=1,
                        class_names=["traffic_cone"],
                        indices=[9],
                        radius=0.175,
                    ),
                ]
            elif self.test_cfg["dataset"] == "Waymo":
                self.tasks = [
                    dict(num_class=1, class_names=["Car"], indices=[0], radius=0.7),
                    dict(
                        num_class=1, class_names=["Pedestrian"], indices=[1], radius=0.7
                    ),
                    dict(num_class=1, class_names=["Cyclist"], indices=[2], radius=0.7),
                ]

            ret_layer = []
            for i in range(batch_size):
                boxes3d = temp[i]["bboxes"]
                scores = temp[i]["scores"]
                labels = temp[i]["labels"]
                ## adopt circle nms for different categories
                if self.test_cfg.get("nms_type") is not None:
                    keep_mask = torch.zeros_like(scores)
                    for task in self.tasks:
                        task_mask = torch.zeros_like(scores)
                        for cls_idx in task["indices"]:
                            task_mask += labels == cls_idx
                        task_mask = task_mask.bool()
                        if task["radius"] > 0:
                            if self.test_cfg["nms_type"] == "circle":
                                boxes_for_nms = torch.cat(
                                    [
                                        boxes3d[task_mask][:, :2],
                                        scores[:, None][task_mask],
                                    ],
                                    dim=1,
                                )
                                task_keep_indices = torch.tensor(
                                    circle_nms(
                                        boxes_for_nms.detach().cpu().numpy(),
                                        task["radius"],
                                    )
                                )
                            else:
                                from ops import nms_gpu
                                boxes_for_nms = xywhr2xyxyr(
                                    metas[i]["box_type_3d"](
                                        boxes3d[task_mask][:, :7], 7
                                    ).bev
                                )
                                top_scores = scores[task_mask]
                                task_keep_indices = nms_gpu(
                                    boxes_for_nms,
                                    top_scores,
                                    thresh=task["radius"],
                                    pre_maxsize=self.test_cfg.get("pre_maxsize"),
                                    post_max_size=self.test_cfg.get("post_maxsize"),
                                )
                        else:
                            task_keep_indices = torch.arange(task_mask.sum())
                        if task_keep_indices.shape[0] != 0:
                            keep_indices = torch.where(task_mask != 0)[0][
                                task_keep_indices
                            ]
                            keep_mask[keep_indices] = 1
                    keep_mask = keep_mask.bool()
                    ret = dict(
                        bboxes=boxes3d[keep_mask],
                        scores=scores[keep_mask],
                        labels=labels[keep_mask],
                    )
                else:  # no nms
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                ret_layer.append(ret)
            rets.append(ret_layer)
        assert len(rets) == 1
        assert len(rets[0]) == 1
        res = [
            [
                metas[0]["box_type_3d"](
                    rets[0][0]["bboxes"], box_dim=rets[0][0]["bboxes"].shape[-1]
                ),
                rets[0][0]["scores"],
                rets[0][0]["labels"].int(),
            ]
        ]
        return res
