"""Shared config parts for all leaf configs (det / seg / multitask / occ).

Leaf configs do `from configs.common import *` and override what they need.
NOTE for leaves: star-imported values are module-level singletons — rebind
top-level names with `dict(old, key=new)`, and deepcopy before mutating
anything nested. Runtime recipe values (epochs / optimizer / lr / ema /
grad-clip) are NOT centralized: every leaf sets them explicitly.

Only plain data (dict/list/str/num/bool/None) ends up in the loaded Config;
modules/functions/classes are skipped by scripts.config.load_config.
"""

from datasets.definitions import (MAP_CLASSES, NUM_CLASSES, OBJECT_CLASSES,
                                  OCC_CLASS_NAMES)

# ---------------------------------------------------------------- data basics
dataset_type = "NuScenesDataset"
dataset_root = "data/nuscenes/"
gt_paste_stop_epoch = -1
reduce_beams = 32
load_dim = 5
use_dim = 5
load_augmented = None
image_size = [256, 704]

augment2d = dict(
    resize=[[0.38, 0.55], [0.48, 0.48]],
    rotate=[-5.4, 5.4],
    gridmask=dict(prob=0.0, fixed_prob=True),
)
augment3d = dict(
    scale=[0.9, 1.1],
    rotate=[-0.78539816, 0.78539816],
    translate=0.5,
)

object_classes = list(OBJECT_CLASSES)
map_classes = list(MAP_CLASSES)
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False,
)

# Collect3D: matrices stacked as (B, 6, 4, 4) tensors and fed to the model
META_KEYS = [
    "camera_intrinsics",
    "camera2ego",
    "lidar2ego",
    "lidar2camera",
    "lidar2image",
    "camera2lidar",
    "img_aug_matrix",
    "lidar_aug_matrix",
]

IMG_NORM = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# ---- occupancy (BEVDet/FlashOCC protocol) ----
occ_class_names = list(OCC_CLASS_NAMES)
num_classes = NUM_CLASSES

occ_grid_config = {
    "x": [-40, 40, 0.4],
    "y": [-40, 40, 0.4],
    "z": [-1, 5.4, 6.4],      # single z bin -> channel-to-height folding
    "depth": [1.0, 45.0, 0.5],
}

occ_data_config = {
    "cams": [
        "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT",
    ],
    "Ncams": 6,
    "input_size": (256, 704),
    "src_size": (900, 1600),
    # augmentation (train only)
    "resize": (-0.06, 0.11),
    "rot": (-5.4, 5.4),
    "flip": True,
    "crop_h": (0.0, 0.0),
    "resize_test": 0.00,
}

bda_aug_conf = dict(rot_lim=(-0.0, 0.0), scale_lim=(1.0, 1.0),
                    flip_dx_ratio=0.5, flip_dy_ratio=0.5)

occ_meta_keys = [
    "sample_idx", "token", "timestamp", "scene_name", "scene_token",
    "occ_gt_path", "lidar2ego_translation", "lidar2ego_rotation",
    "ego2global_translation", "ego2global_rotation",
]

# ---------------------------------------------------------------- runtime
seed = 0
deterministic = False
cudnn_benchmark = False
load_from = None
resume_from = None
max_epochs = 20  # leaves override
val_interval = 1
val_begin = 1    # leaves override (official occ recipe: 20)
fp16 = dict(loss_scale="dynamic")   # torch GradScaler defaults (growth 2000)
checkpoint_config = dict(interval=1, max_keep_ckpts=5)
log_config = dict(interval=50, tensorboard=True)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
evaluation = dict(interval=1)


# ---------------------------------------------------------------- pipelines
def make_train_pipeline(point_cloud_range, sweeps_num, max_epochs, with_bev_seg=False):
    """Training transform chain (det/seg).

    with_bev_seg inserts LoadBEVSegmentation after GlobalRotScaleTrans (mask
    warping needs lidar_aug_matrix) and adds gt_masks_bev to Collect3D keys.
    GTDepth runs after Collect3D (consumes the collected dict).
    """
    pipeline = [
        dict(type="LoadMultiViewImageFromFiles", to_float32=True),
        dict(
            type="LoadPointsFromFile",
            coord_type="LIDAR",
            load_dim=load_dim,
            use_dim=use_dim,
            reduce_beams=reduce_beams,
            load_augmented=load_augmented,
        ),
        dict(
            type="LoadPointsFromMultiSweeps",
            sweeps_num=sweeps_num,
            load_dim=load_dim,
            use_dim=use_dim,
            reduce_beams=reduce_beams,
            pad_empty_sweeps=True,
            remove_close=True,
            load_augmented=load_augmented,
        ),
        dict(
            type="ObjectPaste",
            stop_epoch=gt_paste_stop_epoch,
            db_sampler=dict(
                dataset_root=dataset_root,
                info_path=dataset_root + "nuscenes_dbinfos_train.pkl",
                rate=1.0,
                prepare=dict(
                    filter_by_difficulty=[-1],
                    filter_by_min_points={c: 5 for c in object_classes},
                ),
                classes=object_classes,
                sample_groups=dict(
                    car=2, truck=3, construction_vehicle=7, bus=4, trailer=6,
                    barrier=2, motorcycle=6, bicycle=6, pedestrian=2, traffic_cone=2,
                ),
                points_loader=dict(
                    type="LoadPointsFromFile",
                    coord_type="LIDAR",
                    load_dim=load_dim,
                    use_dim=use_dim,
                    reduce_beams=reduce_beams,
                ),
            ),
        ),
        dict(
            type="ImageAug3D",
            final_dim=image_size,
            resize_lim=augment2d["resize"][0],
            bot_pct_lim=[0.0, 0.0],
            rot_lim=augment2d["rotate"],
            rand_flip=True,
            is_train=True,
        ),
        dict(
            type="GlobalRotScaleTrans",
            resize_lim=augment3d["scale"],
            rot_lim=augment3d["rotate"],
            trans_lim=augment3d["translate"],
            is_train=True,
        ),
    ]
    if with_bev_seg:
        pipeline.append(
            dict(
                type="LoadBEVSegmentation",
                dataset_root=dataset_root,
                xbound=[-50.0, 50.0, 0.5],
                ybound=[-50.0, 50.0, 0.5],
                classes=map_classes,
            )
        )
    collect_keys = ["img", "points", "gt_bboxes_3d", "gt_labels_3d"]
    if with_bev_seg:
        collect_keys.append("gt_masks_bev")
    pipeline += [
        dict(type="RandomFlip3D"),
        dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
        dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
        dict(type="ObjectNameFilter", classes=object_classes),
        dict(type="ImageNormalize", **IMG_NORM),
        dict(
            type="GridMask",
            use_h=True,
            use_w=True,
            max_epoch=max_epochs,
            rotate=1,
            offset=False,
            ratio=0.5,
            mode=1,
            prob=augment2d["gridmask"]["prob"],
            fixed_prob=augment2d["gridmask"]["fixed_prob"],
        ),
        dict(type="PointShuffle"),
        dict(type="DefaultFormatBundle3D", classes=object_classes),
        dict(type="Collect3D", keys=collect_keys, meta_keys=META_KEYS),
        dict(type="GTDepth", keyframe_only=True),
    ]
    return pipeline


def make_test_pipeline(point_cloud_range, with_bev_seg=False):
    """Deterministic inference chain (resize-only ImageAug3D, no GT filters)."""
    pipeline = [
        dict(type="LoadMultiViewImageFromFiles", to_float32=True),
        dict(
            type="LoadPointsFromFile",
            coord_type="LIDAR",
            load_dim=load_dim,
            use_dim=use_dim,
            reduce_beams=reduce_beams,
            load_augmented=load_augmented,
        ),
        dict(
            type="LoadPointsFromMultiSweeps",
            sweeps_num=9,
            load_dim=load_dim,
            use_dim=use_dim,
            reduce_beams=reduce_beams,
            pad_empty_sweeps=True,
            remove_close=True,
            load_augmented=load_augmented,
        ),
        dict(
            type="ImageAug3D",
            final_dim=image_size,
            resize_lim=augment2d["resize"][1],
            bot_pct_lim=[0.0, 0.0],
            rot_lim=[0.0, 0.0],
            rand_flip=False,
            is_train=False,
        ),
        dict(
            type="GlobalRotScaleTrans",
            resize_lim=[1.0, 1.0],
            rot_lim=[0.0, 0.0],
            trans_lim=0.0,
            is_train=False,
        ),
    ]
    if with_bev_seg:
        pipeline.append(
            dict(
                type="LoadBEVSegmentation",
                dataset_root=dataset_root,
                xbound=[-50.0, 50.0, 0.5],
                ybound=[-50.0, 50.0, 0.5],
                classes=map_classes,
            )
        )
    collect_keys = ["img", "points", "gt_bboxes_3d", "gt_labels_3d"]
    if with_bev_seg:
        collect_keys.append("gt_masks_bev")
    pipeline += [
        dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
        dict(type="ImageNormalize", **IMG_NORM),
        dict(type="DefaultFormatBundle3D", classes=object_classes),
        dict(type="Collect3D", keys=collect_keys, meta_keys=META_KEYS),
        dict(type="GTDepth", keyframe_only=True),
    ]
    return pipeline


def make_data_config(train_pipeline, test_pipeline, samples_per_gpu=4, workers_per_gpu=4):
    """data block: CBGS-wrapped train set + bare val/test sets (det/seg)."""
    common_ds = dict(
        dataset_root=dataset_root,
        object_classes=object_classes,
        map_classes=map_classes,
        modality=input_modality,
        box_type_3d="LiDAR",
    )
    return dict(
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        train=dict(
            type="CBGSDataset",
            dataset=dict(
                type=dataset_type,
                ann_file=dataset_root + "nuscenes_infos_train.pkl",
                pipeline=train_pipeline,
                test_mode=False,
                use_valid_flag=True,
                **common_ds,
            ),
        ),
        val=dict(
            type=dataset_type,
            ann_file=dataset_root + "nuscenes_infos_val.pkl",
            pipeline=test_pipeline,
            test_mode=False,
            **common_ds,
        ),
        test=dict(
            type=dataset_type,
            ann_file=dataset_root + "nuscenes_infos_val.pkl",
            pipeline=test_pipeline,
            test_mode=True,
            **common_ds,
        ),
    )


# ---- occupancy pipelines / data ----
def make_occ_train_pipeline(grid_config, sequential=False, data_config=None):
    dc = data_config or occ_data_config
    return [
        dict(type="PrepareImageInputs", is_train=True, data_config=dc,
             sequential=sequential),
        dict(type="LoadAnnotationsBEVDepth", bda_aug_conf=bda_aug_conf,
             is_train=True),
        dict(type="LoadOccGTFromFile"),
        dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=5, use_dim=5),
        dict(type="PointToMultiViewDepth", downsample=1, grid_config=grid_config),
        dict(type="PackOcc3DInputs",
             keys=["img_inputs", "gt_depth", "voxel_semantics", "mask_lidar",
                   "mask_camera"],
             meta_keys=occ_meta_keys),
    ]


def make_occ_test_pipeline(sequential=False, data_config=None):
    dc = data_config or occ_data_config
    return [
        dict(type="PrepareImageInputs", data_config=dc,
             sequential=sequential),
        dict(type="LoadAnnotationsBEVDepth", bda_aug_conf=bda_aug_conf,
             is_train=False),
        dict(type="PackOcc3DInputs", keys=["img_inputs"], meta_keys=occ_meta_keys),
    ]


def make_occ_data_config(train_pipeline, test_pipeline, samples_per_gpu=1,
                         workers_per_gpu=4, stereo=False,
                         multi_adj_frame_id_cfg=None):
    extra = dict(stereo=stereo, multi_adj_frame_id_cfg=multi_adj_frame_id_cfg) \
        if stereo else dict()
    return dict(
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        train=dict(
            type="NuScenesOccDataset",
            ann_file=dataset_root + "bevdetv2-nuscenes_infos_train.pkl",
            pipeline=train_pipeline, is_train=True, **extra,
        ),
        val=dict(
            type="NuScenesOccDataset",
            ann_file=dataset_root + "bevdetv2-nuscenes_infos_val.pkl",
            pipeline=test_pipeline, is_train=False, **extra,
        ),
        test=dict(
            type="NuScenesOccDataset",
            ann_file=dataset_root + "bevdetv2-nuscenes_infos_val.pkl",
            pipeline=test_pipeline, is_train=False, **extra,
        ),
    )


# ---------------------------------------------------------------- model parts
def swin_backbone(drop_path_rate=0.2, with_cp=False):
    return dict(
        type="SwinTransformer",
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=drop_path_rate,
        patch_norm=True,
        out_indices=[1, 2, 3],
        with_cp=with_cp,
    )


CAMERA_NECK = dict(
    type="GeneralizedLSSFPN",
    in_channels=[192, 384, 768],
    out_channels=256,
    start_level=0,
    num_outs=3,
    upsample_cfg=dict(mode="bilinear", align_corners=False),
)


def sparse_encoder(sparse_shape):
    return dict(
        type="SparseEncoder",
        in_channels=5,
        sparse_shape=sparse_shape,
        output_channels=128,
        encoder_channels=[[16, 16, 32], [32, 32, 64], [64, 64, 128], [128, 128]],
        encoder_paddings=[[0, 0, 1], [0, 0, 1], [0, 0, [1, 1, 0]], [0, 0]],
        block_type="basicblock",
    )


DECODER = dict(
    backbone=dict(
        type="SECOND",
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        bn_eps=0.001,
        bn_momentum=0.01,
    ),
    neck=dict(
        type="SECONDFPN",
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        bn_eps=0.001,
        bn_momentum=0.01,
        use_conv_for_no_stride=True,
    ),
)

FUSER = dict(type="ConvFuser", in_channels=[80, 256], out_channels=256)


# ---- occupancy model parts (unified BevPerception schema) ----
def make_occ_model(numC_Trans=64, head_out_dim=256, grid_config=occ_grid_config,
                   input_size=None, with_cp=True, stereo=False, num_adj=1,
                   align_after_view_transfromation=False, backbone_cfg=None,
                   neck_cfg=None, vt_in_channels=None, sid=None,
                   depthnet_cfg=None, loss_depth_weight=None, pre_process=None):
    """Occ-branch model dict: BevPerception with only the occ encoder + head.

    stereo=True switches to OccEncoderStereo4D (temporal + cost-volume depth,
    /4-stride stereo reference feature from backbone out_indices[0]).
    """
    input_size = input_size or occ_data_config["input_size"]
    if backbone_cfg is None:
        backbone_cfg = dict(
            type="ResNetBackbone",
            out_indices=[0, 2, 3] if stereo else [2, 3],
            with_cp=with_cp, pretrained=True,
        )
    if neck_cfg is None:
        neck_cfg = dict(type="CustomFPN", in_channels=[1024, 2048],
                        out_channels=256, num_outs=1, start_level=0, out_ids=[0])
    vt_in_channels = vt_in_channels or neck_cfg.get("out_channels", 256)
    vt = dict(
        type="LSSViewTransformerBEVStereo" if stereo else "LSSViewTransformer",
        grid_config=grid_config,
        input_size=input_size,
        in_channels=vt_in_channels,
        out_channels=numC_Trans,
        sid=(stereo if sid is None else sid),
        collapse_z=True,
        downsample=16,
    )
    if stereo:
        vt["loss_depth_weight"] = 0.05 if loss_depth_weight is None else loss_depth_weight
        vt["depthnet_cfg"] = depthnet_cfg or dict(
            use_dcn=False, aspp_mid_channels=96, stereo=True, bias=5.0)
    occ_encoder = dict(
        type="OccEncoderStereo4D" if stereo else "OccEncoder",
        img_backbone=backbone_cfg,
        img_neck=neck_cfg,
        img_view_transformer=vt,
        img_bev_encoder_backbone=dict(
            type="CustomResNet",
            numC_input=numC_Trans * (num_adj + 1) if stereo else numC_Trans,
            num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8],
        ),
        img_bev_encoder_neck=dict(
            type="FPN_LSS", in_channels=numC_Trans * 8 + numC_Trans * 2,
            out_channels=256,
        ),
    )
    if stereo:
        occ_encoder["num_adj"] = num_adj
        occ_encoder["align_after_view_transfromation"] = align_after_view_transfromation
        occ_encoder["pre_process"] = pre_process or dict(
            type="CustomResNet", numC_input=numC_Trans, num_layer=[1],
            num_channels=[numC_Trans], stride=[1], backbone_output_ids=[0])
    return dict(
        type="BevPerception",
        encoders=dict(occ=occ_encoder),
        fuser=None,
        decoder=None,
        heads=dict(
            object=None,
            map=None,
            occ=dict(
                type="BEVOCCHead2D", in_dim=256, out_dim=head_out_dim, Dz=16,
                num_classes=num_classes, use_mask=True, use_predicter=True,
                class_balance=False,
                loss_occ=dict(type="CrossEntropyLoss", use_sigmoid=False,
                              ignore_index=255, loss_weight=1.0),
            ),
        ),
    )
