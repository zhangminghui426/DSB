"""Preprocessing transforms + pipeline construction.

PIPELINES maps config type strings to transform classes (det/seg + occ tasks);
build_pipeline instantiates them; run_pipeline applies them to a sample dict.
"""

from datasets.preprocess.augment import (
    GlobalRotScaleTrans,
    GridMask,
    ImageAug3D,
    ObjectPaste,
    PointShuffle,
    RandomFlip3D,
)
from datasets.preprocess.filter import (
    ObjectNameFilter,
    ObjectRangeFilter,
    PointsRangeFilter,
)
from datasets.preprocess.format import (
    Collect3D,
    DefaultFormatBundle3D,
    ImageNormalize,
    PackOcc3DInputs,
)
from datasets.preprocess.loading import (
    GTDepth,
    LoadAnnotationsBEVDepth,
    LoadBEVSegmentation,
    LoadMultiViewImageFromFiles,
    LoadOccGTFromFile,
    LoadPointsFromFile,
    LoadPointsFromMultiSweeps,
    PointToMultiViewDepth,
    PrepareImageInputs,
)

PIPELINES = {
    "LoadMultiViewImageFromFiles": LoadMultiViewImageFromFiles,
    "LoadPointsFromFile": LoadPointsFromFile,
    "LoadPointsFromMultiSweeps": LoadPointsFromMultiSweeps,
    "ObjectPaste": ObjectPaste,
    "ImageAug3D": ImageAug3D,
    "GlobalRotScaleTrans": GlobalRotScaleTrans,
    "RandomFlip3D": RandomFlip3D,
    "GridMask": GridMask,
    "PointsRangeFilter": PointsRangeFilter,
    "ObjectRangeFilter": ObjectRangeFilter,
    "ObjectNameFilter": ObjectNameFilter,
    "ImageNormalize": ImageNormalize,
    "PointShuffle": PointShuffle,
    "LoadBEVSegmentation": LoadBEVSegmentation,
    "GTDepth": GTDepth,
    "DefaultFormatBundle3D": DefaultFormatBundle3D,
    "Collect3D": Collect3D,
    "PrepareImageInputs": PrepareImageInputs,
    "LoadAnnotationsBEVDepth": LoadAnnotationsBEVDepth,
    "LoadOccGTFromFile": LoadOccGTFromFile,
    "PointToMultiViewDepth": PointToMultiViewDepth,
    "PackOcc3DInputs": PackOcc3DInputs,
}


def build_pipeline(cfg_list):
    """Instantiate transforms from config dicts ({type: name, **kwargs})."""
    pipeline = []
    for cfg in cfg_list:
        cfg = dict(cfg)
        type_name = cfg.pop("type")
        if type_name not in PIPELINES:
            raise ValueError(
                f"Unknown pipeline type '{type_name}'. Known: {sorted(PIPELINES)}"
            )
        pipeline.append(PIPELINES[type_name](**cfg))
    return pipeline


def run_pipeline(pipeline, data):
    """Apply transforms in order; short-circuits on None."""
    for transform in pipeline:
        if data is None:
            return None
        data = transform(data)
    return data
