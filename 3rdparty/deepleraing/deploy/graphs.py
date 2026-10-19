"""Graph definitions for the module-split ONNX export / TRT runtime.

Single source of truth for the det / seg graphs: input/output tensor names
are shared by the ONNX exporter (deploy.export), the engine builder
(deploy.builder) and the TRT runtime (deploy.runtime / deploy.inferencer).
"""

DET_HEAD_KEYS = [
    "dense_heatmap",
    "heatmap",
    "query_heatmap_score",
    "center",
    "height",
    "dim",
    "rot",
    "vel",
]

# task -> {graph name: ([input names], [output names])}
GRAPH_IO = {
    "det": {
        "camera.backbone": (["img"], ["feats"]),
        "camera.vtransform": (["feats", "depth"], ["depthnet"]),
        "fuser": (["cam_bev", "lidar_bev"], ["decoder_feat"]),
        "head.bbox": (["decoder_feat"], DET_HEAD_KEYS),
    },
    "seg": {
        "camera.backbone": (["img"], ["feats"]),
        "camera.vtransform": (["feats"], ["depthnet"]),
        "fuser": (["cam_bev", "lidar_bev"], ["decoder_feat"]),
        "head.map": (["decoder_feat"], ["masks"]),
    },
}


def graph_names(task):
    """Ordered graph names for a task (export / build / load order)."""
    return list(GRAPH_IO[task])
