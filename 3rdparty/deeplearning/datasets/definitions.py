"""nuScenes class / attribute definitions — single source of truth.

Detection / BEV-seg taxonomies (det, seg heads) plus the occupancy taxonomy
(occ head) and its BEV render palette. Referenced by datasets.dataset,
datasets.postprocess, scripts.render, tools and configs/common.py.
"""

# Detection classes — canonical order (label id == index). Matches configs,
# gt-database tooling and visualization.
OBJECT_CLASSES = (
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
)

# BEV segmentation classes (order == mask channel index).
MAP_CLASSES = (
    "drivable_area",
    "ped_crossing",
    "walkway",
    "stop_line",
    "carpark_area",
    "divider",
)

# Raw nuScenes category name -> detection class.
NAME_MAPPING = {
    "movable_object.barrier": "barrier",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.car": "car",
    "vehicle.construction": "construction_vehicle",
    "vehicle.motorcycle": "motorcycle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "movable_object.trafficcone": "traffic_cone",
    "vehicle.trailer": "trailer",
    "vehicle.truck": "truck",
}

# Default nuScenes attribute per class (submission formatting).
DEFAULT_ATTR = {
    "car": "vehicle.parked",
    "pedestrian": "pedestrian.moving",
    "trailer": "vehicle.parked",
    "truck": "vehicle.parked",
    "bus": "vehicle.moving",
    "motorcycle": "cycle.without_rider",
    "construction_vehicle": "vehicle.parked",
    "bicycle": "cycle.without_rider",
    "barrier": "",
    "traffic_cone": "",
}

# devkit TP-error name -> metric abbreviation.
ERR_NAME_MAPPING = {
    "trans_err": "mATE",
    "scale_err": "mASE",
    "orient_err": "mAOE",
    "vel_err": "mAVE",
    "attr_err": "mAAE",
}


# ---- occupancy task (FlashOCC protocol) ----

OCC_CLASS_NAMES = (
    "others",
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
    "driveable_surface",
    "other_flat",
    "sidewalk",
    "terrain",
    "manmade",
    "vegetation",
    "free",
)

NUM_CLASSES = len(OCC_CLASS_NAMES)  # 18
IGNORE_INDEX = 255  # CE ignore label (not present in Occ3D-style GT 0..17)

# Occupancy grid: [-40, 40] m at 0.4 m -> 200 x 200 x 16 voxels (z in [-1, 5.4])
OCC_GRID_SIZE = (200, 200, 16)

# BEV rendering colors per class (RGB, nuScenes 17-class palette + black for
# free); consumed by scripts/render.py and every visualization entry point.
OCC_COLOR_MAP = [
    (0, 0, 0),        # others
    (112, 128, 144),  # barrier
    (220, 20, 60),    # bicycle
    (255, 127, 80),   # bus
    (255, 158, 0),    # car
    (233, 150, 70),   # construction_vehicle
    (255, 61, 99),    # motorcycle
    (0, 0, 230),      # pedestrian
    (47, 79, 79),     # traffic_cone
    (255, 140, 0),    # trailer
    (255, 99, 71),    # truck
    (0, 207, 191),    # driveable_surface
    (175, 0, 75),     # other_flat
    (75, 0, 75),      # sidewalk
    (112, 180, 60),   # terrain
    (222, 184, 135),  # manmade
    (0, 175, 0),      # vegetation
    (0, 0, 0),        # free
]
