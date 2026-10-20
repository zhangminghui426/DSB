"""SECOND 检测配置。

所有超参数集中管理，训练/预处理/模型统一从这里读取。
坐标系: x 向前, y 向左, z 向上, 雷达原点为车辆后轴中心。
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class VoxelConfig:
    """体素化参数。"""

    voxel_size: Tuple[float, float, float] = (0.1, 0.1, 0.2)
    # 标定板在 (1.73, 0.28, -0.5), 范围聚焦到 0~10m x, -5~5m y
    point_cloud_range: Tuple[float, float, float, float, float, float] = (
        0.0, -5.0, -3.0, 10.0, 5.0, 2.0,
    )
    max_num_points_per_voxel: int = 10
    max_num_voxels: int = 12000

    @property
    def grid_size(self) -> Tuple[int, int, int]:
        """体素网格尺寸 (Dx, Dy, Dz)。"""
        vs = self.voxel_size
        pc_range = self.point_cloud_range
        nx = int(round((pc_range[3] - pc_range[0]) / vs[0]))
        ny = int(round((pc_range[4] - pc_range[1]) / vs[1]))
        nz = int(round((pc_range[5] - pc_range[2]) / vs[2]))
        return (nx, ny, nz)


@dataclass
class ModelConfig:
    """模型结构参数。"""

    num_class: int = 1
    # 每个位置预测的锚框数: 1 个类别 x 2 个朝向(0/90度)
    num_anchor_per_loc: int = 2
    # anchors: [x, y, z, w(x方向), l(y方向), h(z方向), rot]
    # 标定板标注 scale=[0.55, 1.7, 1.18] => x方向0.55, y方向1.7, z方向1.18
    anchor_sizes: List[Tuple[float, float, float]] = field(
        default_factory=lambda: [(0.55, 1.7, 1.18)]
    )
    anchor_rotations: Tuple[float, ...] = (0.0, 1.5707963)

    # VFE 输出通道
    vfe_num_filters: Tuple[int, ...] = (32, 128)
    # 稀疏 backbone 通道数
    sparse_num_filters: Tuple[int, ...] = (64, 128, 256)
    # RPN 上采样后通道
    rpn_upsample_filters: Tuple[int, ...] = (128, 128, 128)
    rpn_input_channels: int = 384


@dataclass
class TrainConfig:
    """训练超参数。"""

    batch_size: int = 1
    num_epochs: int = 60
    lr: float = 0.001
    weight_decay: float = 0.0001
    # 损失权重
    cls_weight: float = 1.0
    loc_weight: float = 2.0
    # 正负样本阈值
    pos_iou_thr: float = 0.5
    neg_iou_thr: float = 0.35
    # anchor 编码均值/标准差
    loc_encode_mean: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    loc_encode_std: Tuple[float, ...] = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
    # 输出目录
    output_dir: str = "output"
    log_interval: int = 1


@dataclass
class DatasetConfig:
    """数据路径配置。"""

    pcd_dir: str = "data/pcd"
    annotation_file: str = "data/annotations.json"
    class_names: Tuple[str, ...] = ("calibration_board",)


@dataclass
class Config:
    voxel: VoxelConfig = field(default_factory=VoxelConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)


def get_config() -> Config:
    return Config()
