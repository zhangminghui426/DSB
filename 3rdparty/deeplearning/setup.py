"""Build the three CUDA extensions in place:

    TORCH_CUDA_ARCH_LIST="12.0" PATH=/usr/local/cuda-12.8/bin:$PATH \
        python setup.py build_ext --inplace
"""

import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def make_cuda_ext(name, module, sources, extra_args=None):
    extra_args = extra_args or []
    return CUDAExtension(
        name=f"{module}.{name}",
        sources=[os.path.join(*module.split("."), "src", s) for s in sources],
        define_macros=[("WITH_CUDA", None)],
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": [
                "-O3",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ] + extra_args,
        },
    )


if __name__ == "__main__":
    setup(
        name="bff-ops",
        version="0.1.0",
        description="CUDA ops for the merged bevfusion+flashocc repo",
        packages=find_packages(include=["ops", "ops.*"]),
        ext_modules=[
            make_cuda_ext(
                name="bev_pool_v2_ext",
                module="ops.bev_pool_v2",
                sources=["bev_pool.cpp", "bev_pool_cuda.cu"],
            ),
            make_cuda_ext(
                name="voxel_layer",
                module="ops.voxel",
                sources=[
                    "voxelization.cpp",
                    "scatter_points_cpu.cpp",
                    "scatter_points_cuda.cu",
                    "voxelization_cpu.cpp",
                    "voxelization_cuda.cu",
                ],
            ),
            make_cuda_ext(
                name="iou3d_cuda",
                module="ops.iou3d",
                sources=["iou3d.cpp", "iou3d_kernel.cu"],
            ),
        ],
        cmdclass={"build_ext": BuildExtension},
    )
