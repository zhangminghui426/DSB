# setup.py —— bf 自定义 CUDA 算子编译（torch cpp_extension）
#
# 用法（deploy/python/bf/ 下，conda env bevfusion 中执行）：
#   python setup.py build_ext --inplace
# 产物：bf/bevfusion_ops.*.so（与 cuda_ops.py 同目录，import 即可）
import os

import torch
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")

setup(
    name="bevfusion_ops",
    ext_modules=[
        CUDAExtension(
            "bevfusion_ops",
            [os.path.join(SRC, "bevfusion_ops.cu")],
            extra_compile_args={
                "nvcc": [
                    "-O3",
                    "--expt-relaxed-constexpr",
                    "-gencode=arch=compute_75,code=sm_75",
                    "-gencode=arch=compute_80,code=sm_80",
                    "-gencode=arch=compute_86,code=sm_86",
                    "-gencode=arch=compute_89,code=sm_89",
                    "-gencode=arch=compute_90,code=sm_90",
                ],
                "cxx": ["-O3", "-std=c++17"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
