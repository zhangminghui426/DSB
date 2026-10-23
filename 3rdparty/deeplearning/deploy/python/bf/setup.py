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
def _nvcc_args():
    args = ["-O3", "--expt-relaxed-constexpr"]
    if torch.cuda.is_available():
        cc = torch.cuda.get_device_capability(0)
        cc_num = cc[0] * 10 + cc[1]
        args += [f"-gencode=arch=compute_{cc_num},code=sm_{cc_num}",
                 f"-gencode=arch=compute_{cc_num},code=compute_{cc_num}"]
    else:
        args += ["-gencode=arch=compute_90,code=sm_90",
                 "-gencode=arch=compute_90,code=compute_90"]
    return args




setup(
    name="bevfusion_ops",
    ext_modules=[
        CUDAExtension(
            "bevfusion_ops",
            [os.path.join(SRC, "bevfusion_ops.cu")],
            extra_compile_args={
                # 按当前 GPU 架构动态生成 gencode（含 PTX 兜底），避免
                # "no kernel image is available"（如 Blackwell sm_120 不在硬编码列表）
                "nvcc": _nvcc_args(),
                "cxx": ["-O3", "-std=c++17"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
