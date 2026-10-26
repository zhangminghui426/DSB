#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# configure the environment
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. ${SCRIPT_DIR}/tool/environment.sh

if [ "$ConfigurationStatus" != "Success" ]; then
    echo "Exit due to configure failure."
    exit
fi

# tensorrt version
# version=`trtexec | grep -m 1 TensorRT | sed -n "s/.*\[TensorRT v\([0-9]*\)\].*/\1/p"`

# resnet50/resnet50-int8/swint-tiny
base=${SCRIPT_DIR}/model/$DEBUG_MODEL  # deploy/model/<name>（扁平布局）

# fp16/int8
precision=$DEBUG_PRECISION

# precision flags
trtexec_fp16_flags="--fp16"
trtexec_dynamic_flags="--fp16"
if [ "$precision" == "int8" ]; then
    trtexec_dynamic_flags="--fp16 --int8"
fi

function get_onnx_number_io(){

    # $1=model
    model=$1

    if [ ! -f "$model" ]; then
        echo The model [$model] not exists.
        return
    fi

    number_of_input=`python3 -c "import onnx;m=onnx.load('$model');print(len(m.graph.input), end='')"`
    number_of_output=`python3 -c "import onnx;m=onnx.load('$model');print(len(m.graph.output), end='')"`
    # echo The model [$model] has $number_of_input inputs and $number_of_output outputs.
}

function compile_trt_model(){

    # $1: name
    # $2: precision_flags
    # $3: number_of_input
    # $4: number_of_output
    # $5: extra_flags
    # $6: onnx_name（可选，默认 $name）
    # $7: output_name（可选，默认 $name；用于 head.bbox.fp32.plan）
    name=$1
    precision_flags=$2
    number_of_input=$3
    number_of_output=$4
    extra_flags=$5
    onnx_name=${6:-$name}
    output_name=${7:-$name}
    result_save_directory=$base/build
    mkdir -p ${SCRIPT_DIR}/model
    onnx=$base/$onnx_name.onnx

    if [ -s "${result_save_directory}/$output_name.plan" ]; then
        echo Model ${result_save_directory}/$output_name.plan already build 🙋🙋🙋.
        return
    fi
    
    # Remove the onnx dependency
    # get_onnx_number_io $onnx
    # echo $number_of_input  $number_of_output

    input_flags="--inputIOFormats="
    output_flags="--outputIOFormats="
    for i in $(seq 1 $number_of_input); do
        input_flags+=fp16:chw,
    done

    for i in $(seq 1 $number_of_output); do
        output_flags+=fp16:chw,
    done

    input_flags=${input_flags%?}
    output_flags=${output_flags%?}

    cmd="--onnx=$onnx ${precision_flags} ${input_flags} ${output_flags} ${extra_flags} \
        --saveEngine=${result_save_directory}/$output_name.plan \
        --memPoolSize=workspace:2048 --verbose --dumpLayerInfo \
        --dumpProfile --separateProfileRun \
        --profilingVerbosity=detailed --exportLayerInfo=${result_save_directory}/$output_name.json"

    mkdir -p $result_save_directory
    echo Building the model: ${result_save_directory}/$output_name.plan, this will take several minutes. Wait a moment 🤗🤗🤗~.
    trtexec $cmd > ${result_save_directory}/$output_name.log 2>&1
    if [ $? != 0 ]; then
        echo 😥 Failed to build model ${result_save_directory}/$output_name.plan.
        echo You can check the error message by ${result_save_directory}/$output_name.log
        exit 1
    fi
}

# maybe int8 / fp16
compile_trt_model "camera.backbone" "$trtexec_dynamic_flags" 2 2
compile_trt_model "fuser" "$trtexec_dynamic_flags" 2 1

# fp16 only
compile_trt_model "camera.vtransform" "$trtexec_fp16_flags" 1 1

# head.bbox 必须用 fp32 层构建（不加 --fp16）：fp16 + myelin/layernorm fusion
# 有 TensorRT 数值 bug（reg 偏差 68~133），C++ 侧（src/main.cpp）加载
# head.bbox.fp32.plan。IO 仍为 fp16:chw，与 C++ binding 的 half 类型一致。
compile_trt_model "head.bbox" "" 1 6 "" "head.bbox" "head.bbox.fp32"

# for layernorm version head.bbox.onnx, accurate but slower
# compile_trt_model "head.bbox.layernormplugin" "$trtexec_fp16_flags" 1 6 "--plugins=libcustom_layernorm.so"
