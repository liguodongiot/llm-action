#!/bin/bash

# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Install Torch, Torch_npu, Apex
pip3 install torch-2.1.0-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

PYTORCH_MANYLINUX=pytorch_v2.1.0-6.0.rc1_py310.tar.gz
TORCH_NPU_IN_PYTORCH_MANYLINUX=torch_npu-2.1.0.post3_20240413-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
APEX_IN_PYTORCH_MANYLINUX=apex-0.1_ascend_20240413-cp310-cp310-linux_aarch64.whl
mkdir torch
cp ${PYTORCH_MANYLINUX} torch \
    && cd torch \
    && tar -xzvf ${PYTORCH_MANYLINUX} \
    && cd ..

echo "start install pytorch, wait for a minute..."
pip3 install torch/${TORCH_NPU_IN_PYTORCH_MANYLINUX} --quiet 2> /dev/null
if [ $? -eq 0 ]; then
    echo "pip3 install torchnpu successfully"
else
    echo "pip3 install torchnpu failed"
fi

pip3 install torch/${APEX_IN_PYTORCH_MANYLINUX} --quiet 2> /dev/null
if [ $? -eq 0 ]; then
    echo "pip3 install apex successfully"
else
    echo "pip3 install apex failed"
fi
rm -rf torch

# Install Ascend Cann Library
CANN_TOOKIT="Ascend-cann-toolkit_8.0.RC1_linux-aarch64.run"
CANN_KERNELS="Ascend-cann-kernels-910b_8.0.RC1_linux.run"
chmod +x *.run
yes | ./${CANN_TOOKIT} --install --quiet
toolkit_status=$?
if [ ${toolkit_status} -eq 0 ]; then
    echo "install toolkit successfully"
else
    echo "install toolkit failed with status ${toolkit_status}"
fi

yes | ./${CANN_KERNELS} --install --quiet
kernels_status=$?
if [ ${kernels_status} -eq 0 ]; then
    echo "install kernels successfully"
else
    echo "install kernels failed with status ${kernels_status}"
fi

# source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Install Atb and Model
if [ ! -d "/home/llm_model" ]; then
    rm -rf /home/llm_model
fi

mkdir -p /usr/local/Ascend/llm_model
MINDIE="Ascend-mindie_*_linux-aarch64.run"
MODEL="Ascend-mindie-atb-models_1.0.RC1_linux-aarch64_torch2.1.0-abi0.tar.gz"
tar -xzf ./${MODEL} -C /usr/local/Ascend/llm_model
yes | ./${MINDIE} --install --quiet 2> /dev/null
atb_status=$?
if [ ${atb_status} -eq 0 ]; then
    echo "install atb successfully"
else
    echo "install atb failed with status ${atb_status}"
fi

source /usr/local/Ascend/mindie/set_env.sh
source /usr/local/Ascend/llm_model/set_env.sh



