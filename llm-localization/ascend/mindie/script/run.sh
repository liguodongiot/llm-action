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
# shellcheck disable=SC2148
SCRIPT_DIR=$(cd $(dirname $0); pwd)
TESTS_DIR=$(cd $SCRIPT_DIR/core/; pwd)

test_mode="performance"
model_type="pa"
model_name=""
weight_dir=""
data_type="fp16"
hardware_type="NPU"
chip_num=0
dataset="CEval"
batch_size=0
case_pair="[]"
use_refactor="True"
max_position_embedding=-1

function fn_prepare()
{
    if [ "$hardware_type" == "NPU" ]; then
        if [ -z "$ASCEND_HOME_PATH" ];then
            echo "env ASCEND_HOME_PATH not exists, fail"
            exit 0
        fi
        if [ -z "$ATB_HOME_PATH" ];then
            echo "env ATB_HOME_PATH not exists, fail"
            exit 0
        fi
    fi

    export INT8_FORMAT_NZ_ENABLE=1
    export PYTHONPATH="${PYTHONPATH}:$(dirname "$(readlink -f "$0")")"
    export PYTHONPATH="${PYTHONPATH}:$(dirname "$(dirname "$(dirname "$(readlink -f "$0")")")")"

    IFS="_"
    read -ra parts <<< "$1"
    model_type="${parts[0]}"
    if [ "$model_type" == "pa" ]; then
        data_type="${parts[1]}"
    fi

    test_mode="$2"
    if ! [ "$test_mode" == "performance" ]; then
        read -ra parts <<< "$2"
        test_mode="${parts[0]}"
        dataset="${parts[1]}"
    fi

    if [ "$test_mode" == "performance" ]; then
        export ATB_LLM_BENCHMARK_ENABLE=1
        export ATB_LLM_BENCHMARK_FILEPATH="${SCRIPT_DIR}/benchmark.csv"
    fi
}

function fn_run_single()
{
    test_file="${model_name}_test.py"
    test_path="${TESTS_DIR}/${test_file}"
    if [[ ! -e "$test_path" ]];then
        echo "model test file $test_path is not found."
        exit 0
    fi
    
    if [ "$chip_num" == 0 ]; then
        code_line=$(grep -A 1 "def get_chip_num(self):" "${test_path}" | tail -n 1)
        if [ -z "$code_line" ]; then
            echo "Warning: get_chip_num() not overwrite in '$test_file', use chip_num 1"
            chip_num=1
        else
            chip_num=$(echo "$code_line" | awk -F 'return ' '{print $2}')
            if ! [[ "$chip_num" =~ ^[1-9]+$ ]]; then
                echo "Error: return value of get_chip_num() in '$test_file' is not a digit."
                exit 1
            fi
        fi
    fi

    if [ "$hardware_type" == "NPU" ]; then
        if ! [ -n "$ASCEND_RT_VISIBLE_DEVICES" ]; then
            devices=""
            for ((i=0; i<chip_num-1; i++)); do
                devices+="$i,"
            done
            devices+="$((chip_num-1))"
            export ASCEND_RT_VISIBLE_DEVICES="$devices"
        fi
    
        random_port=$(( RANDOM  % 9999 + 10001 ))
        torchrun --nproc_per_node "$chip_num" --master_port $random_port "$test_path" \
        --model_type "$model_type" \
        --data_type "$data_type" \
        --test_mode "$test_mode" \
        --batch_size "$batch_size" \
        --model_name "$model_name" \
        --weight_dir "$weight_dir" \
        --dataset_name "$dataset" \
        --hardware_type $hardware_type \
        --case_pair "$case_pair" \
        --use_refactor "$use_refactor" \
        --max_position_embedding "$max_position_embedding"
    else
        if ! [ -n "$CUDA_VISIBLE_DEVICES" ]; then
            world_size_str=$(seq -s, 0 $((chip_num-1)))
            export CUDA_VISIBLE_DEVICES=$world_size_str
        fi
        echo "using cuda device $CUDA_VISIBLE_DEVICES"
        python3 "$test_path" \
        --model_type "$model_type" \
        --data_type "$data_type" \
        --test_mode "$test_mode" \
        --batch_size "$batch_size" \
        --model_name "$model_name" \
        --weight_dir "$weight_dir" \
        --dataset_name "$dataset" \
        --hardware_type $hardware_type \
        --case_pair "$case_pair" \
        --use_refactor "$use_refactor" \
        --max_position_embedding "$max_position_embedding"
    fi

    if [ $? -ne 0 ]; then
        echo "something wrong marked for CI"
        if [ "$test_modes" == "performance" ]; then
            echo "performance test end marked for CI"
        else
            echo "precision test end marked for CI"
        fi
    fi
}

function fn_run_all()
{
    for model_script in "$TESTS_DIR"/*; do
        if [ -f "$model_script" ]; then
            file_name=$(basename "$model_script")
            model_name="${file_name%%_test*}"
            fn_run_single "$model_name"
        fi
    done
}


function fn_main()
{
    if command -v nvidia-smi &> /dev/null; then
        hardware_type="GPU"
        echo "INFO: Detected NVIDIA GPU"
    else
        if command -v npu-smi info &> /dev/null; then
            echo "INFO: Detected Ascend NPU"
        else
            echo "Error: No GPU or NPU detected"
            exit 1
        fi
    fi

    if [ $# -eq 0 ]; then
        echo "Error: require parameter. Please refer to README."
        exit 1
    fi

    model_type=$1
    case "$model_type" in
        fa|pa_fp16|pa_bf16)
            echo "INFO: current model_type: $model_type"
            ;;
        *)
            echo "ERROR: invalid model_type, only support fa, pa_fp16, pa_bf16"
            ;;
    esac
    test_modes=$2
    case "$test_modes" in
        performance|simplified_GSM8K|simplified_TruthfulQA|full_CEval|full_GSM8K|full_MMLU|full_TruthfulQA|full_BoolQ|full_HumanEval)
            echo "INFO: current test_mode: $test_modes"
            ;;
        *)
            echo "ERROR: invalid test_mode, only support performance, simplified_GSM8K, simplified_TruthfulQA, \
            full_CEval, full_GSM8K, full_MMLU, full_TruthfulQA, full_BoolQ, full_HumanEval"
            exit 1
            ;;
    esac

    if [ "$test_modes" == "performance" ]; then
        case_pair=$3
        shift
    fi
  
    batch_size=$3
    model_name=$4

    if [ "$model_name" == "llama" ]; then
        use_refactor=$5
        shift
    fi

    weight_dir=$5
    echo "INFO: current batch_size: $batch_size"
    echo "INFO: current model_name: $model_name"
    echo "INFO: current weight_dir: $weight_dir"

    fn_prepare "$model_type" "$test_modes"


    if ! [[ "$6" =~ ^[1-9]+$ ]]; then
        echo "Error: input chip_num is not a digit."
        exit 1
    fi
    chip_num=$6
    echo "INFO: use input chip_num $chip_num"


    if [ $# -ge 7 ]; then
        if ! [[ "$7" =~ ^[0-9]+$ ]]; then
            echo "Error: input max_position_embedding or max_seq_len is not a digit."
            exit 1
        fi
        max_position_embedding=$7
        echo "INFO: use input max_position_embedding or max_seq_len $max_position_embedding"
    fi
    fn_run_single
}

fn_main "$@"





