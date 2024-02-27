#!/bin/bash

# sh bootstrap.sh -d s3://infer-test/dianxiao-data/train_data_20240105_1k.json -p s3://infer-test/dianxiao-model -o s3://infer-test/dianxiao-output -m s3://infer-test/dianxiao-output/processx.jsonca

echo "入参:" $@

for a in "$@"; do
    #echo $a
    if [[ `echo $a | grep "^--conda_env="` ]]; then
            conda_env=`echo $a | grep "^--conda_env=" | awk -F '=' '{print $2}'`
    fi
    if [[ `echo $a | grep "^--sft_type="` ]]; then
            sft_type=`echo $a | grep "^--sft_type=" | awk -F '=' '{print $2}'`
    fi
    if [[ `echo $a | grep "^--train_dataset_path="` ]]; then
            train_dataset_path=`echo $a | grep "^--train_dataset_path=" | awk -F '=' '{print $2}'`
    fi
    if [[ `echo $a | grep "^--pre_model_path="`  ]]; then
            pre_model_path=`echo $a | grep "^--pre_model_path=" | awk -F '=' '{print $2}'`
    fi
    if [[ `echo $a | grep "^--checkpoint_path="`  ]]; then
            checkpoint_path=`echo $a | grep "^--checkpoint_path=" | awk -F '=' '{print $2}'`
    fi
    if [[  `echo $a | grep "^--model_output_path="`  ]]; then
            model_output_path=`echo $a | grep "^--model_output_path=" | awk -F '=' '{print $2}'`
    fi
    if [[  `echo $a | grep "^--model_metrics_path="`  ]]; then
            model_metrics_path=`echo $a | grep "^--model_metrics_path=" | awk -F '=' '{print $2}'`
    fi
    if [[  `echo $a | grep "^--gpu_num="`  ]]; then
            gpu_num=`echo $a | grep "^--gpu_num=" | awk -F '=' '{print $2}'`
    fi
    if [[ `echo $a | grep "^--epoch="` ]]; then
            epoch=`echo $a | grep "^--epoch=" | awk -F '=' '{print $2}'`
    fi
    if [[ `echo $a | grep "^--batch_size="` ]]; then
            batch_size=`echo $a | grep "^--batch_size=" | awk -F '=' '{print $2}'`
    fi
    if [[ `echo $a | grep "^--learning_rate="`  ]]; then
            learning_rate=`echo $a | grep "^--learning_rate=" | awk -F '=' '{print $2}'`
    fi
    if [[  `echo $a | grep "^--max_seq_length="`  ]]; then
            max_seq_length=`echo $a | grep "^--max_seq_length=" | awk -F '=' '{print $2}'`
    fi
    if [[  `echo $a | grep "^--logging_steps="`  ]]; then
            logging_steps=`echo $a | grep "^--logging_steps=" | awk -F '=' '{print $2}'`
    fi
    if [[  `echo $a | grep "^--warmup_ratio="`  ]]; then
            warmup_ratio=`echo $a | grep "^--warmup_ratio=" | awk -F '=' '{print $2}'`
    fi
    if [[  `echo $a | grep "^--weight_decay="`  ]]; then
            weight_decay=`echo $a | grep "^--weight_decay=" | awk -F '=' '{print $2}'`
    fi
    if [[  `echo $a | grep "^--lora_rank="`  ]]; then
            lora_rank=`echo $a | grep "^--lora_rank=" | awk -F '=' '{print $2}'`
    fi
    if [[  `echo $a | grep "^--lora_alpha="`  ]]; then
            lora_alpha=`echo $a | grep "^--lora_alpha=" | awk -F '=' '{print $2}'`
    fi
    if [[  `echo $a | grep "^--lora_dropout="`  ]]; then
            lora_dropout=`echo $a | grep "^--lora_dropout=" | awk -F '=' '{print $2}'`
    fi
done

echo "平台入参：sft_type=$sft_type, conda_env=$conda_env, TRAIN_DATASET_PATH: $train_dataset_path ，PRE_MODEL_PATH：$pre_model_path ， checkpoint_path: $checkpoint_path, MODEL_OUTPUT_PATH：$model_output_path ，MODEL_METRICS_PATH：$model_metrics_path ，GPU_NUM：$gpu_num"
echo "模型超参：epoch: $epoch, batch_size:$batch_size, learning_rate: $learning_rate, max_seq_length: $max_seq_length , logging_steps: $logging_steps, warmup_ratio: $warmup_ratio, weight_decay: $weight_decay"
echo "模型LoRA超参：lora_rank: $lora_rank, lora_alpha: $lora_alpha, lora_dropout: $lora_dropout"

BASE_SCRIPT_PATH=/task/script
BASE_CODE_PATH=/task/code

LOCAL_TEMP_DIR=/workspace/temp
LOCAL_OUTPUT_PATH="$LOCAL_TEMP_DIR/outputs"
# 进度
LOCAL_PROGRESS_PATH=$LOCAL_OUTPUT_PATH"/progress.json"

if [ -n "$train_dataset_path" ] && [ -n "$pre_model_path" ] && [ -n "$model_output_path" ] && [ -n "$model_metrics_path" ];
then
    echo "--"
else
    echo "变量不能为空"
    exit -1
fi

if [ -n "$gpu_num" ]
then
    echo "--"
else
    echo "GPU卡参数为空，使用默认值1"
    gpu_num=1
fi

if [ -z "$epoch" ]; then
    echo "epoch 为空，使用默认值2"
    epoch=2
fi

if [ -z "$batch_size" ]; then
    echo "batch_size 为空，使用默认值1"
    batch_size=1
fi

if [ -z "$learning_rate" ]; then
    echo "learning_rate 为空，使用默认值：1e-5"
    learning_rate=1e-5
fi

if [ -z "$max_seq_length" ]; then
    echo "max_seq_length 为空，使用默认值：512"
    max_seq_length=512
fi

if [ -z "$logging_steps" ]; then
    echo "logging_steps 为空，使用默认值：1"
    logging_steps=1
fi

if [ -z "$warmup_ratio" ]; then
    echo "warmup_ratio 为空，使用默认值：0.1"
    warmup_ratio=0.1
fi

if [ -z "$weight_decay" ]; then
    echo "weight_decay 为空，使用默认值：0"
    weight_decay=0
fi

echo "模型超参：epoch: $epoch, batch_size:$batch_size learning_rate: $learning_rate, max_seq_length: $max_seq_length , logging_steps: $logging_steps, warmup_ratio: $warmup_ratio, weight_decay: $weight_decay"

source ~/.bashrc
# 切换虚拟环境
if [ -n "$conda_env" ]; then
    echo "切换虚拟环境: $conda_env"
    conda env list && conda activate $conda_env
fi

# 环境初始化
sh $BASE_SCRIPT_PATH/init_env.sh
exit_code=$?

if [ $exit_code -ne 0 ];
then
    msg="环境初始化异常退出：$exit_code, 虚拟环境: $conda_env"
    echo $msg
    echo `cat $LOCAL_PROGRESS_PATH | jq --arg errMsg "$msg" '.errMsg = $errMsg'` > $LOCAL_PROGRESS_PATH
    exit $exit_code
fi

# 下载数据
sh $BASE_SCRIPT_PATH/data_prehandle.sh --train_dataset_path=$train_dataset_path \
    --pre_model_path=$pre_model_path \
    --checkpoint_path=$checkpoint_path

exit_code=$?
if [ $exit_code -ne 0 ];
then
    echo "下载数据异常退出：$exit_code"
    cd $BASE_CODE_PATH && python progress.py --model_metrics_path $model_metrics_path
    exit $exit_code
fi

# 切换训练任务
if [ -z "$sft_type" ]; then
    echo "sft_type 为空，使用默认值：full"
    sft_type == "full"
fi

if [ "$sft_type" == "full" ]; then
    sh $BASE_SCRIPT_PATH/bootstrap-llm-full.sh --conda_env=$conda_env \
    --train_dataset_path=$train_dataset_path  \
    --pre_model_path=$pre_model_path \
    --checkpoint_path=$checkpoint_path \
    --model_output_path=$model_output_path \
    --model_metrics_path=$model_metrics_path \
    --gpu_num=$gpu_num \
    --epoch=$epoch --batch_size=$batch_size --learning_rate=$learning_rate --max_seq_length=$max_seq_length --logging_steps=$logging_steps --warmup_ratio=$warmup_ratio --weight_decay=$weight_decay
    exit_code=$?
    if [ $exit_code -ne 0 ];
    then
        echo "执行训练异常退出：$exit_code"
        cd $BASE_CODE_PATH && python progress.py --model_metrics_path $model_metrics_path
        exit $exit_code
    fi
elif [ "$sft_type" == "lora" ]; then

    if [ -z "$lora_rank" ]; then
        echo "lora_rank 为空，使用默认值：64"
        lora_rank=64
    fi
    if [ -z "$lora_alpha" ]; then
        echo "lora_alpha 为空，使用默认值：16"
        lora_alpha=16
    fi
    if [ -z "$lora_dropout" ]; then
        echo "lora_dropout 为空，使用默认值：0.05"
        lora_dropout=0
    fi
    echo "模型LoRA超参：lora_rank: $lora_rank, lora_alpha: $lora_alpha, lora_dropout: $lora_dropout"

    sh $BASE_SCRIPT_PATH/bootstrap-llm-lora.sh --conda_env=$conda_env \
    --train_dataset_path=$train_dataset_path  \
    --pre_model_path=$pre_model_path \
    --model_output_path=$model_output_path \
    --model_metrics_path=$model_metrics_path \
    --gpu_num=$gpu_num \
    --epoch=$epoch --batch_size=$batch_size --learning_rate=$learning_rate --max_seq_length=$max_seq_length --logging_steps=$logging_steps --warmup_ratio=$warmup_ratio --weight_decay=$weight_decay \
    --lora_rank=$lora_rank --lora_alpha=$lora_alpha --lora_dropout=$lora_dropout
    exit_code=$?
    if [ $exit_code -ne 0 ];
    then
        echo "执行训练异常退出：$exit_code"
        cd $BASE_CODE_PATH && python progress.py --model_metrics_path $model_metrics_path
        exit $exit_code
    fi
else
    msg="暂不支持该微调类型：$sft_type"
    echo $msg
    echo `cat $LOCAL_PROGRESS_PATH | jq --arg errMsg "$msg" '.errMsg = $errMsg'` > $LOCAL_PROGRESS_PATH
    cd $BASE_CODE_PATH && python progress.py --model_metrics_path $model_metrics_path
    exit -1
fi 

echo "------------------"
