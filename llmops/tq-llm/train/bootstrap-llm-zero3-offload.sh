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

LOCAL_TEMP_DIR=/workspace/temp
LOCAL_DATASET_PATH="$LOCAL_TEMP_DIR/datas"
LOCAL_MODEL_PATH="$LOCAL_TEMP_DIR/models"
LOCAL_OUTPUT_PATH="$LOCAL_TEMP_DIR/outputs"
LOCAL_LOG_PATH="$LOCAL_TEMP_DIR/logs"
# lora
LOCAL_MERGE_PATH="$LOCAL_TEMP_DIR/merges"
# 进度
LOCAL_PROGRESS_PATH=$LOCAL_OUTPUT_PATH"/progress.json"

BASE_CODE_PATH=/task/code
PROJECT_PATH="$BASE_CODE_PATH/Firefly"

echo "train task start..."

#FILENAME=$(basename -- "$TRAIN_DATASET_PATH")
#EXTENSION="${FILENAME##*.}"
#FILENAME="${FILENAME%.*}"
#FILENAME=$FILENAME"."$EXTENSION
#LOCAL_DATASET_FILE_PATH="$LOCAL_DATASET_PATH/$FILENAME"

echo "LOCAL_DATASET_PATH: $LOCAL_DATASET_PATH"

TRAIN_ARGS_PATH="$PROJECT_PATH/sft-config.json"

cat <<EOF > $TRAIN_ARGS_PATH
{
    "output_dir": "$model_output_path",
    "logging_dir": "$LOCAL_LOG_PATH",
    "model_name_or_path": "$LOCAL_MODEL_PATH",
    "deepspeed": "./train_args/ds_z3_config_offload.json",
    "train_file": "$LOCAL_DATASET_PATH",
    "model_metrics_path": "$model_metrics_path",
    "num_train_epochs": $epoch,
    "per_device_train_batch_size": $batch_size,
    "gradient_accumulation_steps": 4,
    "learning_rate": $learning_rate,
    "max_seq_length": $max_seq_length,
    "logging_steps": $logging_steps,
    "save_steps": 500,
    "save_total_limit": 1,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": $warmup_ratio,
    "gradient_checkpointing": true,
    "disable_tqdm": false,
    "optim": "adamw_hf",
    "seed": 42,
    "fp16": true,
    "report_to": "tensorboard",
    "dataloader_num_workers": 5,
    "save_strategy": "steps",
    "weight_decay": $weight_decay,
    "max_grad_norm": 1.0,
    "remove_unused_columns": false
}
EOF

echo "训练参数: "
cat $TRAIN_ARGS_PATH

echo "执行训练任务脚本："
echo "cd $PROJECT_PATH && deepspeed --num_gpus=$gpu_num train_s3.py --train_args_file $TRAIN_ARGS_PATH"
cd $PROJECT_PATH && deepspeed --num_gpus=$gpu_num train_s3.py --train_args_file $TRAIN_ARGS_PATH
exit_code=$?
if [ $exit_code -ne 0 ];
then
    exit $exit_code
fi

ls -al -R $LOCAL_TEMP_DIR

echo "train task end..."