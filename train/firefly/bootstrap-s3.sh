#!/bin/bash



# sh bootstrap2.sh -d s3://infer-test/dianxiao-data/train_data_20240105_1k.json -p s3://infer-test/dianxiao-model -o s3://infer-test/dianxiao-output -m s3://infer-test/dianxiao-output/processx.jsonca

func() {
    echo "Usage:"
    echo "test.sh [-d TRAIN_DATASET_PATH] [-p PRE_MODEL_PATH] [-o MODEL_OUTPUT_PATH] [-m MODEL_METRICS_PATH]"
    echo "Description:"
    echo "TRAIN_DATASET_PATH, 训练数据集路径."
    echo "PRE_MODEL_PATH, 预训练模型路径."
    echo "MODEL_OUTPUT_PATH, 模型输出模型."
    echo "MODEL_METRICS_PATH, 模型指标路径."
    exit -1
}


while getopts 'd:p:o:m:h' OPT; do
    case $OPT in
        d) TRAIN_DATASET_PATH="$OPTARG";;
        p) PRE_MODEL_PATH="$OPTARG";;
        o) MODEL_OUTPUT_PATH="$OPTARG";;
        m) MODEL_METRICS_PATH="$OPTARG";;
        h) func;;
        ?) func;;
    esac
done

echo "入参：TRAIN_DATASET_PATH: $TRAIN_DATASET_PATH ，PRE_MODEL_PATH：$PRE_MODEL_PATH ， MODEL_OUTPUT_PATH：$MODEL_OUTPUT_PATH ，MODEL_METRICS_PATH：$MODEL_METRICS_PATH"


if [ -n "$TRAIN_DATASET_PATH" ] && [ -n "$PRE_MODEL_PATH" ] && [ -n "$MODEL_OUTPUT_PATH" ] && [ -n "$MODEL_METRICS_PATH" ];
then
    echo "--"
else 
    echo "变量不能为空"
    exit -1
fi

PROJECT_PATH=/workspace/code/Firefly
# PROJECT_PATH=/home/guodong.li/workspace/code/Firefly


LOCAL_TEMP_DIR=/workspace/temp
LOCAL_DATASET_PATH="$LOCAL_TEMP_DIR/datas"
LOCAL_MODEL_PATH="$LOCAL_TEMP_DIR/models"
LOCAL_OUTPUT_PATH="$LOCAL_TEMP_DIR/outputs"
LOCAL_LOG_PATH="$LOCAL_TEMP_DIR/logs"

rm -rf $LOCAL_TEMP_DIR
mkdir -p $LOCAL_DATASET_PATH
mkdir -p $LOCAL_MODEL_PATH
mkdir -p $LOCAL_OUTPUT_PATH
mkdir -p $LOCAL_LOG_PATH

echo "download s3 from s3..."

cd /workspace/code/Firefly && python download.py --model_name_or_path "s3://infer-test/dianxiao-model" --train_file "s3://infer-test/dianxiao-data/train_data_20240105_1k.json"

echo "执行数据和模型下载脚本："
echo "cd $PROJECT_PATH && python download.py --model_name_or_path $PRE_MODEL_PATH --train_file $TRAIN_DATASET_PATH"

echo "download data end from s3..."

echo "train task start..."


FILENAME=$(basename -- "$TRAIN_DATASET_PATH")
#EXTENSION="${FILENAME##*.}"
#FILENAME="${FILENAME%.*}"
#FILENAME=$FILENAME"."$EXTENSION

LOCAL_DATASET_FILE_PATH="$LOCAL_DATASET_PATH/$FILENAME"

echo "LOCAL_DATASET_PATH: $LOCAL_DATASET_FILE_PATH"

TRAIN_ARGS_PATH="$PROJECT_PATH/sft-config.json"

cat <<EOF > $TRAIN_ARGS_PATH
{
    "output_dir": "$MODEL_OUTPUT_PATH",
    "logging_dir": "$LOCAL_LOG_PATH",
    "model_name_or_path": "$LOCAL_MODEL_PATH,
    "deepspeed": "./train_args/ds_z3_config.json",
    "train_file": "$LOCAL_DATASET_FILE_PATH",
    "num_train_epochs": 1,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 1e-5,
    "max_seq_length": 512,
    "logging_steps": 1,
    "save_steps": 500,
    "save_total_limit": 1,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1,
    "gradient_checkpointing": false,
    "disable_tqdm": false,
    "optim": "adamw_hf",
    "seed": 42,
    "fp16": true,
    "report_to": "tensorboard",
    "dataloader_num_workers": 5,
    "save_strategy": "steps",
    "weight_decay": 0,
    "max_grad_norm": 1.0,
    "remove_unused_columns": false
}
EOF


echo "训练参数: "
cat $TRAIN_ARGS_PATH


echo "执行训练任务脚本："
echo "cd $PROJECT_PATH && source /opt/rh/devtoolset-9/enable && deepspeed --num_gpus=8 train_s3.py --train_args_file $TRAIN_ARGS_PATH"

cd /workspace/code/Firefly && source /opt/rh/devtoolset-9/enable && deepspeed --num_gpus=8 train_s3.py --train_args_file train_args/sft-docker-s3.json

echo "train task end..."