





## 训练


- https://github.com/THUDM/ChatGLM3/blob/main/finetune_chatmodel_demo/scripts/finetune_pt_multiturn.sh

```
#! /usr/bin/env bash

set -ex

LR=1e-4
NUM_GPUS=4
MAX_SEQ_LEN=2048
DEV_BATCH_SIZE=16
GRAD_ACCUMULARION_STEPS=1
MAX_STEP=200
SAVE_INTERVAL=50

DATESTR=`date +%Y%m%d-%H%M%S`
RUN_NAME=tool_alpaca_ft
DATASET_PATH=formatted_data/tool_alpaca.jsonl

BASE_MODEL_PATH=THUDM/chatglm3-6b
OUTPUT_DIR=output/${RUN_NAME}-${DATESTR}-${LR}

mkdir -p $OUTPUT_DIR

torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS finetune.py \
    --train_format multi-turn \
    --train_file $DATASET_PATH \
    --max_seq_length $MAX_SEQ_LEN \
    --preprocessing_num_workers 1 \
    --model_name_or_path $BASE_MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size $DEV_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUMULARION_STEPS \
    --max_steps $MAX_STEP \
    --logging_steps 1 \
    --save_steps $SAVE_INTERVAL \
    --fp16 \
    --deepspeed configs/deepspeed.json 2>&1 | tee ${OUTPUT_DIR}/train.log
```

- https://github.com/baichuan-inc/Baichuan-7B/blob/main/scripts/train.sh

- https://github.com/LianjiaTech/BELLE/tree/main/train/scripts


-https://github.com/LianjiaTech/BELLE/blob/main/train/scripts/run_sft.sh 


```
#! /bin/bash
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
export WANDB_PROJECT=...
export WANDB_RUN_ID=...
export WANDB_RESUME=allow
export ABS_PATH=...
export PYTHONPATH="$ABS_PATH/BELLE/train"
model_name_or_path=/path_to_llm/hf_llama_7b/ # or bloomz-7b1-mt

train_file=belleMath.json
validation_file=belleMath-dev1K.json
output_dir="$ABS_PATH/BELLE/saved_models/${WANDB_PROJECT}_${WANDB_RUN_ID}"
mkdir -p ${output_dir}

cache_dir=hf_cache_dir
mkdir -p ${cache_dir}
cutoff_len=1024

#FT
# torchrun --nproc_per_node 8 src/entry_point/sft_train.py \
#     --ddp_timeout 36000 \
#     --model_name_or_path ${model_name_or_path} \
#     --llama \
#     --deepspeed configs/deepspeed_config.json \
#     --train_file ${train_file} \
#     --validation_file ${validation_file} \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 2 \
#     --gradient_accumulation_steps 4 \
#     --num_train_epochs 2 \
#     --model_max_length ${cutoff_len} \
#     --save_strategy "steps" \
#     --save_total_limit 3 \
#     --learning_rate 8e-6 \
#     --weight_decay 0.00001 \
#     --warmup_ratio 0.05 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 10 \
#     --evaluation_strategy "steps" \
#     --torch_dtype "bfloat16" \
#     --bf16 \
#     --seed 1234 \
#     --gradient_checkpointing \
#     --cache_dir ${cache_dir} \
#     --output_dir ${output_dir} \
#    # --use_flash_attention
#    # --resume_from_checkpoint ...


#LoRA with 8bit
# torchrun --nproc_per_node 8 src/entry_point/sft_train.py \
#     --ddp_timeout 36000 \
#     --model_name_or_path ${model_name_or_path} \
#     --llama \
#     --use_lora \
#     --use_int8_training \
#     --lora_config configs/lora_config_llama.json \
#     --train_file ${train_file} \
#     --validation_file ${validation_file} \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 8 \
#     --num_train_epochs 2 \
#     --model_max_length ${cutoff_len} \
#     --save_strategy "steps" \
#     --save_total_limit 3 \
#     --learning_rate 8e-6 \
#     --weight_decay 0.00001 \
#     --warmup_ratio 0.05 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 10 \
#     --evaluation_strategy "steps" \
#     --torch_dtype "bfloat16" \
#     --bf16 \
#     --seed 1234 \
#     --gradient_checkpointing \
#     --cache_dir ${cache_dir} \
#     --output_dir ${output_dir} \
#    # --use_flash_attention
#    # --resume_from_checkpoint ...

# LoRA without 8bit
torchrun --nproc_per_node 8 src/entry_point/sft_train.py \
    --ddp_timeout 36000 \
    --model_name_or_path ${model_name_or_path} \
    --llama \
    --use_lora \
    --deepspeed configs/deepspeed_config_stage3.json \
    --lora_config configs/lora_config_llama.json \
    --train_file ${train_file} \
    --validation_file ${validation_file} \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 10 \
    --model_max_length ${cutoff_len} \
    --save_strategy "steps" \
    --save_total_limit 3 \
    --learning_rate 3e-4 \
    --weight_decay 0.00001 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --evaluation_strategy "steps" \
    --torch_dtype "bfloat16" \
    --bf16 \
    --seed 1234 \
    --gradient_checkpointing \
    --cache_dir ${cache_dir} \
    --output_dir ${output_dir} \
   # --use_flash_attention
   # --resume_from_checkpoint ...
```



- https://github.com/yangjianxin1/Firefly







