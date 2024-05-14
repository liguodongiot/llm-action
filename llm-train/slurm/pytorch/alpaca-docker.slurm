#!/bin/bash

#SBATCH --job-name=alpaca #作业名称
#SBATCH --partition=a800 #分区
#SBATCH --output=log/%j.out #日志
#SBATCH --error=log/%j.err #日志
#SBATCH -n 10 #总共运行几个进程
#SBATCH --ntasks-per-node=20 #单节点运行几个进程
#SBATCH -c 3
#SBATCH --gpus-per-node=8


sudo docker run -it --gpus all --network=host \
--shm-size 4G \
-v /data/hpc/home/guodong.li:/workspaces \
harbor.maip.io/base/pytorch-alpaca:v3 \
torchrun --nproc_per_node=8 --master_port=25001 train.py \
--model_name_or_path /workspaces/llama-7b \
--data_path /workspaces/alpaca_data_cleaned.json \
--output_dir /workspaces/output \
--bf16 True \
--max_steps 200 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 2 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 2000 \
--save_total_limit 1 \
--learning_rate 2e-5 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--report_to "tensorboard" \
--fsdp "full_shard auto_wrap" \
--fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
--tf32 True

