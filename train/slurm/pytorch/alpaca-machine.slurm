#!/bin/bash

#SBATCH --job-name=alpaca #作业名称
#SBATCH --partition=a800 #分区
#SBATCH --output=log/%j.out #日志
#SBATCH --error=log/%j.err #日志
#SBATCH -n 10 #总共运行几个进程
#SBATCH --ntasks-per-node=20 #单节点运行几个进程
#SBATCH -c 3
#SBATCH --gpus-per-node=8


module load anaconda/3-2023.03

source activate
conda activate liguodong-310

module load cuda-cudnn8.9/11.7.1


cd /data/hpc/home/guodong.li/stanford_alpaca

torchrun --nproc_per_node=8 --master_port=11223 train.py \
--model_name_or_path /data/hpc/home/guodong.li/llama-7b \
--data_path /data/hpc/home/guodong.li/alpaca_data_cleaned.json \
--output_dir /data/hpc/home/guodong.li/output \
--max_steps 200 \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 2 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 100 \
--save_total_limit 1 \
--learning_rate 2e-5 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--report_to "tensorboard" \
--gradient_checkpointing True \
--fp16 True \
--deepspeed ds_config_zero2.json

