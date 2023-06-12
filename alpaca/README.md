
# Stanford Alpaca
- 源码： https://github.com/tatsu-lab/stanford_alpaca
- commit id: `73cac8be49a66ca5d159ee9199428804e1e6aabe`



## 启动命令 

```
torchrun --nproc_per_node=8 --master_port=11223 train.py \
--model_name_or_path /data/nfs/guodong.li/pretrain/hf-llama-model/llama-7b \
--data_path /data/nfs/guodong.li/data/alpaca_data_cleaned.json \
--output_dir /data/nfs/guodong.li/output/alpaca/sft_7b \
--num_train_epochs 1 \
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
--deepspeed ds_config.json



deepspeed --num_gpus=4  train.py \
--deepspeed ds_config.json \
--model_name_or_path /data/nfs/guodong.li/pretrain/hf-llama-model/llama-7b \
--data_path /data/nfs/guodong.li/data/alpaca_data_cleaned.json \
--output_dir /data/nfs/guodong.li/output/alpaca/sft_7b \
--num_train_epochs 1 \
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
--fp16 True


deepspeed --include localhost:4,5,6,7 train.py \
--deepspeed ds_config.json \
--model_name_or_path /data/nfs/guodong.li/pretrain/hf-llama-model/llama-7b \
--data_path /data/nfs/guodong.li/data/alpaca_data_cleaned.json \
--output_dir /data/nfs/guodong.li/output/alpaca/sft_7b \
--num_train_epochs 1 \
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
--fp16 True

```
