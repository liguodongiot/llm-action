
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



## DDP

单机多卡：
```
deepspeed --include localhost:0,1,2,3,4,5,6,7 train_ddp.py \
--deepspeed ds_config_zero2_ddp.json \
--model_name_or_path /home/guodong.li/h800-workspace/model/llama-13b \
--data_path /home/guodong.li/h800-workspace/data/alpaca_data_cleaned.json \
--output_dir /home/guodong.li/h800-workspace/output/llama-13b-sft \
--max_steps 100 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 4 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 50 \
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

多机多卡：
```
deepspeed --hostfile=/home/guodong.li/code/hostfile train_ddp.py \
--deepspeed ds_config_zero2_ddp.json \
--model_name_or_path /data/nfs/guodong.li/pretrain/hf-llama-model/llama-13b \
--data_path /data/nfs/guodong.li/data/alpaca_data_cleaned.json \
--output_dir /data/nfs/guodong.li/output/llama-13b-sft-multinode \
--max_steps 100 \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 4 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 50 \
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



## H800（单机多卡）


运行命令：
```
deepspeed --include localhost:0,1,2,3,4,5,6,7 train_ddp.py \
--deepspeed ds_config_zero2.json \
--model_name_or_path /home/h800/h800-work/h800-workspace/llama-13b/merge \
--data_path /home/h800/h800-work/h800-workspace/data/alpaca_data_cleaned.json \
--output_dir /home/h800/h800-work/h800-workspace/output/llama-13b-sft \
--max_steps 100 \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 4 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 50 \
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

训练时长：
```
{'train_runtime': 1356.6135, 'train_samples_per_second': 4.718, 'train_steps_per_second': 0.074, 'train_loss': 1.1297898769378663, 'epoch': 0.12}
100%|███████████████████████████████████████████████████████████████████████████████████████| 100/100 [22:36<00:00, 13.57s/it]
```

```
{'train_runtime': 1416.9025, 'train_samples_per_second': 4.517, 'train_steps_per_second': 0.071, 'train_loss': 1.1295330429077148, 'epoch': 0.12}
100%|███████████████████████████████████████████████████████████████████████████████████████| 100/100 [23:36<00:00, 14.17s/it]
```

显存占用：
```
> nvidia-smi
Wed Jun 28 12:49:57 2023
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.105.17   Driver Version: 525.105.17   CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA H800         Off  | 00000000:18:00.0 Off |                  Off |
| N/A   39C    P0   119W / 700W |  65759MiB / 81559MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA H800         Off  | 00000000:3E:00.0 Off |                  Off |
| N/A   40C    P0   124W / 700W |  65785MiB / 81559MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   2  NVIDIA H800         Off  | 00000000:51:00.0 Off |                  Off |
| N/A   37C    P0   120W / 700W |  68925MiB / 81559MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   3  NVIDIA H800         Off  | 00000000:65:00.0 Off |                  Off |
| N/A   36C    P0   118W / 700W |  68513MiB / 81559MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   4  NVIDIA H800         Off  | 00000000:98:00.0 Off |                  Off |
| N/A   40C    P0   119W / 700W |  59785MiB / 81559MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   5  NVIDIA H800         Off  | 00000000:BD:00.0 Off |                  Off |
| N/A   39C    P0   122W / 700W |  62864MiB / 81559MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   6  NVIDIA H800         Off  | 00000000:CF:00.0 Off |                  Off |
| N/A   36C    P0   117W / 700W |  59992MiB / 81559MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   7  NVIDIA H800         Off  | 00000000:E2:00.0 Off |                  Off |
| N/A   36C    P0   122W / 700W |  59510MiB / 81559MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A    125892      C   ...nv-py310-cu118/bin/python    65726MiB |
|    1   N/A  N/A    125893      C   ...nv-py310-cu118/bin/python    65752MiB |
|    2   N/A  N/A    125894      C   ...nv-py310-cu118/bin/python    68892MiB |
|    3   N/A  N/A    125895      C   ...nv-py310-cu118/bin/python    68480MiB |
|    4   N/A  N/A    125896      C   ...nv-py310-cu118/bin/python    59752MiB |
|    5   N/A  N/A    125897      C   ...nv-py310-cu118/bin/python    62832MiB |
|    6   N/A  N/A    125898      C   ...nv-py310-cu118/bin/python    59960MiB |
|    7   N/A  N/A    125899      C   ...nv-py310-cu118/bin/python    59478MiB |
+-----------------------------------------------------------------------------+
```








