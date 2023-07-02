
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



## A800-DDP

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

```
deepspeed --include localhost:7 train.py \
--deepspeed ds_config_zero2.json \
--model_name_or_path /data/nfs/guodong.li/pretrain/hf-llama-model/llama-7b \
--data_path /data/nfs/guodong.li/data/alpaca_data_cleaned.json \
--output_dir /data/nfs/guodong.li/output/llama-7b-sft \
--max_steps 800 \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 4 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 1000 \
--save_total_limit 1 \
--learning_rate 2e-5 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--report_to "tensorboard" \
--gradient_checkpointing True \
--fp16 True


{'loss': 1.4187, 'learning_rate': 2e-05, 'epoch': 0.12}
{'train_runtime': 5606.7798, 'train_samples_per_second': 1.141, 'train_steps_per_second': 0.143, 'train_loss': 1.1899223804473877, 'epoch': 0.12}
100%|█████████████████████████████████████████████████████████████████████| 800/800 [1:33:26<00:00,  7.01s/it]
[2023-07-02 22:28:34,667] [INFO] [launch.py:350:main] Process 57893 exits successfully.



deepspeed --include localhost:0,1,2,3,4,5,6,7 train.py \
--deepspeed ds_config_zero2.json \
--model_name_or_path /data/nfs/guodong.li/pretrain/hf-llama-model/llama-7b \
--data_path /data/nfs/guodong.li/data/alpaca_data_cleaned.json \
--output_dir /data/nfs/guodong.li/output/llama-7b-sft \
--max_steps 100 \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 4 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 1000 \
--save_total_limit 1 \
--learning_rate 2e-5 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--report_to "tensorboard" \
--gradient_checkpointing True \
--fp16 True


{'train_runtime': 1323.6301, 'train_samples_per_second': 4.835, 'train_steps_per_second': 0.076, 'train_loss': 1.1723517608642577, 'epoch': 0.12}
100%|███████████████████████████████████████████████████████████████████████| 100/100 [22:03<00:00, 13.24s/it]
[2023-07-02 20:27:07,744] [INFO] [launch.py:350:main] Process 43249 exits successfully.

```

## H800-DDP

### 单机单卡
运行命令：
```
deepspeed --include localhost:7 train_ddp.py \
--deepspeed ds_config_zero2.json \
--model_name_or_path /home/h800/h800-work/h800-workspace/llama-13b/merge \
--data_path /home/h800/h800-work/h800-workspace/data/alpaca_data_cleaned.json \
--output_dir /home/h800/h800-work/h800-workspace/output/llama-13b-sft \
--max_steps 800 \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 4 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 1000 \
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
{'train_runtime': 7284.2064, 'train_samples_per_second': 0.879, 'train_steps_per_second': 0.11, 'train_loss': 1.0952609968185425, 'epoch': 0.12}
100%|█████████████████████████████████████████████████████████████████████████████████████| 800/800 [2:01:24<00:00,  9.11s/it]
```

显存占用：
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.105.17   Driver Version: 525.105.17   CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   7  NVIDIA H800         On   | 00000000:E2:00.0 Off |                  Off |
| N/A   44C    P0   153W / 700W |  60787MiB / 81559MiB |     23%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    7   N/A  N/A     82977      C   ...nv-py310-cu118/bin/python    60570MiB |
+-----------------------------------------------------------------------------+
```

### 单机多卡


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
--save_steps 1000 \
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
{'train_runtime': 779.6666, 'train_samples_per_second': 8.209, 'train_steps_per_second': 0.128, 'train_loss': 1.129374122619629, 'epoch': 0.12}
100%|███████████████████████████████████████████████████████████████████████████████████████| 100/100 [12:59<00:00,  7.80s/it]
```


显存占用：
```
> nvidia-smi
Fri Jun 30 19:25:22 2023
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.105.17   Driver Version: 525.105.17   CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA H800         On   | 00000000:18:00.0 Off |                  Off |
| N/A   45C    P0   218W / 700W |  65759MiB / 81559MiB |    100%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA H800         On   | 00000000:3E:00.0 Off |                  Off |
| N/A   46C    P0   191W / 700W |  65785MiB / 81559MiB |    100%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   2  NVIDIA H800         On   | 00000000:51:00.0 Off |                  Off |
| N/A   41C    P0   215W / 700W |  68925MiB / 81559MiB |    100%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   3  NVIDIA H800         On   | 00000000:65:00.0 Off |                  Off |
| N/A   40C    P0   184W / 700W |  68513MiB / 81559MiB |    100%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   4  NVIDIA H800         On   | 00000000:98:00.0 Off |                  Off |
| N/A   45C    P0   191W / 700W |  59785MiB / 81559MiB |    100%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   5  NVIDIA H800         On   | 00000000:BD:00.0 Off |                  Off |
| N/A   44C    P0   193W / 700W |  62865MiB / 81559MiB |    100%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   6  NVIDIA H800         On   | 00000000:CF:00.0 Off |                  Off |
| N/A   39C    P0   183W / 700W |  59993MiB / 81559MiB |    100%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   7  NVIDIA H800         On   | 00000000:E2:00.0 Off |                  Off |
| N/A   40C    P0   203W / 700W |  59511MiB / 81559MiB |    100%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A    112523      C   ...nv-py310-cu118/bin/python    65726MiB |
|    1   N/A  N/A    112524      C   ...nv-py310-cu118/bin/python    65752MiB |
|    2   N/A  N/A    112525      C   ...nv-py310-cu118/bin/python    68892MiB |
|    3   N/A  N/A    112526      C   ...nv-py310-cu118/bin/python    68480MiB |
|    4   N/A  N/A    112527      C   ...nv-py310-cu118/bin/python    59752MiB |
|    5   N/A  N/A    112528      C   ...nv-py310-cu118/bin/python    62832MiB |
|    6   N/A  N/A    112529      C   ...nv-py310-cu118/bin/python    59960MiB |
|    7   N/A  N/A    112530      C   ...nv-py310-cu118/bin/python    59478MiB |
+-----------------------------------------------------------------------------+
```

有效负载和协议开销：
```
> nvidia-smi nvlink -i 0 -gt d
GPU 0: NVIDIA H800 (UUID: GPU-34bf77d1-c686-6821-79a8-32d326c5039c)
         Link 0: Data Tx: 1390774681 KiB
         Link 0: Data Rx: 1387831436 KiB
         Link 1: Data Tx: 1390715554 KiB
         Link 1: Data Rx: 1387856699 KiB
         Link 2: Data Tx: 1390689916 KiB
         Link 2: Data Rx: 1387846800 KiB
         Link 3: Data Tx: 1390772616 KiB
         Link 3: Data Rx: 1387795114 KiB
         Link 4: Data Tx: 1391305436 KiB
         Link 4: Data Rx: 1387910526 KiB
         Link 5: Data Tx: 1391288579 KiB
         Link 5: Data Rx: 1387888125 KiB
         Link 6: Data Tx: 1391348992 KiB
         Link 6: Data Rx: 1387832695 KiB
         Link 7: Data Tx: 1391348007 KiB
         Link 7: Data Rx: 1387855953 KiB
> nvidia-smi nvlink -i 0 -gt r
GPU 0: NVIDIA H800 (UUID: GPU-34bf77d1-c686-6821-79a8-32d326c5039c)
         Link 0: Raw Tx: 1933426555 KiB
         Link 0: Raw Rx: 1975423057 KiB
         Link 1: Raw Tx: 1915132335 KiB
         Link 1: Raw Rx: 1958569530 KiB
         Link 2: Raw Tx: 1916865102 KiB
         Link 2: Raw Rx: 1958463156 KiB
         Link 3: Raw Tx: 1916412075 KiB
         Link 3: Raw Rx: 1958028986 KiB
         Link 4: Raw Tx: 1913329166 KiB
         Link 4: Raw Rx: 1957374521 KiB
         Link 5: Raw Tx: 1913784453 KiB
         Link 5: Raw Rx: 1957230286 KiB
         Link 6: Raw Tx: 1916726453 KiB
         Link 6: Raw Rx: 1957614726 KiB
         Link 7: Raw Tx: 1919300185 KiB
         Link 7: Raw Rx: 1957241622 KiB

```

```
> nvidia-smi nvlink -gt d
GPU 0: NVIDIA H800 (UUID: GPU-34bf77d1-c686-6821-79a8-32d326c5039c)
         Link 0: Data Tx: 695737216 KiB
         Link 0: Data Rx: 692677038 KiB
         Link 1: Data Tx: 695707855 KiB
         Link 1: Data Rx: 692688658 KiB
         Link 2: Data Tx: 695695437 KiB
         Link 2: Data Rx: 692684920 KiB
         Link 3: Data Tx: 695736221 KiB
         Link 3: Data Rx: 692657460 KiB
         Link 4: Data Tx: 696001810 KiB
         Link 4: Data Rx: 692715987 KiB
         Link 5: Data Tx: 695993350 KiB
         Link 5: Data Rx: 692704827 KiB
         Link 6: Data Tx: 696023479 KiB
         Link 6: Data Rx: 692676721 KiB
         Link 7: Data Tx: 696023012 KiB
         Link 7: Data Rx: 692688332 KiB
GPU 1: NVIDIA H800 (UUID: GPU-f5046fa5-3db4-45e8-870a-dc1376becaa5)
         Link 0: Data Tx: 696084671 KiB
         Link 0: Data Rx: 696267925 KiB
         Link 1: Data Tx: 696094870 KiB
         Link 1: Data Rx: 695949355 KiB
         Link 2: Data Tx: 696093966 KiB
         Link 2: Data Rx: 695981158 KiB
         Link 3: Data Tx: 696084771 KiB
         Link 3: Data Rx: 696258016 KiB
         Link 4: Data Tx: 696150675 KiB
         Link 4: Data Rx: 696246802 KiB
         Link 5: Data Tx: 696143361 KiB
         Link 5: Data Rx: 695989426 KiB
         Link 6: Data Tx: 696112444 KiB
         Link 6: Data Rx: 695941292 KiB
         Link 7: Data Tx: 696103562 KiB
         Link 7: Data Rx: 696238339 KiB
GPU 2: NVIDIA H800 (UUID: GPU-9de407ad-ba9c-af12-ce09-65828829a67c)
         Link 0: Data Tx: 1054002679 KiB()
         Link 0: Data Rx: 1030362744 KiB
         Link 1: Data Tx: 1055193703 KiB
         Link 1: Data Rx: 1055345162 KiB
         Link 2: Data Tx: 1053892379 KiB
         Link 2: Data Rx: 1030586401 KiB
         Link 3: Data Tx: 1053653135 KiB
         Link 3: Data Rx: 1030480973 KiB
         Link 4: Data Tx: 1030522575 KiB
         Link 4: Data Rx: 1054103382 KiB
         Link 5: Data Tx: 1030611700 KiB
         Link 5: Data Rx: 1053822819 KiB
         Link 6: Data Tx: 1030748509 KiB
         Link 6: Data Rx: 1030246144 KiB
         Link 7: Data Tx: 1030367287 KiB
         Link 7: Data Rx: 1054044341 KiB
GPU 3: NVIDIA H800 (UUID: GPU-b54d703a-dee5-a9da-aeb9-465003acdd4b)
         Link 0: Data Tx: 1054001301 KiB
         Link 0: Data Rx: 1030523294 KiB
         Link 1: Data Tx: 1055202838 KiB
         Link 1: Data Rx: 1055183348 KiB
         Link 2: Data Tx: 1053891154 KiB
         Link 2: Data Rx: 1030368684 KiB
         Link 3: Data Tx: 1053651756 KiB
         Link 3: Data Rx: 1030612272 KiB
         Link 4: Data Tx: 1030523716 KiB
         Link 4: Data Rx: 1054012179 KiB
         Link 5: Data Tx: 1030612840 KiB
         Link 5: Data Rx: 1053652291 KiB
         Link 6: Data Tx: 1030739151 KiB
         Link 6: Data Rx: 1053901558 KiB
         Link 7: Data Tx: 1030368258 KiB
         Link 7: Data Rx: 1030738386 KiB
GPU 4: NVIDIA H800 (UUID: GPU-09c6e33a-ffcf-b330-e68b-e1e9f745eae6)
         Link 0: Data Tx: 695983535 KiB
         Link 0: Data Rx: 695959477 KiB
         Link 1: Data Tx: 695954083 KiB
         Link 1: Data Rx: 696236361 KiB
         Link 2: Data Tx: 695941649 KiB
         Link 2: Data Rx: 695952294 KiB
         Link 3: Data Tx: 695982204 KiB
         Link 3: Data Rx: 696257044 KiB
         Link 4: Data Tx: 696244163 KiB
         Link 4: Data Rx: 695969585 KiB
         Link 5: Data Tx: 696235572 KiB
         Link 5: Data Rx: 696265919 KiB
         Link 6: Data Tx: 696265768 KiB
         Link 6: Data Rx: 696245293 KiB
         Link 7: Data Tx: 696265336 KiB
         Link 7: Data Rx: 695981346 KiB
GPU 5: NVIDIA H800 (UUID: GPU-9a8ef0b8-9816-459d-fa13-cda74cf19d37)
         Link 0: Data Tx: 695981354 KiB
         Link 0: Data Rx: 695951276 KiB
         Link 1: Data Tx: 695951855 KiB
         Link 1: Data Rx: 696266080 KiB
         Link 2: Data Tx: 695939389 KiB
         Link 2: Data Rx: 696244607 KiB
         Link 3: Data Tx: 695980158 KiB
         Link 3: Data Rx: 695981676 KiB
         Link 4: Data Tx: 696245738 KiB
         Link 4: Data Rx: 695963221 KiB
         Link 5: Data Tx: 696237228 KiB
         Link 5: Data Rx: 696235790 KiB
         Link 6: Data Tx: 696267509 KiB
         Link 6: Data Rx: 695983623 KiB
         Link 7: Data Tx: 696267040 KiB
         Link 7: Data Rx: 696245994 KiB
GPU 6: NVIDIA H800 (UUID: GPU-70c5b9a8-82a3-4199-d7f5-adb9186459eb)
         Link 0: Data Tx: 695982560 KiB
         Link 0: Data Rx: 695989774 KiB
         Link 1: Data Tx: 695953061 KiB
         Link 1: Data Rx: 696258815 KiB
         Link 2: Data Tx: 695940592 KiB
         Link 2: Data Rx: 696267496 KiB
         Link 3: Data Tx: 695981364 KiB
         Link 3: Data Rx: 695950474 KiB
         Link 4: Data Tx: 696244494 KiB
         Link 4: Data Rx: 696237459 KiB
         Link 5: Data Tx: 696236034 KiB
         Link 5: Data Rx: 695948053 KiB
         Link 6: Data Tx: 696266315 KiB
         Link 6: Data Rx: 696246682 KiB
         Link 7: Data Tx: 696265896 KiB
         Link 7: Data Rx: 695971565 KiB
GPU 7: NVIDIA H800 (UUID: GPU-474d838c-171f-d249-4f45-bbc01a8eb74a)
         Link 0: Data Tx: 692716997 KiB
         Link 0: Data Rx: 695992399 KiB
         Link 1: Data Tx: 692687737 KiB
         Link 1: Data Rx: 695708329 KiB
         Link 2: Data Tx: 692675318 KiB
         Link 2: Data Rx: 696013146 KiB
         Link 3: Data Tx: 692705914 KiB
         Link 3: Data Rx: 696012245 KiB
         Link 4: Data Tx: 692675705 KiB
         Link 4: Data Rx: 695715307 KiB
         Link 5: Data Tx: 692657460 KiB
         Link 5: Data Rx: 696000518 KiB
         Link 6: Data Tx: 692687639 KiB
         Link 6: Data Rx: 695737987 KiB
         Link 7: Data Tx: 692687169 KiB
         Link 7: Data Rx: 695736450 KiB
```

```
 nvidia-smi nvlink -gt r
GPU 0: NVIDIA H800 (UUID: GPU-34bf77d1-c686-6821-79a8-32d326c5039c)
         Link 0: Raw Tx: 1118147700 KiB
         Link 0: Raw Rx: 1141810354 KiB
         Link 1: Raw Tx: 1100264974 KiB
         Link 1: Raw Rx: 1124745637 KiB
         Link 2: Raw Tx: 1101254662 KiB
         Link 2: Raw Rx: 1124690315 KiB
         Link 3: Raw Tx: 1100988364 KiB
         Link 3: Raw Rx: 1124421469 KiB
         Link 4: Raw Tx: 1099230440 KiB
         Link 4: Raw Rx: 1124030263 KiB
         Link 5: Raw Tx: 1099473327 KiB
         Link 5: Raw Rx: 1123951611 KiB
         Link 6: Raw Tx: 1101179263 KiB
         Link 6: Raw Rx: 1124173618 KiB
         Link 7: Raw Tx: 1102660262 KiB
         Link 7: Raw Rx: 1123953314 KiB
GPU 1: NVIDIA H800 (UUID: GPU-f5046fa5-3db4-45e8-870a-dc1376becaa5)
         Link 0: Raw Tx: 1124539188 KiB
         Link 0: Raw Rx: 1149691251 KiB
         Link 1: Raw Tx: 1106569621 KiB
         Link 1: Raw Rx: 1132256711 KiB
         Link 2: Raw Tx: 1106684022 KiB
         Link 2: Raw Rx: 1132329569 KiB
         Link 3: Raw Tx: 1107315009 KiB
         Link 3: Raw Rx: 1132156000 KiB
         Link 4: Raw Tx: 1107035948 KiB
         Link 4: Raw Rx: 1132086945 KiB
         Link 5: Raw Tx: 1106096289 KiB
         Link 5: Raw Rx: 1132285164 KiB
         Link 6: Raw Tx: 1107377748 KiB
         Link 6: Raw Rx: 1132187161 KiB
         Link 7: Raw Tx: 1108717934 KiB
         Link 7: Raw Rx: 1132182460 KiB
GPU 2: NVIDIA H800 (UUID: GPU-9de407ad-ba9c-af12-ce09-65828829a67c)
         Link 0: Raw Tx: 1673417033 KiB
         Link 0: Raw Rx: 1714587360 KiB
         Link 1: Raw Tx: 1688236924 KiB
         Link 1: Raw Rx: 1747327387 KiB
         Link 2: Raw Tx: 1658634463 KiB
         Link 2: Raw Rx: 1698396992 KiB
         Link 3: Raw Tx: 1657197117 KiB
         Link 3: Raw Rx: 1697257614 KiB
         Link 4: Raw Tx: 1637381248 KiB
         Link 4: Raw Rx: 1705031044 KiB
         Link 5: Raw Tx: 1638024606 KiB
         Link 5: Raw Rx: 1704136803 KiB
         Link 6: Raw Tx: 1621719052 KiB
         Link 6: Raw Rx: 1671949800 KiB
         Link 7: Raw Tx: 1643219053 KiB
         Link 7: Raw Rx: 1704623653 KiB
GPU 3: NVIDIA H800 (UUID: GPU-b54d703a-dee5-a9da-aeb9-465003acdd4b)
         Link 0: Raw Tx: 1660184402 KiB
         Link 0: Raw Rx: 1704949814 KiB
         Link 1: Raw Tx: 1676194053 KiB
         Link 1: Raw Rx: 1735431635 KiB
         Link 2: Raw Tx: 1643363616 KiB
         Link 2: Raw Rx: 1687110380 KiB
         Link 3: Raw Tx: 1643714408 KiB
         Link 3: Raw Rx: 1687846412 KiB
         Link 4: Raw Tx: 1627544154 KiB
         Link 4: Raw Rx: 1697198060 KiB
         Link 5: Raw Tx: 1627175698 KiB
         Link 5: Raw Rx: 1696806696 KiB
         Link 6: Raw Tx: 1628803919 KiB
         Link 6: Raw Rx: 1698178327 KiB
         Link 7: Raw Tx: 1615146142 KiB
         Link 7: Raw Rx: 1661416229 KiB
GPU 4: NVIDIA H800 (UUID: GPU-09c6e33a-ffcf-b330-e68b-e1e9f745eae6)
         Link 0: Raw Tx: 1116896202 KiB
         Link 0: Raw Rx: 1140861139 KiB
         Link 1: Raw Tx: 1098957850 KiB
         Link 1: Raw Rx: 1123561565 KiB
         Link 2: Raw Tx: 1099305681 KiB
         Link 2: Raw Rx: 1123628097 KiB
         Link 3: Raw Tx: 1099503500 KiB
         Link 3: Raw Rx: 1123434061 KiB
         Link 4: Raw Tx: 1098724092 KiB
         Link 4: Raw Rx: 1122809689 KiB
         Link 5: Raw Tx: 1099197545 KiB
         Link 5: Raw Rx: 1123062487 KiB
         Link 6: Raw Tx: 1099519806 KiB
         Link 6: Raw Rx: 1123139720 KiB
         Link 7: Raw Tx: 1102029177 KiB
         Link 7: Raw Rx: 1122923779 KiB
GPU 5: NVIDIA H800 (UUID: GPU-9a8ef0b8-9816-459d-fa13-cda74cf19d37)
         Link 0: Raw Tx: 1117757774 KiB
         Link 0: Raw Rx: 1141564205 KiB
         Link 1: Raw Tx: 1099573176 KiB
         Link 1: Raw Rx: 1124358860 KiB
         Link 2: Raw Tx: 1100362767 KiB
         Link 2: Raw Rx: 1124040224 KiB
         Link 3: Raw Tx: 1100442868 KiB
         Link 3: Raw Rx: 1123887831 KiB
         Link 4: Raw Tx: 1099327134 KiB
         Link 4: Raw Rx: 1123351729 KiB
         Link 5: Raw Tx: 1100074359 KiB
         Link 5: Raw Rx: 1123445692 KiB
         Link 6: Raw Tx: 1100760170 KiB
         Link 6: Raw Rx: 1123527654 KiB
         Link 7: Raw Tx: 1101743143 KiB
         Link 7: Raw Rx: 1123491315 KiB
GPU 6: NVIDIA H800 (UUID: GPU-70c5b9a8-82a3-4199-d7f5-adb9186459eb)
         Link 0: Raw Tx: 1115218985 KiB
         Link 0: Raw Rx: 1138984021 KiB
         Link 1: Raw Tx: 1097337560 KiB
         Link 1: Raw Rx: 1121694117 KiB
         Link 2: Raw Tx: 1098050521 KiB
         Link 2: Raw Rx: 1121656750 KiB
         Link 3: Raw Tx: 1098194347 KiB
         Link 3: Raw Rx: 1121634496 KiB
         Link 4: Raw Tx: 1097914184 KiB
         Link 4: Raw Rx: 1120783655 KiB
         Link 5: Raw Tx: 1097139278 KiB
         Link 5: Raw Rx: 1120772788 KiB
         Link 6: Raw Tx: 1097912808 KiB
         Link 6: Raw Rx: 1120798006 KiB
         Link 7: Raw Tx: 1100057763 KiB
         Link 7: Raw Rx: 1120943586 KiB
GPU 7: NVIDIA H800 (UUID: GPU-474d838c-171f-d249-4f45-bbc01a8eb74a)
         Link 0: Raw Tx: 1113051394 KiB
         Link 0: Raw Rx: 1138308453 KiB
         Link 1: Raw Tx: 1095779616 KiB
         Link 1: Raw Rx: 1120874570 KiB
         Link 2: Raw Tx: 1096191412 KiB
         Link 2: Raw Rx: 1121071616 KiB
         Link 3: Raw Tx: 1095899434 KiB
         Link 3: Raw Rx: 1120937868 KiB
         Link 4: Raw Tx: 1094655428 KiB
         Link 4: Raw Rx: 1120572831 KiB
         Link 5: Raw Tx: 1095674926 KiB
         Link 5: Raw Rx: 1121019745 KiB
         Link 6: Raw Tx: 1095850556 KiB
         Link 6: Raw Rx: 1120703022 KiB
         Link 7: Raw Tx: 1098176499 KiB
         Link 7: Raw Rx: 1120627325 KiB

```




