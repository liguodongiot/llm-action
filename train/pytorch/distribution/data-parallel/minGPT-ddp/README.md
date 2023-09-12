

# miniGPT

- https://pytorch.org/tutorials/intermediate/ddp_series_minGPT.html

用于训练的文件：

- trainer.py ：包含 Trainer 类，该类使用提供的数据集在模型上运行分布式训练迭代。
- model.py ：定义模型架构。
- char_dataset.py ：包含字符级数据集的 Dataset 类。
- gpt2_train_cfg.yaml ：包含数据、模型、优化器和训练运行的配置。
- main.py ：训练作业的入口点。 它设置 DDP 进程组，读取所有配置并运行训练作业。







## 单机多卡


```
> torchrun --standalone --nproc_per_node=4 main.py
WARNING:torch.distributed.run:
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
*****************************************
[2023-09-08 17:48:02,237][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 1
[2023-09-08 17:48:02,246][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 0
[2023-09-08 17:48:02,253][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 2
[2023-09-08 17:48:02,256][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 3
[2023-09-08 17:48:02,257][torch.distributed.distributed_c10d][INFO] - Rank 3: Completed store-based barrier for key:store_based_barrier_key:1 with 4 nodes.
[2023-09-08 17:48:02,257][torch.distributed.distributed_c10d][INFO] - Rank 0: Completed store-based barrier for key:store_based_barrier_key:1 with 4 nodes.
[2023-09-08 17:48:02,258][torch.distributed.distributed_c10d][INFO] - Rank 1: Completed store-based barrier for key:store_based_barrier_key:1 with 4 nodes.
[2023-09-08 17:48:02,264][torch.distributed.distributed_c10d][INFO] - Rank 2: Completed store-based barrier for key:store_based_barrier_key:1 with 4 nodes.
Data has 55769 characters, 59 unique.
Data has 55769 characters, 59 unique.
Data has 55769 characters, 59 unique.
Data has 55769 characters, 59 unique.
number of parameters: 27.32M
number of parameters: 27.32M
number of parameters: 27.32M
number of parameters: 27.32M
Snapshot not found. Training model from scratch
Snapshot not found. Training model from scratch
Snapshot not found. Training model from scratch
Snapshot not found. Training model from scratch
[GPU3] Epoch 1 | Iter 0 | Train Loss 4.18844
[GPU0] Epoch 1 | Iter 0 | Train Loss 4.18055
[2023-09-08 17:48:06,586][torch.nn.parallel.distributed][INFO] - Reducer buckets have been rebuilt in this iteration.
[GPU2] Epoch 1 | Iter 0 | Train Loss 4.18575
[GPU1] Epoch 1 | Iter 0 | Train Loss 4.18100
[2023-09-08 17:48:06,590][torch.nn.parallel.distributed][INFO] - Reducer buckets have been rebuilt in this iteration.
[2023-09-08 17:48:06,590][torch.nn.parallel.distributed][INFO] - Reducer buckets have been rebuilt in this iteration.
[2023-09-08 17:48:06,590][torch.nn.parallel.distributed][INFO] - Reducer buckets have been rebuilt in this iteration.
[GPU0] Epoch 1 | Iter 0 | Eval Loss 2.35044
[GPU2] Epoch 1 | Iter 0 | Eval Loss 2.33801
...
[GPU1] Epoch 3 | Iter 0 | Train Loss 2.21209
[GPU2] Epoch 3 | Iter 0 | Train Loss 2.21229
Snapshot saved at epoch 3
[GPU2] Epoch 3 | Iter 0 | Eval Loss 2.12978
[GPU1] Epoch 3 | Iter 0 | Eval Loss 2.12159
...
[GPU2] Epoch 6 | Iter 0 | Train Loss 1.96697
[GPU1] Epoch 6 | Iter 0 | Train Loss 1.97281
Snapshot saved at epoch 6
[GPU2] Epoch 6 | Iter 0 | Eval Loss 1.84860
[GPU3] Epoch 6 | Iter 0 | Eval Loss 1.85607
[GPU1] Epoch 6 | Iter 0 | Eval Loss 1.86408
[GPU0] Epoch 6 | Iter 0 | Eval Loss 1.87735
...
[GPU2] Epoch 9 | Iter 0 | Train Loss 1.43359
[GPU0] Epoch 9 | Iter 0 | Train Loss 1.45036
[GPU1] Epoch 9 | Iter 0 | Train Loss 1.44486
[GPU3] Epoch 9 | Iter 0 | Train Loss 1.41639
Snapshot saved at epoch 9
[GPU2] Epoch 9 | Iter 0 | Eval Loss 1.28527
[GPU3] Epoch 9 | Iter 0 | Eval Loss 1.29209
...
[GPU3] Epoch 10 | Iter 0 | Eval Loss 1.13304
[GPU1] Epoch 10 | Iter 0 | Eval Loss 1.13287
```

## 多机多卡


### 方案一

```
export NCCL_IB_DISABLE=1 && export NCCL_SOCKET_IFNAME=bond0 && torchrun \
--nproc_per_node=2 --nnodes=2 --node_rank=0 \
--master_addr=xx.99.2.xx --master_port=29500 \
main.py


export NCCL_IB_DISABLE=1 && export NCCL_SOCKET_IFNAME=bond0 && torchrun \
--nproc_per_node=2 --nnodes=2 --node_rank=1 \
--master_addr=xx.99.2.xx --master_port=29500 \
main.py
```


### 方案二

```
slurm/sbatch_run.sh
```

日志：

```
> tail -100f slurm-949.out
0
ai-app-2-45 ai-app-2-46
ai-app-2-45
2
WARNING:torch.distributed.run:
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
*****************************************
WARNING:torch.distributed.run:
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
*****************************************
[2023-09-11 20:52:55,311][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 0
[2023-09-11 20:52:55,411][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 2
[2023-09-11 20:52:55,545][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 7
[2023-09-11 20:52:55,548][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 6
[2023-09-11 20:52:55,553][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 5
[2023-09-11 20:52:55,555][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 4
[2023-09-11 20:52:55,590][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 1
[2023-09-11 20:52:55,593][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 3
[2023-09-11 20:52:55,593][torch.distributed.distributed_c10d][INFO] - Rank 3: Completed store-based barrier for key:store_based_barrier_key:1 with 8 nodes.
[2023-09-11 20:52:55,594][torch.distributed.distributed_c10d][INFO] - Rank 2: Completed store-based barrier for key:store_based_barrier_key:1 with 8 nodes.
[2023-09-11 20:52:55,594][torch.distributed.distributed_c10d][INFO] - Rank 5: Completed store-based barrier for key:store_based_barrier_key:1 with 8 nodes.
[2023-09-11 20:52:55,595][torch.distributed.distributed_c10d][INFO] - Rank 0: Completed store-based barrier for key:store_based_barrier_key:1 with 8 nodes.
[2023-09-11 20:52:55,596][torch.distributed.distributed_c10d][INFO] - Rank 4: Completed store-based barrier for key:store_based_barrier_key:1 with 8 nodes.
[2023-09-11 20:52:55,596][torch.distributed.distributed_c10d][INFO] - Rank 7: Completed store-based barrier for key:store_based_barrier_key:1 with 8 nodes.
[2023-09-11 20:52:55,599][torch.distributed.distributed_c10d][INFO] - Rank 6: Completed store-based barrier for key:store_based_barrier_key:1 with 8 nodes.
[2023-09-11 20:52:55,601][torch.distributed.distributed_c10d][INFO] - Rank 1: Completed store-based barrier for key:store_based_barrier_key:1 with 8 nodes.
Data has 55769 characters, 59 unique.
Data has 55769 characters, 59 unique.
Data has 55769 characters, 59 unique.
Data has 55769 characters, 59 unique.
Data has 55769 characters, 59 unique.
Data has 55769 characters, 59 unique.
Data has 55769 characters, 59 unique.
Data has 55769 characters, 59 unique.
number of parameters: 27.32M
number of parameters: 27.32M
number of parameters: 27.32M
number of parameters: 27.32M
number of parameters: 27.32M
number of parameters: 27.32M
number of parameters: 27.32M
number of parameters: 27.32M
Snapshot not found. Training model from scratch
Snapshot not found. Training model from scratch
Snapshot not found. Training model from scratch
Snapshot not found. Training model from scratch
Snapshot not found. Training model from scratch
Snapshot not found. Training model from scratch
Snapshot not found. Training model from scratch
Snapshot not found. Training model from scratch
[GPU5] Epoch 1 | Iter 0 | Train Loss 4.15149
[GPU4] Epoch 1 | Iter 0 | Train Loss 4.14975
[GPU6] Epoch 1 | Iter 0 | Train Loss 4.15208
[GPU7] Epoch 1 | Iter 0 | Train Loss 4.15188
[GPU3] Epoch 1 | Iter 0 | Train Loss 4.14933
[GPU2] Epoch 1 | Iter 0 | Train Loss 4.15260
[GPU0] Epoch 1 | Iter 0 | Train Loss 4.15156
[GPU1] Epoch 1 | Iter 0 | Train Loss 4.14797
[2023-09-11 20:53:06,174][torch.nn.parallel.distributed][INFO] - Reducer buckets have been rebuilt in this iteration.
[2023-09-11 20:53:06,177][torch.nn.parallel.distributed][INFO] - Reducer buckets have been rebuilt in this iteration.
[2023-09-11 20:53:06,177][torch.nn.parallel.distributed][INFO] - Reducer buckets have been rebuilt in this iteration.
[2023-09-11 20:53:06,177][torch.nn.parallel.distributed][INFO] - Reducer buckets have been rebuilt in this iteration.
[2023-09-11 20:53:06,184][torch.nn.parallel.distributed][INFO] - Reducer buckets have been rebuilt in this iteration.
[2023-09-11 20:53:06,184][torch.nn.parallel.distributed][INFO] - Reducer buckets have been rebuilt in this iteration.
[2023-09-11 20:53:06,184][torch.nn.parallel.distributed][INFO] - Reducer buckets have been rebuilt in this iteration.
[2023-09-11 20:53:06,184][torch.nn.parallel.distributed][INFO] - Reducer buckets have been rebuilt in this iteration.
[GPU0] Epoch 1 | Iter 0 | Eval Loss 2.41668
[GPU5] Epoch 1 | Iter 0 | Eval Loss 2.41234
[GPU6] Epoch 1 | Iter 0 | Eval Loss 2.42844
[GPU7] Epoch 1 | Iter 0 | Eval Loss 2.42169
[GPU4] Epoch 1 | Iter 0 | Eval Loss 2.41316
[GPU2] Epoch 1 | Iter 0 | Eval Loss 2.40139
[GPU1] Epoch 1 | Iter 0 | Eval Loss 2.41490
[GPU3] Epoch 1 | Iter 0 | Eval Loss 2.42488
...
[GPU5] Epoch 10 | Iter 0 | Train Loss 1.99076
[GPU4] Epoch 10 | Iter 0 | Train Loss 1.99094
[GPU7] Epoch 10 | Iter 0 | Train Loss 2.00854
[GPU6] Epoch 10 | Iter 0 | Train Loss 2.01149
[GPU1] Epoch 10 | Iter 0 | Train Loss 2.01096
[GPU2] Epoch 10 | Iter 0 | Train Loss 2.01783
[GPU0] Epoch 10 | Iter 0 | Train Loss 1.99053
[GPU3] Epoch 10 | Iter 0 | Train Loss 1.97779
[GPU5] Epoch 10 | Iter 0 | Eval Loss 1.96394
[GPU6] Epoch 10 | Iter 0 | Eval Loss 1.95636
[GPU7] Epoch 10 | Iter 0 | Eval Loss 1.99337
[GPU4] Epoch 10 | Iter 0 | Eval Loss 1.96288
[GPU3] Epoch 10 | Iter 0 | Eval Loss 1.97101
[GPU0] Epoch 10 | Iter 0 | Eval Loss 1.94594
[GPU1] Epoch 10 | Iter 0 | Eval Loss 1.98116
[GPU2] Epoch 10 | Iter 0 | Eval Loss 1.94512

```







