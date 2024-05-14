


- https://pytorch.org/tutorials/intermediate/ddp_series_minGPT.html




---


## 单机多卡

```
torchrun --standalone --nproc_per_node=4 main.py
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


```

```






