




## 全量微调


生成用于Ascend芯片分布式通信的芯片资源信息配置文件（RANK_TABLE_FILE）。

Ascend HCCL RANK_TABLE_FILE 文件提供Ascend分布式训练作业的集群信息。

```
# 如生成8卡的rank_table_file
> python ./mindformers/tools/hccl_tools.py --device_num "[0,8)"

start ./mindformers/tools/hccl_tools.py
visible_devices:['0', '1', '2', '3', '4', '5', '6', '7']
server_id:192.168.1.196
device_num_list: [0, 1, 2, 3, 4, 5, 6, 7]
rank_id:0, device_id:0, device_ip:192.168.100.101
rank_id:1, device_id:1, device_ip:192.168.101.101
rank_id:2, device_id:2, device_ip:192.168.102.101
rank_id:3, device_id:3, device_ip:192.168.103.101
rank_id:4, device_id:4, device_ip:192.168.100.100
rank_id:5, device_id:5, device_ip:192.168.101.100
rank_id:6, device_id:6, device_ip:192.168.102.100
rank_id:7, device_id:7, device_ip:192.168.103.100
Completed: hccl file was save in : /root/workspace/code/mindformers/hccl_8p_01234567_192.168.1.196.json
```


### 修改配置

```
cd /root/workspace/code/mindformers
vim configs/glm/run_glm_6b_finetune.yaml
```


### 启动训练任务

```
> bash run_distribute.sh /root/workspace/code/mindformers/hccl_8p_01234567_192.168.1.196.json ../configs/glm/run_glm_6b_finetune.yaml '[0,8]' finetune
start training for rank 0, device 0
start training for rank 1, device 1
start training for rank 2, device 2
start training for rank 3, device 3
start training for rank 4, device 4
start training for rank 5, device 5
start training for rank 6, device 6
start training for rank 7, device 7
```


## LoRA微调


### 修改配置
```
cd /root/workspace/code/mindformers
vim configs/glm/run_glm_6b_lora.yaml
```




模型训练启动成功，输出目录的结构如下所示。
```
output/
├── checkpoint
├── log
└── strategy
```
其中，checkpoint文件夹放置权重文件，log文件夹方式日志文件，strategy文件夹放置模型切分策略文件。


查看日志：
```
# cd /root/workspace/code/mindformers/output/
cd log/rank_0
tail -100f info.log 
```

模型输出权重文件：

```
> tree -h checkpoint/
checkpoint/
├── [ 4.0K]  rank_0
│   ├── [ 3.4G]  glm-6b-lora_rank_0-31_4.ckpt
│   └── [ 6.5M]  glm-6b-lora_rank_0-graph.meta
├── [ 4.0K]  rank_1
│   ├── [ 3.4G]  glm-6b-lora_rank_1-31_4.ckpt
│   └── [ 6.5M]  glm-6b-lora_rank_1-graph.meta
├── [ 4.0K]  rank_2
│   ├── [ 3.4G]  glm-6b-lora_rank_2-31_4.ckpt
│   └── [ 6.5M]  glm-6b-lora_rank_2-graph.meta
└── [ 4.0K]  rank_3
    ├── [ 3.4G]  glm-6b-lora_rank_3-31_4.ckpt
    └── [ 6.5M]  glm-6b-lora_rank_3-graph.meta

4 directories, 8 files
```
模型切分策略文件。
```
> tree -h strategy/
strategy/
├── [  22K]  ckpt_strategy_rank_0.ckpt
├── [  22K]  ckpt_strategy_rank_1.ckpt
├── [  22K]  ckpt_strategy_rank_2.ckpt
└── [  22K]  ckpt_strategy_rank_3.ckpt
```



## 权重合并


### 全量微调

```
python3 merge_ckpt.py --src_postfix=31_4 \
> --src_checkpoints_dir=/root/workspace/output/fullft_output \
> --src_strategy_file=/root/workspace/output/fullft_output/strategy/ckpt_strategy_rank_0.ckpt \
> --dst_checkpoints_dir=/root/workspace/output/fullft_merge_checkpoint/

args_opt.src_strategy_file:  /root/workspace/output/fullft_output/strategy/ckpt_strategy_rank_0.ckpt
checkpoint_file_map {7: '/root/workspace/output/fullft_output/checkpoint/rank_7/glm-6b_rank_7-31_4.ckpt', 6: '/root/workspace/output/fullft_output/checkpoint/rank_6/glm-6b_rank_6-31_4.ckpt', 5: '/root/workspace/output/fullft_output/checkpoint/rank_5/glm-6b_rank_5-31_4.ckpt', 4: '/root/workspace/output/fullft_output/checkpoint/rank_4/glm-6b_rank_4-31_4.ckpt', 3: '/root/workspace/output/fullft_output/checkpoint/rank_3/glm-6b_rank_3-31_4.ckpt', 2: '/root/workspace/output/fullft_output/checkpoint/rank_2/glm-6b_rank_2-31_4.ckpt', 1: '/root/workspace/output/fullft_output/checkpoint/rank_1/glm-6b_rank_1-31_4.ckpt', 0: '/root/workspace/output/fullft_output/checkpoint/rank_0/glm-6b_rank_0-31_4.ckpt'}
save_checkpoint_path /root/workspace/output/fullft_merge_checkpoint/transformed.ckpt

[WARNING] ME(8507:281472874964864,MainProcess):2023-07-11-15:32:38.347.469 [mindspore/parallel/_parallel_serialization.py:351] The parameter scale_sense is not in src_strategy.
[WARNING] ME(8507:281472874964864,MainProcess):2023-07-11-15:32:38.347.863 [mindspore/parallel/_parallel_serialization.py:351] The parameter global_step is not in src_strategy.
[WARNING] ME(8507:281472874964864,MainProcess):2023-07-11-15:35:28.541.985 [mindspore/parallel/_parallel_serialization.py:351] The parameter current_iterator_step is not in src_strategy.
[WARNING] ME(8507:281472874964864,MainProcess):2023-07-11-15:35:28.542.313 [mindspore/parallel/_parallel_serialization.py:351] The parameter last_overflow_iterator_step is not in src_strategy.
[WARNING] ME(8507:281472874964864,MainProcess):2023-07-11-15:35:28.542.392 [mindspore/parallel/_parallel_serialization.py:351] The parameter epoch_num is not in src_strategy.
[WARNING] ME(8507:281472874964864,MainProcess):2023-07-11-15:35:28.542.460 [mindspore/parallel/_parallel_serialization.py:351] The parameter step_num is not in src_strategy.
[WARNING] ME(8507:281472874964864,MainProcess):2023-07-11-15:35:28.542.523 [mindspore/parallel/_parallel_serialization.py:351] The parameter loss_scale is not in src_strategy.

transform ckpt done.
Filtering ckpt, this may take a while.

100%|###############################################################################| 1027/1027 [00:35<00:00, 28.57it/s]
```

合并之后的权重文件如下所示：
```
> tree -h /root/workspace/output/fullft_merge_checkpoint
/root/workspace/output/fullft_merge_checkpoint
├── [  13G]  filtered_transformed.ckpt
└── [  63G]  transformed.ckpt
```

