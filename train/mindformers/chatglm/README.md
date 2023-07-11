




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
