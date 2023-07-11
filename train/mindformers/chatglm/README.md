




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


部分训练日志如下所示：

```
...
[INFO] 2023-07-11 10:35:39,223 [run_mindformer.py:71] main: moe config is: <mindformers.modules.transformer.moe.MoEConfig object at 0xffff10297b10>
[INFO] 2023-07-11 10:35:39,223 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/trainer/base_trainer.py:75] __init__: Now Running Task is: text_generation, Model is: glm_6b
[INFO] 2023-07-11 10:35:39,224 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/trainer/base_trainer.py:177] _check_global_batch_size_for_auto_parallel: The current parallel mode is semi_auto_parallel, full batch is True,so global batch size will be changed: global_batch_size = batch_size * data_parallel * micro_batch_interleave_num = 8 * 1 * 1 = 8
[INFO] 2023-07-11 10:35:39,224 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/trainer/base_trainer.py:514] training_process: .........Build Dataset For Train..........
[INFO] 2023-07-11 10:35:39,224 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/trainer/base_trainer.py:268] create_train_dataset: .........Build Dataset From Config..........
[INFO] 2023-07-11 10:35:39,224 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/dataset/causal_language_model_dataset.py:98] __new__: Now Create Causal Language Model Dataset.
[INFO] 2023-07-11 10:35:39,224 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/dataset/base_dataset.py:50] init_dataset_config: Now the semi auto parallel mode is used and full_batch is True,and the shuffle of the dataset is required to be False,so as to ensure that the data loaded on each card is consistent and to avoid the problem of non-convergence of loss.
[INFO] 2023-07-11 10:35:39,231 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/trainer/utils.py:133] check_runner_config: Will be Training epochs:1, sink_size:4
[INFO] 2023-07-11 10:35:39,231 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/trainer/utils.py:134] check_runner_config: Create training dataset finish, dataset size:125
[INFO] 2023-07-11 10:35:39,231 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/trainer/base_trainer.py:521] training_process: .........Build Net For Train..........
[INFO] 2023-07-11 10:35:39,231 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/trainer/base_trainer.py:282] create_network: .........Build Network From Config..........
[INFO] 2023-07-11 10:38:43,280 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/models/base_model.py:80] load_checkpoint: weights in /root/workspace/model/chatglm-convert/ms_glm_6b.ckpt are loaded
[INFO] 2023-07-11 10:38:43,299 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/trainer/base_trainer.py:425] count_parameters: Network Parameters: 6707 M.
[INFO] 2023-07-11 10:38:43,299 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/trainer/base_trainer.py:544] training_process: .........Build Optimizer For Train..........
[INFO] 2023-07-11 10:38:43,299 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/trainer/base_trainer.py:321] create_optimizer_scheduler: .........Build Optimizer From Config..........
[INFO] 2023-07-11 10:38:43,299 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/trainer/base_trainer.py:354] create_lr_scheduler: .........Build LR Schedule From Config..........
[WARNING] 2023-07-11 10:38:43,306 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/trainer/optimizer_grouped_parameters.py:74] get_optimizer_grouped_parameters: dynamic_lr_schedule will be reset and invalid when layer_scale is False.
...
[INFO] 2023-07-11 10:38:43,568 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/trainer/base_trainer.py:550] training_process: .........Build Running Wrapper From Config For Train..........
[INFO] 2023-07-11 10:38:43,568 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/trainer/base_trainer.py:391] create_model_wrapper: .........Build Model Wrapper for Train From Config..........
[INFO] 2023-07-11 10:38:43,582 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/trainer/base_trainer.py:562] training_process: .........Starting Init Train Model..........
[INFO] 2023-07-11 10:38:43,583 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/trainer/base_trainer.py:581] training_process: .........Build Callbacks For Train..........
[INFO] 2023-07-11 10:38:43,583 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/trainer/base_trainer.py:400] create_callbacks: .........Build Callbacks for Train From Config..........
[INFO] 2023-07-11 10:38:43,584 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/core/callback/callback.py:340] __init__: Integrated_save is changed to False when using auto_parallel.
[INFO] 2023-07-11 10:38:43,585 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/trainer/base_trainer.py:609] training_process: .........Starting Training Model..........
[INFO] 2023-07-11 10:38:43,585 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/trainer/base_trainer.py:610] training_process: .........Model Compiling, Please Wait a Moment...........
[INFO] 2023-07-11 10:47:36,427 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/core/callback/callback.py:269] print_output_info: Epoch:[  1/  1], step:[    4/  125], loss:[2.244/2.244], time:507844.205 ms, lr:[0.], overflow cond: True, loss_scale: 268435460.0
[INFO] 2023-07-11 10:47:37,342 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/core/callback/callback.py:146] epoch_end: Per sink_size step time: 533756.177 ms, per step time: 133439.044 ms, avg loss: 2.244
[INFO] 2023-07-11 10:47:44,861 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/core/callback/callback.py:269] print_output_info: Epoch:[  1/  1], step:[    8/  125], loss:[2.499/2.499], time:7480.938 ms, lr:[0.], overflow cond: True, loss_scale: 16777216.0
[INFO] 2023-07-11 10:47:44,874 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/core/callback/callback.py:146] epoch_end: Per sink_size step time: 7518.224 ms, per step time: 1879.556 ms, avg loss: 2.499
...
[INFO] 2023-07-11 10:48:35,199 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/core/callback/callback.py:146] epoch_end: Per sink_size step time: 1958.791 ms, per step time: 489.698 ms, avg loss: 2.091
[INFO] 2023-07-11 10:48:37,162 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/core/callback/callback.py:269] print_output_info: Epoch:[  1/  1], step:[  116/  125], loss:[2.220/2.220], time:1951.612 ms, lr:[2.4499998e-06], overflow cond: False, loss_scale: 16384.0
[INFO] 2023-07-11 10:48:37,163 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/core/callback/callback.py:146] epoch_end: Per sink_size step time: 1963.915 ms, per step time: 490.979 ms, avg loss: 2.220
[INFO] 2023-07-11 10:48:39,125 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/core/callback/callback.py:269] print_output_info: Epoch:[  1/  1], step:[  120/  125], loss:[2.092/2.092], time:1953.753 ms, lr:[2.5499999e-06], overflow cond: False, loss_scale: 16384.0
[INFO] 2023-07-11 10:48:39,126 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/core/callback/callback.py:146] epoch_end: Per sink_size step time: 1962.049 ms, per step time: 490.512 ms, avg loss: 2.092
[INFO] 2023-07-11 10:48:41,083 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/core/callback/callback.py:269] print_output_info: Epoch:[  1/  1], step:[  124/  125], loss:[2.346/2.346], time:1949.995 ms, lr:[2.65e-06], overflow cond: False, loss_scale: 16384.0
[INFO] 2023-07-11 10:48:41,084 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/core/callback/callback.py:146] epoch_end: Per sink_size step time: 1958.268 ms, per step time: 489.567 ms, avg loss: 2.346
[INFO] 2023-07-11 10:49:26,307 [/root/workspace/code/mindformers/scripts/mf_parallel1/mindformers/trainer/base_trainer.py:616] training_process: .........Training Over!.............
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


### LoRA 微调
```
python3 merge_ckpt.py --src_postfix=31_4 \
--src_checkpoints_dir=/root/workspace/output/lora_output \
--src_strategy_file=/root/workspace/code/mindformers/output/strategy/ckpt_strategy_rank_0.ckpt \
--dst_checkpoints_dir=/root/workspace/output/lora_merge_checkpoint_v2/
```


## 模型评估

### 全量微调模型评估

<details><summary>详细输出</summary><p>

```
python run_mindformer.py \
> --config ./configs/glm/run_glm_6b_infer.yaml \
> --run_mode eval \
> --load_checkpoint /root/workspace/output/fullft_merge_checkpoint/filtered_transformed.ckpt \
> --eval_dataset_dir /root/workspace/data/AdvertiseGen-ms/eval_0711_256.mindrecord \
> --device_id 7
2023-07-11 16:52:59,073 - mindformers - INFO - full_batch will be forced to False when the parallel mode is stand_alone or data_parallel
2023-07-11 16:52:59,075 - mindformers - INFO - .........Build context config..........
2023-07-11 16:52:59,075 - mindformers - INFO - initial moe_config from dict: {'expert_num': 1, 'capacity_factor': 1.05, 'aux_loss_factor': 0.05, 'num_experts_chosen': 1}
2023-07-11 16:52:59,075 - mindformers - INFO - initial recompute_config from dict: {'recompute': False, 'parallel_optimizer_comm_recompute': False, 'mp_comm_recompute': True, 'recompute_slice_activation': False}
2023-07-11 16:52:59,075 - mindformers - INFO - initial parallel_config from dict: {'data_parallel': 1, 'model_parallel': 1, 'pipeline_stage': 1, 'expert_parallel': 1, 'optimizer_shard': False, 'micro_batch_num': 1, 'vocab_emb_dp': True, 'gradient_aggregation_group': 4}
2023-07-11 16:52:59,075 - mindformers - INFO - context config is: [ParallelConfig]
_recompute:[ParallelConfig]
_recompute:False
_parallel_optimizer_comm_recompute:False
_mp_comm_recompute:True
_recompute_slice_activation:False

_optimizer_shard:False
_gradient_aggregation_group:4
_embed_dp_mp_config:[ParallelConfig]
_dp_mp_config:[ParallelConfig]
_data_parallel:1
_model_parallel:1

_vocab_emb_dp:True

_pp_config:[ParallelConfig]
_pipeline_stage:1
_micro_batch_num:1

_moe_config:[ParallelConfig]
_dpmp:[ParallelConfig]
_data_parallel:1
_model_parallel:1

_expert_parallel:1


2023-07-11 16:52:59,076 - mindformers - INFO - moe config is: <mindformers.modules.transformer.moe.MoEConfig object at 0xffff73d74690>
{'auto_trans_ckpt': False,
 'auto_tune': False,
 'autotune_per_step': 10,
 'callbacks': [OrderedDict([('type', 'MFLossMonitor')]),
               OrderedDict([('type', 'SummaryMonitor'),
                            ('keep_default_action', True)]),
               OrderedDict([('type', 'CheckpointMointor'),
                            ('prefix', 'glm-6b'),
                            ('save_checkpoint_steps', 500),
                            ('keep_checkpoint_max', 2),
                            ('integrated_save', False),
                            ('async_save', False)]),
               OrderedDict([('type', 'ObsMonitor'), ('keep_last', False)])],
 'context': {'device_id': 7,
             'device_target': 'Ascend',
             'enable_graph_kernel': False,
             'graph_kernel_flags': '--disable_expand_ops=Softmax,Dropout '
                                   '--enable_parallel_fusion=true '
                                   '--reduce_fuse_depth=8 '
                                   '--enable_auto_tensor_inplace=true',
             'max_call_depth': 10000,
             'save_graphs': False,
             'save_graphs_path': './graph'},
 'device_num': 1,
 'eval_callbacks': [OrderedDict([('type', 'ObsMonitor'),
                                 ('keep_last', False)])],
 'eval_dataset': {'batch_size': 1,
                  'data_loader': {'dataset_dir': '/root/workspace/data/AdvertiseGen-ms/eval_0711_256.mindrecord',
                                  'shuffle': True,
                                  'type': 'MindDataset'},
                  'drop_remainder': True,
                  'input_columns': ['input_ids', 'label'],
                  'num_parallel_workers': 8,
                  'numa_enable': False,
                  'prefetch_size': 1,
                  'python_multiprocessing': False,
                  'repeat': 1,
                  'seed': 0},
 'eval_dataset_task': {'dataset_config': {'batch_size': 1,
                                          'data_loader': {'dataset_dir': '',
                                                          'shuffle': True,
                                                          'type': 'MindDataset'},
                                          'drop_remainder': True,
                                          'input_columns': ['input_ids',
                                                            'label'],
                                          'num_parallel_workers': 8,
                                          'numa_enable': False,
                                          'prefetch_size': 1,
                                          'python_multiprocessing': False,
                                          'repeat': 1,
                                          'seed': 0},
                       'type': 'CausalLanguageModelDataset'},
 'filepath_prefix': './autotune',
 'init_start_profile': True,
 'load_checkpoint': None,
 'local_rank': 0,
 'lr_schedule': {'learning_rate': 5e-05,
                 'lr_end': 1e-06,
                 'total_steps': -1,
                 'type': 'polynomial',
                 'warmup_steps': 2000},
 'metric': {'tokenizer_type': 'glm_6b', 'type': 'ADGENMetric'},
 'micro_batch_interleave_num': 1,
 'model': {'arch': {'type': 'GLMChatModel'},
           'model_config': {'activation_func': 'GELU',
                            'attention_dropout_rate': 0.0,
                            'bos_token_id': 130004,
                            'checkpoint_name_or_path': '/root/workspace/output/fullft_merge_checkpoint/filtered_transformed.ckpt',
                            'compute_dtype': 'float16',
                            'do_sample': True,
                            'embedding_dropout_prob': 0.0,
                            'eos_token_id': 130005,
                            'gmask_token_id': 130001,
                            'hidden_dropout_rate': 0.0,
                            'hidden_size': 4096,
                            'hidden_size_per_attention_head': None,
                            'inner_hidden_size': 16384,
                            'is_enhanced_encoder': True,
                            'is_npu_acceleration': True,
                            'layernorm_compute_type': 'float32',
                            'layernorm_epsilon': 1e-05,
                            'layernorm_order': 'post',
                            'mask_token_id': 130000,
                            'max_decode_length': 2048,
                            'num_heads': 32,
                            'num_layers': 28,
                            'pad_token_id': 3,
                            'param_init_type': 'float16',
                            'position_encoding_2d': True,
                            'repetition_penalty': 1,
                            'seq_length': 512,
                            'softmax_compute_type': 'float32',
                            'top_k': 1,
                            'top_p': 1,
                            'type': 'GLMConfig',
                            'use_final_layernorm': True,
                            'use_past': True,
                            'vocab_size': 130528}},
 'moe_config': <mindformers.modules.transformer.moe.MoEConfig object at 0xffff73d74690>,
 'only_save_strategy': False,
 'optimizer': {'beta1': 0.9,
               'beta2': 0.95,
               'eps': 1e-08,
               'type': 'FusedAdamWeightDecay',
               'weight_decay': 0.1},
 'output_dir': './output',
 'parallel': {'enable_alltoall': False,
              'enable_parallel_optimizer': False,
              'full_batch': True,
              'gradients_mean': False,
              'loss_repeated_mean': True,
              'parallel_mode': 0,
              'search_mode': 'sharding_propagation',
              'strategy_ckpt_save_file': './output/strategy/./ckpt_strategy_rank_0.ckpt'},
 'parallel_config': <mindformers.modules.transformer.transformer.TransformerOpParallelConfig object at 0xffff3085a690>,
 'processor': {'return_tensors': 'ms',
               'tokenizer': {'bos_token': '<sop>',
                             'end_token': '</s>',
                             'eos_token': '<eop>',
                             'gmask_token': '[gMASK]',
                             'mask_token': '[MASK]',
                             'pad_token': '<pad>',
                             'padding_side': 'left',
                             'type': 'ChatGLMTokenizer',
                             'unk_token': '<unk>'},
               'type': 'GLMProcessor'},
 'profile': False,
 'profile_communication': True,
 'profile_memory': True,
 'profile_start_step': 1,
 'profile_stop_step': 10,
 'recompute_config': <mindformers.modules.transformer.transformer.TransformerRecomputeConfig object at 0xffff302b7bd0>,
 'remote_save_url': 'Please input obs url on AICC platform.',
 'resume_training': False,
 'run_mode': 'eval',
 'runner_config': {'batch_size': 1,
                   'epochs': 1,
                   'sink_mode': True,
                   'sink_size': 4},
 'runner_wrapper': {'scale_sense': {'loss_scale_value': 4294967296,
                                    'scale_factor': 2,
                                    'scale_window': 1000,
                                    'type': 'DynamicLossScaleUpdateCell'},
                    'type': 'MFTrainOneStepCell',
                    'use_clip_grad': True},
 'seed': 0,
 'train_dataset': {'batch_size': 1,
                   'data_loader': {'dataset_dir': '',
                                   'shuffle': True,
                                   'type': 'MindDataset'},
                   'drop_remainder': True,
                   'input_columns': ['input_ids',
                                     'label',
                                     'position_ids',
                                     'attention_mask'],
                   'num_parallel_workers': 8,
                   'numa_enable': False,
                   'prefetch_size': 1,
                   'python_multiprocessing': False,
                   'repeat': 1,
                   'seed': 0},
 'train_dataset_task': {'dataset_config': {'batch_size': 1,
                                           'data_loader': {'dataset_dir': '',
                                                           'shuffle': True,
                                                           'type': 'MindDataset'},
                                           'drop_remainder': True,
                                           'input_columns': ['input_ids',
                                                             'label',
                                                             'position_ids',
                                                             'attention_mask'],
                                           'num_parallel_workers': 8,
                                           'numa_enable': False,
                                           'prefetch_size': 1,
                                           'python_multiprocessing': False,
                                           'repeat': 1,
                                           'seed': 0},
                        'type': 'CausalLanguageModelDataset'},
 'trainer': {'model_name': 'glm_6b', 'type': 'CausalLanguageModelingTrainer'},
 'use_parallel': False}
2023-07-11 16:52:59,081 - mindformers - INFO - Now Running Task is: text_generation, Model is: glm_6b
2023-07-11 16:52:59,082 - mindformers - INFO - The current parallel mode is stand_alone, batch size per card will not be changed: batch_size_per_card = 1
2023-07-11 16:52:59,082 - mindformers - INFO - global_batch_size = batch_size_per_card * device_num = 1 * 1 = 1
2023-07-11 16:52:59,082 - mindformers - INFO - parallel_config will be change to default config: [ParallelConfig]
_recompute:[ParallelConfig]
_recompute:False
_parallel_optimizer_comm_recompute:False
_mp_comm_recompute:True
_recompute_slice_activation:False

_optimizer_shard:False
_gradient_aggregation_group:4
_embed_dp_mp_config:[ParallelConfig]
_dp_mp_config:[ParallelConfig]
_data_parallel:1
_model_parallel:1

_vocab_emb_dp:True

_pp_config:[ParallelConfig]
_pipeline_stage:1
_micro_batch_num:1

_moe_config:[ParallelConfig]
_dpmp:[ParallelConfig]
_data_parallel:1
_model_parallel:1

_expert_parallel:1

.
2023-07-11 16:52:59,082 - mindformers - INFO - .........Build Dataset For Evaluate..........
2023-07-11 16:52:59,082 - mindformers - INFO - .........Build Dataset From Config..........
2023-07-11 16:52:59,083 - mindformers - INFO - Now Create Causal Language Model Dataset.
[WARNING] ME(69632:281472846833536,MainProcess):2023-07-11-16:52:59.839.42 [mindspore/dataset/core/validator_helpers.py:806] 'TypeCast' from mindspore.dataset.transforms.c_transforms is deprecated from version 1.8 and will be removed in a future version. Use 'TypeCast' from mindspore.dataset.transforms instead.
2023-07-11 16:52:59,084 - mindformers - INFO - .........Build Network From Config..........
[WARNING] ME(69632:281472846833536,MainProcess):2023-07-11-16:53:02.472.83 [mindspore/common/_decorator.py:40] 'TensorAdd' is deprecated from version 1.1 and will be removed in a future version, use 'Add' instead.
[WARNING] ME(69632:281472846833536,MainProcess):2023-07-11-16:56:17.765.324 [mindspore/train/serialization.py:1058] For 'load_param_into_net', 56 parameters in the 'net' are not loaded, because they are not in the 'parameter_dict', please check whether the network structure is consistent when training and loading checkpoint.
[WARNING] ME(69632:281472846833536,MainProcess):2023-07-11-16:56:17.765.833 [mindspore/train/serialization.py:1060] transformer.layers.0.key_past is not loaded.
...
[WARNING] ME(69632:281472846833536,MainProcess):2023-07-11-16:56:17.768.487 [mindspore/train/serialization.py:1060] transformer.layers.27.value_past is not loaded.
2023-07-11 16:56:17,768 - mindformers - INFO - weights in /root/workspace/output/fullft_merge_checkpoint/filtered_transformed.ckpt are loaded
2023-07-11 16:56:17,878 - mindformers - INFO - Network Parameters: 6825 M.
2023-07-11 16:56:17,878 - mindformers - INFO - .........Build Compute Metrics For Evaluate..........
2023-07-11 16:56:17,878 - mindformers - INFO - Config in the yaml file ./checkpoint_download/glm/glm_6b.yaml are used for tokenizer building.
2023-07-11 16:56:18,492 - mindformers - INFO - Load the tokenizer name ChatGLMTokenizer from the ./checkpoint_download/glm/glm_6b.yaml
2023-07-11 16:56:18,528 - mindformers - INFO - config in the yaml file ./checkpoint_download/glm/glm_6b.yaml are used for tokenizer building.
2023-07-11 16:56:18,564 - mindformers - WARNING - Can't find the tokenizer_config.json in the file_dict. The content of file_dict is : {}
2023-07-11 16:56:18,565 - mindformers - INFO - build tokenizer class name is: ChatGLMTokenizer using args {'bos_token': '<sop>', 'eos_token': '<eop>', 'end_token': '</s>', 'mask_token': '[MASK]', 'gmask_token': '[gMASK]', 'padding_side': 'left', 'pad_token': '<pad>', 'unk_token': '<unk>', 'vocab_file': './checkpoint_download/glm/ice_text.model'}.
2023-07-11 16:56:18,969 - mindformers - INFO - ChatGLMTokenizer Tokenizer built successfully!
2023-07-11 16:56:18,969 - mindformers - INFO - .........Starting Init Evaluate Model..........
2023-07-11 16:56:18,970 - mindformers - INFO - .........Starting Evaluate Model..........

2023-07-11 17:00:33,480 - mindformers - INFO - Epoch 1 Finished, cost time 254.49472093582153,  every example cost time is 254.49472093582153, generate speed: 0.13359805608138378 tokens/s, avg speed: 0.0 tokens/s
pred is:
 以白色为底,以清新、淡雅的刺绣花朵为装饰,将v领与抽褶的元素融入其中,将简约与浪漫完美演绎。
 label is:
 简单大气纯白色连衣裙,是开春季节最美好的穿搭单品。简单的小v领点缀领部,加以独特的花边绣花点缀,满满的清新活力悠然散发。加以纯粹的白色选料,上身亲肤透气,自带自然的褶皱肌理。同时,中长款式,修饰好身材,十分美腻。
Building prefix dict from the default dictionary ...
Loading model from cache /tmp/jieba.cache
Loading model cost 1.166 seconds.
Prefix dict has been built successfully.
2023-07-11 17:00:40,003 - mindformers - INFO - Epoch 2 Finished, cost time 5.341211557388306,  every example cost time is 5.341211557388306, generate speed: 24.151823722757985 tokens/s, avg speed: 24.151371552472405 tokens/s
pred is:
 一款吊带裙,以清新的白色为主色调,简约而优雅。采用薄而透气的雪纺面料,在细节处加入精致的花边,让整件裙子在细节处尽显女性的柔美。 v领设计,露出性感的锁骨,性感又迷人。 腰部的拼接设计,将整件裙子的层次感演绎的恰到好处,既不失简约的大气,又凸显了女性的柔美。 下摆的宽松设计,在行走中自然流动,将整件裙子的优雅气质展现的淋淋尽致。 整体设计以休闲为主,在细节处加入精致的蝴蝶结,让整件裙子更显甜美。
 label is:
 优美而动感的上衣。采用半透的雪纺材质工艺,深黑色系给您以非常魅惑的穿着体验,内里需要搭配深黑色的吊带。花边v字领口连襟拼接,举手投足更加优雅迷人,适合搭配各种半身裙和休闲长裤。
2023-07-11 17:00:41,904 - mindformers - INFO - Epoch 3 Finished, cost time 1.8842175006866455,  every example cost time is 1.8842175006866455, generate speed: 23.88259316326333 tokens/s, avg speed: 24.08128160602231 tokens/s
pred is:
 以白色为底,以黑色线条为装饰,将高腰与a字裙的设计元素融合在一起,打造出一种独特的时尚风格。将简约与个性元素相结合,让穿着者更加有型有款。
 label is:
 这款裙子采用黑色的颜色打底,裙身上装饰着白色的线条以及扣子装饰,丰富视觉上的变化。另外整体上a字裙裙型搭配高腰的设计,修身效果出众,还有着不规则的裙摆,展现出十足的设计感。
2023-07-11 17:00:43,475 - mindformers - INFO - Epoch 4 Finished, cost time 1.561234951019287,  every example cost time is 1.561234951019287, generate speed: 23.699187605199157 tokens/s, avg speed: 24.013391025594462 tokens/s
pred is:
 以蕾丝为底,以宫廷刺绣为装饰,以大裙摆和泡泡袖为特点,将时尚与复古元素相结合,打造一款华丽、浪漫的蕾丝裙。
 label is:
 宫廷风的甜美蕾丝设计,清醒的蕾丝拼缝处,刺绣定制的贝壳花边,增添了裙子的精致感觉。超大的裙摆,加上精细的小花边设计,上身后既带着仙气撩人又很有女人味。泡泡袖上的提花面料,在细节处增加了浪漫感,春日的仙女姐姐。浪漫蕾丝布满整个裙身,美丽明艳,气质超仙。
...
2023-07-11 17:02:40,918 - mindformers - INFO - Epoch 45 Finished, cost time 5.272047996520996,  every example cost time is 5.272047996520996, generate speed: 25.98610636519352 tokens/s, avg speed: 24.71114413227348 tokens/s
pred is:
 黑色条纹裤是经典的时尚单品,不仅简约大方,还非常百搭,可以搭配各种上衣,让你时尚又气质。

搭配白色T恤,简约清新,白色与黑色条纹的搭配非常显瘦,还很有层次感。

搭配印花T恤,印花元素非常可爱,搭配黑色条纹裤,很有小清新的感觉。

搭配牛仔衬衫,牛仔衬衫的休闲感与黑色条纹裤的时尚感相结合,非常时髦。

搭配皮质外套,皮质外套的帅气与黑色条纹裤的休闲感相结合,很有时尚感。

黑色条纹裤可以搭配各种上衣,非常百搭,而且简约大方,让你时尚又气质。
 label is:
 传承动感简约气质的条纹衣身,结合包边圆领和半开襟设计,造型显得活力有范,又不失男孩子的时尚帅气。胸前单侧小口袋点缀,让男宝宝帅气加倍。搭配纯黑色的底裤,整体显得层次十足,视觉也十分有美感,男宝宝穿起来独特魅力尽显。
2023-07-11 17:02:42,877 - mindformers - INFO - Epoch 46 Finished, cost time 1.9328925609588623,  every example cost time is 1.9328925609588623, generate speed: 25.350607162403406 tokens/s, avg speed: 24.720831918537765 tokens/s
pred is:
 以简约为灵魂,以纯色为基石,以条纹为元素,将时尚与舒适结合,将简约与个性展现。这款外套,时尚与实用并存,在细节处展现品质,在简约中彰显个性。
 label is:
 来自巴拉巴拉的女童长款外套,设计师采用直筒式衣袖裁剪,并在袖口加饰有纯色条纹,在打破了整体的单一性的同时,还增添了一丝简约时尚气息。再加上对称的斜插口袋,既能给予娇嫩双手温暖,同时还可放置孩子的随身物品,暖心又很实用呢。
2023-07-11 17:02:45,750 - mindformers - INFO - Epoch 47 Finished, cost time 2.860628366470337,  every example cost time is 2.860628366470337, generate speed: 25.16929526530534 tokens/s, avg speed: 24.730666589943375 tokens/s
pred is:
 以蝴蝶结为元素,将层叠的网纱与系带进行搭配,将整体设计打造为半裙的半身裙,在腰部加入门襟设计,将整体设计打造为系带的蝴蝶结,在腰部的层叠层叠的网纱与蝴蝶结的点缀下,让整件裙子在细节处尽显尽显的时尚感。
 label is:
 层叠网纱,仙气飘飘,却不会过于膨胀。腰间的蝴蝶结系带,恰到好处的增添了柔美感。膝盖以下,长度刚刚好的半身裙,比起“一览无遗魅力尽显”,专注于“完美隐藏”
2023-07-11 17:02:47,855 - mindformers - INFO - Epoch 48 Finished, cost time 2.094264030456543,  every example cost time is 2.094264030456543, generate speed: 24.829725022142583 tokens/s, avg speed: 24.732231816627035 tokens/s
pred is:
 焦糖色连衣裙,简约而不失优雅,宽松的版型,穿上身更显气质。百褶的裙摆,在腰部的收腰设计,更显身材的纤细。搭配宽松的腰带,将身材比例修饰的更好。整体的设计,更显女性的柔美。
 label is:
 来自<UNK>自制的连衣裙采用今年大热的焦糖色,就像巧克力一样,甜蜜又不腻人。腰带的贴心设计,让宽松的版型也能拥有s曲线。上身简约的衬衫式翻领,衬托小v脸,带来一股职场ol风,加以百褶下摆的点缀,一起述说无尽温柔。
2023-07-11 17:02:50,774 - mindformers - INFO - Epoch 49 Finished, cost time 2.9023842811584473,  every example cost time is 2.9023842811584473, generate speed: 25.15173489392764 tokens/s, avg speed: 24.74122134217671 tokens/s
pred is:
 一款时尚舒适的羊毛九分微喇裤,采用优质羊毛面料打造,柔软亲肤,保暖舒适。微喇裤的裤口设计,修饰腿型,展现优美曲线。九分的长度,修饰脚踝,显瘦显高挑。裤身的流线型设计,修饰腰部,展现身材比例。搭配一件简约的毛衣,即可穿出气质。
 label is:
 不同于一般的西服裤。这款<UNK>小喇叭羊毛裤在样式上显得更加时髦优雅,特地采用微微的九分喇叭裤腿设计,视觉上将脚踝处显得更加纤细。并且特地甄选柔软的羊毛材质,就算直接贴肤穿着,也不会觉得寒冷,比较适合初秋穿噢。
2023-07-11 17:02:53,838 - mindformers - INFO - Epoch 50 Finished, cost time 3.0505833625793457,  every example cost time is 3.0505833625793457, generate speed: 25.241073869521973 tokens/s, avg speed: 24.75223162149713 tokens/s
pred is:
 以绿色为主色调,将复古与时尚相结合,设计宽松的版型,在细节处加入复古图案,让整件裙子更加有层次感。在腰部加入的褶皱设计,让腰部更加修饰,同时让整件裙子更加有型。在裙摆处加入的灯笼袖设计,让整件裙子更加有层次感,同时让裙摆更加修饰身材。
 label is:
 袖子有灯笼袖的既视感,中世纪的复古韵味轻松展现,版型宽松舒适,上身贴合身材,不会显胖。超级百搭,秋季单穿,搭配裙子裤子都ok!冬天也能做打底,外搭毛呢大衣,气质满满。
2023-07-11 17:02:53,853 - mindformers - INFO - metric: Text Generation Metric 
rouge-1: 25.986109999999993 
rouge-2: 4.22121 
rouge-l: 21.202881999999995 
bleu-4:  4.540767999999999 
2023-07-11 17:02:53,854 - mindformers - INFO - ...........Evaluate Over!...............
```

</p></details>



### LoRA 微调模型评估

```
python run_mindformer.py \
--config ./configs/glm/run_glm_6b_lora_infer.yaml \
--run_mode eval \
--load_checkpoint /root/workspace/output/lora_merge_checkpoint_v2/filtered_transformed.ckpt \
--eval_dataset_dir /root/workspace/data/AdvertiseGen-ms/eval_0711_256.mindrecord \
--device_id 0
```




