

## 环境


```
cd /home/guodong.li/virtual-venv
virtualenv -p /usr/bin/python3.10 peft-venv-py310-cu117
source /home/guodong.li/virtual-venv/peft-venv-py310-cu117/bin/activate


pip install torch-1.13.1+cu117-cp310-cp310-linux_x86_64.whl
pip install torchvision-0.14.1+cu117-cp310-cp310-linux_x86_64.whl

git clone https://github.com/huggingface/peft
cd peft
git checkout 42ab106
pip install -e .

pip install datasets

pip install jupyterlab

pip install deepspeed
```

生成配置文件：
```
> jupyter lab --generate-config
Writing default config to: /home/guodong.li/.jupyter/jupyter_lab_config.py
```

对密码进行加密：
```
from jupyter_server.auth import passwd; passwd()
```


修改配置文件：
```
c.ServerApp.allow_origin = '*'
c.ServerApp.allow_remote_access = True
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.open_browser = False  
c.ServerApp.password = '加密后的密码'
c.ServerApp.port = 9999
```

启动：
```
jupyter lab --allow-root
nohup jupyter lab --allow-root > jupyterlab.log 2>&1 &
```



# 任务类型

- 因果语言模型（Causal Language Modeling）
- 条件生成（Conditional Generation）
- 序列分类（Sequence Classification）：整个序列输出一个标签。
- Token 分类（Token Classification）：每个 Token 输出一个标签。
- 文本-图像生成（Text-to-Image Generation）
- 图像分类（Image Classification）
- 序列到序列语言模型（Seq2Seq LM）
- 问答任务（Question Answering）：返回给定问题的答案。常见的问答任务有两种类型：
  - 提取：从给定的上下文中提取答案。
  - 摘要：从上下文中生成正确回答问题的答案。


以下是一些常见 NLP 任务：

-   序列分类（Sequence Classification），对整个句子进行分类。如: 获取评论的情绪，检测电子邮件是否为垃圾邮件，确定句子在语法上是否正确或两个句子在逻辑上是否相关等
-   Token分类（Token Classification），对句子中的每个词进行分类。如: 识别句子的语法成分（名词、动词、形容词）或命名实体（人、地点、组织）。
-   **生成文本内容**: 用自动生成的文本完成提示，用屏蔽词填充文本中的空白
-   **从文本中提取答案**: 给定问题和上下文，根据上下文中提供的信息提取问题的答案
-   **从输入文本生成新句子**: 将文本翻译成另一种语言，总结文本




## 高效微调

### 大模型参数高效微调技术实战（二）-Prompt Tuning

### 大模型参数高效微调技术实战（三）-P-Tuning


LSTM:
```
PeftModelForCausalLM(
  (base_model): BloomForCausalLM(
    (transformer): BloomModel(
      (word_embeddings): Embedding(250880, 1024)
      (word_embeddings_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
      (h): ModuleList(
        (0): BloomBlock(
          (input_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          (self_attention): BloomAttention(
            (query_key_value): Linear(in_features=1024, out_features=3072, bias=True)
            (dense): Linear(in_features=1024, out_features=1024, bias=True)
            (attention_dropout): Dropout(p=0.0, inplace=False)
          )
          (post_attention_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          (mlp): BloomMLP(
            (dense_h_to_4h): Linear(in_features=1024, out_features=4096, bias=True)
            (gelu_impl): BloomGelu()
            (dense_4h_to_h): Linear(in_features=4096, out_features=1024, bias=True)
          )
        )
        ...
        (23): BloomBlock(
          ...
        )
      )
      (ln_f): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
    )
    (lm_head): Linear(in_features=1024, out_features=250880, bias=False)
  )
  (prompt_encoder): ModuleDict(
    (default): PromptEncoder(
      (embedding): Embedding(20, 1024)
      (lstm_head): LSTM(1024, 128, num_layers=2, batch_first=True, bidirectional=True)
      (mlp_head): Sequential(
        (0): Linear(in_features=256, out_features=256, bias=True)
        (1): ReLU()
        (2): Linear(in_features=256, out_features=1024, bias=True)
      )
    )
  )
  (word_embeddings): Embedding(250880, 1024)
)
```


### 大模型参数高效微调技术实战（四）-Prefix Tuning 



### 大模型参数高效微调技术实战（三）-LoRA


<details><summary>详细输出：</summary><p>
  
```
> accelerate launch --config_file accelerate_ds_zero3_cpu_offload_config.yaml peft_lora_clm_accelerate_ds_zero3_offload.py
[2023-07-18 18:07:40,939] [INFO] [real_accelerator.py:133:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2023-07-18 18:07:43,645] [INFO] [real_accelerator.py:133:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2023-07-18 18:07:45,313] [WARNING] [comm.py:152:init_deepspeed_backend] NCCL backend in DeepSpeed not yet implemented
[2023-07-18 18:07:45,313] [INFO] [comm.py:616:init_distributed] cdb=None
[2023-07-18 18:07:45,313] [INFO] [comm.py:643:init_distributed] Initializing TorchBackend in DeepSpeed with backend nccl
Found cached dataset raft (/home/guodong.li/data/peft/data/raft/twitter_complaints/1.1.0/79c4de1312c1e3730043f7db07179c914f48403101f7124e2fe336f6f54d9f84)
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 731.03it/s]
Loading cached processed dataset at /home/guodong.li/data/peft/data/raft/twitter_complaints/1.1.0/79c4de1312c1e3730043f7db07179c914f48403101f7124e2fe336f6f54d9f84/cache-fa180e92c989bf46.arrow
Loading cached processed dataset at /home/guodong.li/data/peft/data/raft/twitter_complaints/1.1.0/79c4de1312c1e3730043f7db07179c914f48403101f7124e2fe336f6f54d9f84/cache-923041e0448cc3ac.arrow
Loading cached processed dataset at /home/guodong.li/data/peft/data/raft/twitter_complaints/1.1.0/79c4de1312c1e3730043f7db07179c914f48403101f7124e2fe336f6f54d9f84/cache-85bb53e2cdd61aab.arrow
Loading cached processed dataset at /home/guodong.li/data/peft/data/raft/twitter_complaints/1.1.0/79c4de1312c1e3730043f7db07179c914f48403101f7124e2fe336f6f54d9f84/cache-954cbf329129cf3b.arrow
[2023-07-18 18:07:48,082] [INFO] [partition_parameters.py:326:__exit__] finished initializing model with 0.82B parameters
trainable params: 786,432 || all params: 560,001,024 || trainable%: 0.14043402892063284
[2023-07-18 18:07:49,237] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed info: version=0.10.0, git-hash=unknown, git-branch=unknown
[2023-07-18 18:07:49,293] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed Flops Profiler Enabled: False
[2023-07-18 18:07:49,294] [INFO] [logging.py:96:log_dist] [Rank 0] Removing param_group that has no 'params' in the client Optimizer
[2023-07-18 18:07:49,294] [INFO] [logging.py:96:log_dist] [Rank 0] Using client Optimizer as basic optimizer
[2023-07-18 18:07:49,311] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed Basic Optimizer = AdamW
[2023-07-18 18:07:49,311] [INFO] [utils.py:54:is_zero_supported_optimizer] Checking ZeRO support for optimizer=AdamW type=<class 'torch.optim.adamw.AdamW'>
[2023-07-18 18:07:49,311] [INFO] [logging.py:96:log_dist] [Rank 0] Creating fp16 ZeRO stage 3 optimizer, MiCS is enabled False, Hierarchical params gather False
[2023-07-18 18:07:49,311] [INFO] [logging.py:96:log_dist] [Rank 0] Creating torch.float32 ZeRO stage 3 optimizer
[2023-07-18 18:07:49,375] [INFO] [utils.py:785:see_memory_usage] Stage 3 initialize beginning
[2023-07-18 18:07:49,376] [INFO] [utils.py:786:see_memory_usage] MA 3.04 GB         Max_MA 4.95 GB         CA 5.95 GB         Max_CA 6 GB
[2023-07-18 18:07:49,376] [INFO] [utils.py:793:see_memory_usage] CPU Virtual Memory:  used = 39.9 GB, percent = 4.0%
[2023-07-18 18:07:49,379] [INFO] [stage3.py:117:__init__] Reduce bucket size 500,000,000
[2023-07-18 18:07:49,379] [INFO] [stage3.py:118:__init__] Prefetch bucket size 50,000,000
[2023-07-18 18:07:49,437] [INFO] [utils.py:785:see_memory_usage] DeepSpeedZeRoOffload initialize [begin]
...
[2023-07-18 18:07:50,074] [INFO] [utils.py:786:see_memory_usage] MA 4.92 GB         Max_MA 4.92 GB         CA 4.94 GB         Max_CA 5 GB
[2023-07-18 18:07:50,074] [INFO] [utils.py:793:see_memory_usage] CPU Virtual Memory:  used = 39.9 GB, percent = 4.0%
[2023-07-18 18:07:50,074] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed Final Optimizer = AdamW
[2023-07-18 18:07:50,074] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed using client LR scheduler
[2023-07-18 18:07:50,074] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed LR Scheduler = None
[2023-07-18 18:07:50,074] [INFO] [logging.py:96:log_dist] [Rank 0] step=0, skipped=0, lr=[0.003], mom=[(0.9, 0.999)]
[2023-07-18 18:07:50,075] [INFO] [config.py:960:print] DeepSpeedEngine configuration:
[2023-07-18 18:07:50,075] [INFO] [config.py:964:print]   activation_checkpointing_config  {
    "partition_activations": false,
    "contiguous_memory_optimization": false,
    "cpu_checkpointing": false,
    "number_checkpoints": null,
    "synchronize_checkpoint_boundary": false,
    "profile": false
}
[2023-07-18 18:07:50,076] [INFO] [config.py:964:print]   aio_config ................... {'block_size': 1048576, 'queue_depth': 8, 'thread_count': 1, 'single_submit': False, 'overlap_events': True}
[2023-07-18 18:07:50,076] [INFO] [config.py:964:print]   amp_enabled .................. False
[2023-07-18 18:07:50,076] [INFO] [config.py:964:print]   amp_params ................... False
[2023-07-18 18:07:50,076] [INFO] [config.py:964:print]   autotuning_config ............ {
    "enabled": false,
    "start_step": null,
    "end_step": null,
    "metric_path": null,
    "arg_mappings": null,
    "metric": "throughput",
    "model_info": null,
    "results_dir": "autotuning_results",
    "exps_dir": "autotuning_exps",
    "overwrite": true,
    "fast": true,
    "start_profile_step": 3,
    "end_profile_step": 5,
    "tuner_type": "gridsearch",
    "tuner_early_stopping": 5,
    "tuner_num_trials": 50,
    "model_info_path": null,
    "mp_size": 1,
    "max_train_batch_size": null,
    "min_train_batch_size": 1,
    "max_train_micro_batch_size_per_gpu": 1.024000e+03,
    "min_train_micro_batch_size_per_gpu": 1,
    "num_tuning_micro_batch_sizes": 3
}
[2023-07-18 18:07:50,076] [INFO] [config.py:964:print]   bfloat16_enabled ............. False
[2023-07-18 18:07:50,076] [INFO] [config.py:964:print]   checkpoint_parallel_write_pipeline  False
[2023-07-18 18:07:50,076] [INFO] [config.py:964:print]   checkpoint_tag_validation_enabled  True
[2023-07-18 18:07:50,076] [INFO] [config.py:964:print]   checkpoint_tag_validation_fail  False
[2023-07-18 18:07:50,076] [INFO] [config.py:964:print]   comms_config ................. <deepspeed.comm.config.DeepSpeedCommsConfig object at 0x7f36e0102350>
[2023-07-18 18:07:50,076] [INFO] [config.py:964:print]   communication_data_type ...... None
[2023-07-18 18:07:50,076] [INFO] [config.py:964:print]   compression_config ........... {'weight_quantization': {'shared_parameters': {'enabled': False, 'quantizer_kernel': False, 'schedule_offset': 0, 'quantize_groups': 1, 'quantize_verbose': False, 'quantization_type': 'symmetric', 'quantize_weight_in_forward': False, 'rounding': 'nearest', 'fp16_mixed_quantize': False, 'quantize_change_ratio': 0.001}, 'different_groups': {}}, 'activation_quantization': {'shared_parameters': {'enabled': False, 'quantization_type': 'symmetric', 'range_calibration': 'dynamic', 'schedule_offset': 1000}, 'different_groups': {}}, 'sparse_pruning': {'shared_parameters': {'enabled': False, 'method': 'l1', 'schedule_offset': 1000}, 'different_groups': {}}, 'row_pruning': {'shared_parameters': {'enabled': False, 'method': 'l1', 'schedule_offset': 1000}, 'different_groups': {}}, 'head_pruning': {'shared_parameters': {'enabled': False, 'method': 'topk', 'schedule_offset': 1000}, 'different_groups': {}}, 'channel_pruning': {'shared_parameters': {'enabled': False, 'method': 'l1', 'schedule_offset': 1000}, 'different_groups': {}}, 'layer_reduction': {'enabled': False}}
[2023-07-18 18:07:50,076] [INFO] [config.py:964:print]   curriculum_enabled_legacy .... False
[2023-07-18 18:07:50,076] [INFO] [config.py:964:print]   curriculum_params_legacy ..... False
[2023-07-18 18:07:50,076] [INFO] [config.py:964:print]   data_efficiency_config ....... {'enabled': False, 'seed': 1234, 'data_sampling': {'enabled': False, 'num_epochs': 1000, 'num_workers': 0, 'curriculum_learning': {'enabled': False}}, 'data_routing': {'enabled': False, 'random_ltd': {'enabled': False, 'layer_token_lr_schedule': {'enabled': False}}}}
[2023-07-18 18:07:50,076] [INFO] [config.py:964:print]   data_efficiency_enabled ...... False
[2023-07-18 18:07:50,076] [INFO] [config.py:964:print]   dataloader_drop_last ......... False
[2023-07-18 18:07:50,076] [INFO] [config.py:964:print]   disable_allgather ............ False
[2023-07-18 18:07:50,076] [INFO] [config.py:964:print]   dump_state ................... False
[2023-07-18 18:07:50,077] [INFO] [config.py:964:print]   dynamic_loss_scale_args ...... None
[2023-07-18 18:07:50,077] [INFO] [config.py:964:print]   eigenvalue_enabled ........... False
[2023-07-18 18:07:50,077] [INFO] [config.py:964:print]   eigenvalue_gas_boundary_resolution  1
[2023-07-18 18:07:50,077] [INFO] [config.py:964:print]   eigenvalue_layer_name ........ bert.encoder.layer
[2023-07-18 18:07:50,077] [INFO] [config.py:964:print]   eigenvalue_layer_num ......... 0
[2023-07-18 18:07:50,077] [INFO] [config.py:964:print]   eigenvalue_max_iter .......... 100
[2023-07-18 18:07:50,077] [INFO] [config.py:964:print]   eigenvalue_stability ......... 1e-06
[2023-07-18 18:07:50,077] [INFO] [config.py:964:print]   eigenvalue_tol ............... 0.01
[2023-07-18 18:07:50,077] [INFO] [config.py:964:print]   eigenvalue_verbose ........... False
[2023-07-18 18:07:50,077] [INFO] [config.py:964:print]   elasticity_enabled ........... False
[2023-07-18 18:07:50,077] [INFO] [config.py:964:print]   flops_profiler_config ........ {
    "enabled": false,
    "recompute_fwd_factor": 0.0,
    "profile_step": 1,
    "module_depth": -1,
    "top_modules": 1,
    "detailed": true,
    "output_file": null
}
[2023-07-18 18:07:50,077] [INFO] [config.py:964:print]   fp16_auto_cast ............... None
[2023-07-18 18:07:50,077] [INFO] [config.py:964:print]   fp16_enabled ................. False
[2023-07-18 18:07:50,077] [INFO] [config.py:964:print]   fp16_master_weights_and_gradients  False
[2023-07-18 18:07:50,077] [INFO] [config.py:964:print]   global_rank .................. 0
[2023-07-18 18:07:50,077] [INFO] [config.py:964:print]   grad_accum_dtype ............. None
[2023-07-18 18:07:50,077] [INFO] [config.py:964:print]   gradient_accumulation_steps .. 1
[2023-07-18 18:07:50,077] [INFO] [config.py:964:print]   gradient_clipping ............ 1.0
[2023-07-18 18:07:50,077] [INFO] [config.py:964:print]   gradient_predivide_factor .... 1.0
[2023-07-18 18:07:50,077] [INFO] [config.py:964:print]   hybrid_engine ................ enabled=False max_out_tokens=512 inference_tp_size=1 release_inference_cache=False pin_parameters=True tp_gather_partition_size=8
[2023-07-18 18:07:50,077] [INFO] [config.py:964:print]   initial_dynamic_scale ........ 65536
[2023-07-18 18:07:50,077] [INFO] [config.py:964:print]   load_universal_checkpoint .... False
[2023-07-18 18:07:50,077] [INFO] [config.py:964:print]   loss_scale ................... 0
[2023-07-18 18:07:50,077] [INFO] [config.py:964:print]   memory_breakdown ............. False
[2023-07-18 18:07:50,077] [INFO] [config.py:964:print]   mics_hierarchial_params_gather  False
[2023-07-18 18:07:50,077] [INFO] [config.py:964:print]   mics_shard_size .............. -1
[2023-07-18 18:07:50,077] [INFO] [config.py:964:print]   monitor_config ............... tensorboard=TensorBoardConfig(enabled=False, output_path='', job_name='DeepSpeedJobName') wandb=WandbConfig(enabled=False, group=None, team=None, project='deepspeed') csv_monitor=CSVConfig(enabled=False, output_path='', job_name='DeepSpeedJobName') enabled=False
[2023-07-18 18:07:50,077] [INFO] [config.py:964:print]   nebula_config ................ {
    "enabled": false,
    "persistent_storage_path": null,
    "persistent_time_interval": 100,
    "num_of_version_in_retention": 2,
    "enable_nebula_load": true,
    "load_path": null
}
[2023-07-18 18:07:50,077] [INFO] [config.py:964:print]   optimizer_legacy_fusion ...... False
[2023-07-18 18:07:50,077] [INFO] [config.py:964:print]   optimizer_name ............... None
[2023-07-18 18:07:50,077] [INFO] [config.py:964:print]   optimizer_params ............. None
[2023-07-18 18:07:50,077] [INFO] [config.py:964:print]   pipeline ..................... {'stages': 'auto', 'partition': 'best', 'seed_layers': False, 'activation_checkpoint_interval': 0}
[2023-07-18 18:07:50,077] [INFO] [config.py:964:print]   pld_enabled .................. False
[2023-07-18 18:07:50,077] [INFO] [config.py:964:print]   pld_params ................... False
[2023-07-18 18:07:50,077] [INFO] [config.py:964:print]   prescale_gradients ........... False
[2023-07-18 18:07:50,077] [INFO] [config.py:964:print]   scheduler_name ............... None
[2023-07-18 18:07:50,077] [INFO] [config.py:964:print]   scheduler_params ............. None
[2023-07-18 18:07:50,078] [INFO] [config.py:964:print]   sparse_attention ............. None
[2023-07-18 18:07:50,078] [INFO] [config.py:964:print]   sparse_gradients_enabled ..... False
[2023-07-18 18:07:50,078] [INFO] [config.py:964:print]   steps_per_print .............. inf
[2023-07-18 18:07:50,078] [INFO] [config.py:964:print]   train_batch_size ............. 8
[2023-07-18 18:07:50,078] [INFO] [config.py:964:print]   train_micro_batch_size_per_gpu  8
[2023-07-18 18:07:50,078] [INFO] [config.py:964:print]   use_node_local_storage ....... False
[2023-07-18 18:07:50,078] [INFO] [config.py:964:print]   wall_clock_breakdown ......... False
[2023-07-18 18:07:50,078] [INFO] [config.py:964:print]   world_size ................... 1
[2023-07-18 18:07:50,078] [INFO] [config.py:964:print]   zero_allow_untested_optimizer  True
[2023-07-18 18:07:50,078] [INFO] [config.py:964:print]   zero_config .................. stage=3 contiguous_gradients=True reduce_scatter=True reduce_bucket_size=500,000,000 allgather_partitions=True allgather_bucket_size=500,000,000 overlap_comm=True load_from_fp32_weights=True elastic_checkpoint=False offload_param=DeepSpeedZeroOffloadParamConfig(device='none', nvme_path=None, buffer_count=5, buffer_size=100,000,000, max_in_cpu=1,000,000,000, pin_memory=False) offload_optimizer=DeepSpeedZeroOffloadOptimizerConfig(device='none', nvme_path=None, buffer_count=4, pin_memory=False, pipeline=False, pipeline_read=False, pipeline_write=False, fast_init=False) sub_group_size=1,000,000,000 cpu_offload_param=None cpu_offload_use_pin_memory=None cpu_offload=None prefetch_bucket_size=50,000,000 param_persistence_threshold=100,000 model_persistence_threshold=sys.maxsize max_live_parameters=1,000,000,000 max_reuse_distance=1,000,000,000 gather_16bit_weights_on_model_save=True stage3_gather_fp16_weights_on_model_save=False ignore_unused_parameters=True legacy_stage1=False round_robin_gradients=False zero_hpz_partition_size=1 zero_quantized_weights=False zero_quantized_gradients=False mics_shard_size=-1 mics_hierarchical_params_gather=False memory_efficient_linear=True
[2023-07-18 18:07:50,078] [INFO] [config.py:964:print]   zero_enabled ................. True
[2023-07-18 18:07:50,078] [INFO] [config.py:964:print]   zero_force_ds_cpu_optimizer .. True
[2023-07-18 18:07:50,078] [INFO] [config.py:964:print]   zero_optimization_stage ...... 3
[2023-07-18 18:07:50,078] [INFO] [config.py:950:print_user_config]   json = {
    "train_batch_size": 8,
    "train_micro_batch_size_per_gpu": 8,
    "gradient_accumulation_steps": 1,
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "none",
            "nvme_path": null
        },
        "offload_param": {
            "device": "none",
            "nvme_path": null
        },
        "stage3_gather_16bit_weights_on_model_save": true
    },
    "gradient_clipping": 1.0,
    "steps_per_print": inf,
    "fp16": {
        "enabled": false
    },
    "bf16": {
        "enabled": false
    },
    "zero_allow_untested_optimizer": true
}
DeepSpeedEngine(
  (module): PeftModelForCausalLM(
    (base_model): LoraModel(
      (model): BloomForCausalLM(
        (transformer): BloomModel(
          (word_embeddings): Embedding(250880, 1024)
          (word_embeddings_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          (h): ModuleList(
            (0): BloomBlock(
              (input_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (self_attention): BloomAttention(
                (query_key_value): Linear(
                  in_features=1024, out_features=3072, bias=True
                  (lora_dropout): ModuleDict(
                    (default): Dropout(p=0.1, inplace=False)
                  )
                  (lora_A): ModuleDict(
                    (default): Linear(in_features=1024, out_features=8, bias=False)
                  )
                  (lora_B): ModuleDict(
                    (default): Linear(in_features=8, out_features=3072, bias=False)
                  )
                  (lora_embedding_A): ParameterDict()
                  (lora_embedding_B): ParameterDict()
                )
                (dense): Linear(in_features=1024, out_features=1024, bias=True)
                (attention_dropout): Dropout(p=0.0, inplace=False)
              )
              (post_attention_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (mlp): BloomMLP(
                (dense_h_to_4h): Linear(in_features=1024, out_features=4096, bias=True)
                (gelu_impl): BloomGelu()
                (dense_4h_to_h): Linear(in_features=4096, out_features=1024, bias=True)
              )
            )
           ...
            (23): BloomBlock(
              (input_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (self_attention): BloomAttention(
                (query_key_value): Linear(
                  in_features=1024, out_features=3072, bias=True
                  (lora_dropout): ModuleDict(
                    (default): Dropout(p=0.1, inplace=False)
                  )
                  (lora_A): ModuleDict(
                    (default): Linear(in_features=1024, out_features=8, bias=False)
                  )
                  (lora_B): ModuleDict(
                    (default): Linear(in_features=8, out_features=3072, bias=False)
                  )
                  (lora_embedding_A): ParameterDict()
                  (lora_embedding_B): ParameterDict()
                )
                (dense): Linear(in_features=1024, out_features=1024, bias=True)
                (attention_dropout): Dropout(p=0.0, inplace=False)
              )
              (post_attention_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (mlp): BloomMLP(
                (dense_h_to_4h): Linear(in_features=1024, out_features=4096, bias=True)
                (gelu_impl): BloomGelu()
                (dense_4h_to_h): Linear(in_features=4096, out_features=1024, bias=True)
              )
            )
          )
          (ln_f): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        )
        (lm_head): Linear(in_features=1024, out_features=250880, bias=False)
      )
    )
  )
)
/home/guodong.li/virtual-venv/peft-venv-py310-cu117/lib/python3.10/site-packages/torch/cuda/memory.py:282: FutureWarning: torch.cuda.reset_max_memory_allocated now calls torch.cuda.reset_peak_memory_stats, which resets /all/ peak memory stats.
  warnings.warn(
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:04<00:00,  1.62it/s]
GPU Memory before entering the train : 5038
GPU Memory consumed at the end of the train (end-begin): 146
GPU Peak Memory consumed during the train (max-begin): 2917
GPU Total Peak Memory consumed during the train (max): 7955
CPU Memory before entering the train : 1798
CPU Memory consumed at the end of the train (end-begin): 939
CPU Peak Memory consumed during the train (max-begin): 939
CPU Total Peak Memory consumed during the train (max): 2737
epoch=0: train_ppl=tensor(8.6437e+09, device='cuda:0') train_epoch_loss=tensor(22.8801, device='cuda:0')
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:13<00:00,  1.99s/it]
GPU Memory before entering the eval : 5184
GPU Memory consumed at the end of the eval (end-begin): -146
GPU Peak Memory consumed during the eval (max-begin): 692
GPU Total Peak Memory consumed during the eval (max): 5876
CPU Memory before entering the eval : 2737
CPU Memory consumed at the end of the eval (end-begin): 5
CPU Peak Memory consumed during the eval (max-begin): 5
CPU Total Peak Memory consumed during the eval (max): 2742
accuracy=0.0
eval_preds[:10]=['complaint complaint yang diutarakan oleh polisi, polisi', 'complaint complaint yang sudah diinstruksikan oleh polisi', 'complaint polisi juga complaint yang sudah diinstruksikan', 'complaint complaint yang diutarakan oleh polisi, polisi', 'complaint complaint yang kedua, polisi juga harus meminta', 'complaint polisi juga complaint yang diajukan polisi di negara', 'complaint yang harus didỏi adalah complaint yang harus', 'complaint polisi juga meminta maaf dan meminta maaf kepada', 'complaint polisi juga complaint yang dikeluarkan polisi di negara', 'complaint complaint yang diutarakan oleh polisi, polisi']
dataset['train'][label_column][:10]=['no complaint', 'no complaint', 'complaint', 'complaint', 'no complaint', 'no complaint', 'no complaint', 'complaint', 'complaint', 'no complaint']
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.34it/s]
GPU Memory before entering the train : 5038
GPU Memory consumed at the end of the train (end-begin): 146
GPU Peak Memory consumed during the train (max-begin): 2917
GPU Total Peak Memory consumed during the train (max): 7955
CPU Memory before entering the train : 2743
CPU Memory consumed at the end of the train (end-begin): 0
CPU Peak Memory consumed during the train (max-begin): 0
CPU Total Peak Memory consumed during the train (max): 2743
epoch=1: train_ppl=tensor(52.4372, device='cuda:0') train_epoch_loss=tensor(3.9596, device='cuda:0')
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:14<00:00,  2.04s/it]
GPU Memory before entering the eval : 5184
GPU Memory consumed at the end of the eval (end-begin): -146
GPU Peak Memory consumed during the eval (max-begin): 692
GPU Total Peak Memory consumed during the eval (max): 5876
CPU Memory before entering the eval : 2743
CPU Memory consumed at the end of the eval (end-begin): 0
CPU Peak Memory consumed during the eval (max-begin): 0
CPU Total Peak Memory consumed during the eval (max): 2743
accuracy=0.0
eval_preds[:10]=['no complaintossos್ಬ البطنಿಸteznoಿಖ economista', 'no complaintেষ্টেষ্ট Magdalnoಿಖ economistano complaint', 'no complaintেষ্টnoضاًضاًضاًਧਮाशिवाय', 'no complaintছ البطنುಗಳನ್ನುਧನೆಯಲ್ಲಿছেষ্টestant', 'no complaint細 البطن البطنno complaint أعلمضاً密西西比', 'no complaint阿拉斯加no complaint أعلمುಗಳನ್ನು্যওছಿಸ', 'no complaintছ البطنছਧਮnoেষ্টেষ্ট', 'no complaint阿拉斯加ossosਧನೆಯಲ್ಲಿেছে,no complaint', 'no complaint أعلم أعلمضاًضاًضاًਮেছে,', 'no complaintossosਧਮno執法ਬ البطنಿಸ']
dataset['train'][label_column][:10]=['no complaint', 'no complaint', 'complaint', 'complaint', 'no complaint', 'no complaint', 'no complaint', 'complaint', 'complaint', 'no complaint']
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.34it/s]
GPU Memory before entering the train : 5038
GPU Memory consumed at the end of the train (end-begin): 146
GPU Peak Memory consumed during the train (max-begin): 2917
GPU Total Peak Memory consumed during the train (max): 7955
CPU Memory before entering the train : 2744
CPU Memory consumed at the end of the train (end-begin): 0
CPU Peak Memory consumed during the train (max-begin): 0
CPU Total Peak Memory consumed during the train (max): 2744
epoch=2: train_ppl=tensor(7.5966, device='cuda:0') train_epoch_loss=tensor(2.0277, device='cuda:0')
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:13<00:00,  1.97s/it]
GPU Memory before entering the eval : 5184
GPU Memory consumed at the end of the eval (end-begin): -146
GPU Peak Memory consumed during the eval (max-begin): 692
GPU Total Peak Memory consumed during the eval (max): 5876
CPU Memory before entering the eval : 2744
CPU Memory consumed at the end of the eval (end-begin): 0
CPU Peak Memory consumed during the eval (max-begin): 0
CPU Total Peak Memory consumed during the eval (max): 2744
accuracy=12.0
eval_preds[:10]=['complaint', 'complaint buồn buồn', 'complaint', 'complaint buồn buồn', 'complaint buồn', 'complaint buồn', 'complaint buồn buồn buồn', 'complaint', 'complaint buồn', 'complaint']
dataset['train'][label_column][:10]=['no complaint', 'no complaint', 'complaint', 'complaint', 'no complaint', 'no complaint', 'no complaint', 'complaint', 'complaint', 'no complaint']
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:03<00:00,  2.13it/s]
...
GPU Memory before entering the train : 5038
GPU Memory consumed at the end of the train (end-begin): 146
GPU Peak Memory consumed during the train (max-begin): 2917
GPU Total Peak Memory consumed during the train (max): 7955
CPU Memory before entering the train : 2752
CPU Memory consumed at the end of the train (end-begin): 0
CPU Peak Memory consumed during the train (max-begin): 0
CPU Total Peak Memory consumed during the train (max): 2752
epoch=9: train_ppl=tensor(1.0111, device='cuda:0') train_epoch_loss=tensor(0.0110, device='cuda:0')
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:16<00:00,  2.36s/it]
GPU Memory before entering the eval : 5184
GPU Memory consumed at the end of the eval (end-begin): -146
GPU Peak Memory consumed during the eval (max-begin): 692
GPU Total Peak Memory consumed during the eval (max): 5876
CPU Memory before entering the eval : 2752
CPU Memory consumed at the end of the eval (end-begin): 0
CPU Peak Memory consumed during the eval (max-begin): 0
CPU Total Peak Memory consumed during the eval (max): 2752
accuracy=100.0
eval_preds[:10]=['no complaint', 'no complaint', 'complaint', 'complaint', 'no complaint', 'no complaint', 'no complaint', 'complaint', 'complaint', 'no complaint']
dataset['train'][label_column][:10]=['no complaint', 'no complaint', 'complaint', 'complaint', 'no complaint', 'no complaint', 'no complaint', 'complaint', 'complaint', 'no complaint']
model_output: /data/nfs/llm/model/bloomz-560m_LORA_CAUSAL_LM
```
  
</p></details>





```
> tree -h /data/nfs/llm/model/bloomz-560m_LORA_CAUSAL_LM
/data/nfs/llm/model/bloomz-560m_LORA_CAUSAL_LM
├── [ 447]  adapter_config.json
├── [ 14K]  adapter_model.bin
└── [  93]  README.md

0 directories, 3 files
```




### 大模型参数高效微调技术实战（四）-AdaLoRA


### 大模型参数高效微调技术实战（五）-QLoRA










- 大模型参数高效微调技术实战（一）-Prefix Tuning 
- 大模型参数高效微调技术实战（二）-Prompt Tuning
- 大模型参数高效微调技术实战（三）-P-Tuning
- 大模型参数高效微调技术实战（三）-LoRA
- 大模型参数高效微调技术实战（四）-AdaLoRA
- 大模型参数高效微调技术实战（五）-QLoRA


