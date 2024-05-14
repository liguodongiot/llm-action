

### 单机单卡


<details><summary>详细输出：</summary><p>

```
CUDA_VISIBLE_DEVICES=3 sh examples/pretrain_gpt.sh 
using world size: 1, data-parallel-size: 1, tensor-model-parallel size: 1, pipeline-model-parallel size: 1 
using torch.float16 for parameters ...
------------------------ arguments ------------------------
  accumulate_allreduce_grads_in_fp32 .............. False
  adam_beta1 ...................................... 0.9
  adam_beta2 ...................................... 0.999
  adam_eps ........................................ 1e-08
  add_bias_linear ................................. True
  add_position_embedding .......................... True
  adlr_autoresume ................................. False
  adlr_autoresume_interval ........................ 1000
  apply_layernorm_1p .............................. False
  apply_query_key_layer_scaling ................... True
  apply_residual_connection_post_layernorm ........ False
  async_tensor_model_parallel_allreduce ........... True
  attention_dropout ............................... 0.1
  attention_softmax_in_fp32 ....................... False
  barrier_with_L1_time ............................ True
  bert_binary_head ................................ True
  bert_embedder_type .............................. megatron
  bert_load ....................................... None
  bf16 ............................................ False
  bias_dropout_fusion ............................. True
  bias_gelu_fusion ................................ True
  biencoder_projection_dim ........................ 0
  biencoder_shared_query_context_model ............ False
  block_data_path ................................. None
  classes_fraction ................................ 1.0
  clip_grad ....................................... 1.0
  consumed_train_samples .......................... 0
  consumed_valid_samples .......................... 0
  data_cache_path ................................. None
  data_impl ....................................... mmap
  data_parallel_random_init ....................... False
  data_parallel_size .............................. 1
  data_path ....................................... ['/workspace/data/my-gpt2_text_document']
  data_per_class_fraction ......................... 1.0
  data_sharding ................................... True
  dataloader_type ................................. single
  DDP_impl ........................................ local
  decoder_num_layers .............................. None
  decoder_seq_length .............................. None
  dino_bottleneck_size ............................ 256
  dino_freeze_last_layer .......................... 1
  dino_head_hidden_size ........................... 2048
  dino_local_crops_number ......................... 10
  dino_local_img_size ............................. 96
  dino_norm_last_layer ............................ False
  dino_teacher_temp ............................... 0.07
  dino_warmup_teacher_temp ........................ 0.04
  dino_warmup_teacher_temp_epochs ................. 30
  distribute_saved_activations .................... False
  distributed_backend ............................. nccl
  distributed_timeout_minutes ..................... 10
  embedding_path .................................. None
  empty_unused_memory_level ....................... 0
  encoder_num_layers .............................. 24
  encoder_seq_length .............................. 1024
  end_weight_decay ................................ 0.01
  eod_mask_loss ................................... False
  eval_interval ................................... 1000
  eval_iters ...................................... 10
  evidence_data_path .............................. None
  exit_duration_in_mins ........................... None
  exit_interval ................................... None
  exit_on_missing_checkpoint ...................... False
  exit_signal_handler ............................. False
  ffn_hidden_size ................................. 4096
  finetune ........................................ False
  fp16 ............................................ True
  fp16_lm_cross_entropy ........................... False
  fp32_residual_connection ........................ False
  fp8_amax_compute_algo ........................... most_recent
  fp8_amax_history_len ............................ 1
  fp8_e4m3 ........................................ False
  fp8_hybrid ...................................... False
  fp8_interval .................................... 1
  fp8_margin ...................................... 0
  fp8_wgrad ....................................... True
  global_batch_size ............................... 2
  gradient_accumulation_fusion .................... True
  head_lr_mult .................................... 1.0
  hidden_dropout .................................. 0.1
  hidden_size ..................................... 1024
  hysteresis ...................................... 2
  ict_head_size ................................... None
  ict_load ........................................ None
  img_h ........................................... 224
  img_w ........................................... 224
  indexer_batch_size .............................. 128
  indexer_log_interval ............................ 1000
  inference_batch_times_seqlen_threshold .......... 512
  init_method_std ................................. 0.02
  init_method_xavier_uniform ...................... False
  initial_loss_scale .............................. 4294967296
  iter_per_epoch .................................. 1250
  kv_channels ..................................... 64
  layernorm_epsilon ............................... 1e-05
  lazy_mpu_init ................................... None
  load ............................................ /workspace/model/megatron-models/345m
  local_rank ...................................... None
  log_batch_size_to_tensorboard ................... False
  log_interval .................................... 100
  log_learning_rate_to_tensorboard ................ True
  log_loss_scale_to_tensorboard ................... True
  log_memory_to_tensorboard ....................... False
  log_num_zeros_in_grad ........................... False
  log_params_norm ................................. False
  log_timers_to_tensorboard ....................... False
  log_validation_ppl_to_tensorboard ............... False
  log_world_size_to_tensorboard ................... False
  loss_scale ...................................... None
  loss_scale_window ............................... 1000
  lr .............................................. 0.00015
  lr_decay_iters .................................. 320000
  lr_decay_samples ................................ None
  lr_decay_style .................................. cosine
  lr_warmup_fraction .............................. 0.01
  lr_warmup_iters ................................. 0
  lr_warmup_samples ............................... 0
  make_vocab_size_divisible_by .................... 128
  mask_factor ..................................... 1.0
  mask_prob ....................................... 0.15
  mask_type ....................................... random
  masked_softmax_fusion ........................... True
  max_position_embeddings ......................... 1024
  max_tokens_to_oom ............................... 12000
  merge_file ...................................... /workspace/model/gpt2-vocab/gpt2-merges.txt
  micro_batch_size ................................ 1
  min_loss_scale .................................. 1.0
  min_lr .......................................... 1e-05
  mmap_warmup ..................................... False
  no_load_optim ................................... None
  no_load_rng ..................................... None
  no_persist_layer_norm ........................... False
  no_save_optim ................................... None
  no_save_rng ..................................... None
  num_attention_heads ............................. 16
  num_channels .................................... 3
  num_classes ..................................... 1000
  num_experts ..................................... None
  num_layers ...................................... 24
  num_layers_per_virtual_pipeline_stage ........... None
  num_workers ..................................... 2
  onnx_safe ....................................... None
  openai_gelu ..................................... False
  optimizer ....................................... adam
  output_bert_embeddings .......................... False
  overlap_p2p_comm ................................ False
  override_opt_param_scheduler .................... False
  params_dtype .................................... torch.float16
  patch_dim ....................................... 16
  perform_initialization .......................... True
  pipeline_model_parallel_size .................... 1
  pipeline_model_parallel_split_rank .............. None
  query_in_block_prob ............................. 0.1
  rampup_batch_size ............................... None
  rank ............................................ 0
  recompute_granularity ........................... None
  recompute_method ................................ None
  recompute_num_layers ............................ 1
  reset_attention_mask ............................ False
  reset_position_ids .............................. False
  retriever_report_topk_accuracies ................ []
  retriever_score_scaling ......................... False
  retriever_seq_length ............................ 256
  retro_add_retriever ............................. False
  retro_cyclic_train_iters ........................ None
  retro_encoder_attention_dropout ................. 0.1
  retro_encoder_hidden_dropout .................... 0.1
  retro_encoder_layers ............................ 2
  retro_num_neighbors ............................. 2
  retro_num_retrieved_chunks ...................... 2
  retro_return_doc_ids ............................ False
  retro_workdir ................................... None
  rotary_percent .................................. 1.0
  sample_rate ..................................... 1.0
  save ............................................ /workspace/model/megatron-models/345m
  save_interval ................................... 10000
  scatter_gather_tensors_in_pipeline .............. True
  seed ............................................ 1234
  seq_length ...................................... 1024
  sequence_parallel ............................... False
  sgd_momentum .................................... 0.9
  short_seq_prob .................................. 0.1
  split ........................................... 700,200,100
  squared_relu .................................... False
  standalone_embedding_stage ...................... False
  start_weight_decay .............................. 0.01
  swiglu .......................................... False
  swin_backbone_type .............................. tiny
  tensor_model_parallel_size ...................... 1
  tensorboard_dir ................................. None
  tensorboard_log_interval ........................ 1
  tensorboard_queue_size .......................... 1000
  test_data_path .................................. None
  timing_log_level ................................ 0
  timing_log_option ............................... minmax
  titles_data_path ................................ None
  tokenizer_model ................................. None
  tokenizer_type .................................. GPT2BPETokenizer
  train_data_path ................................. None
  train_iters ..................................... 5000
  train_samples ................................... None
  transformer_impl ................................ local
  transformer_pipeline_model_parallel_size ........ 1
  untie_embeddings_and_output_weights ............. False
  use_checkpoint_args ............................. False
  use_checkpoint_opt_param_scheduler .............. False
  use_contiguous_buffers_in_local_ddp ............. True
  use_cpu_initialization .......................... None
  use_distributed_optimizer ....................... False
  use_flash_attn .................................. False
  use_one_sent_docs ............................... False
  use_ring_exchange_p2p ........................... False
  use_rotary_position_embeddings .................. False
  valid_data_path ................................. None
  variable_seq_lengths ............................ False
  virtual_pipeline_model_parallel_size ............ None
  vision_backbone_type ............................ vit
  vision_pretraining .............................. False
  vision_pretraining_type ......................... classify
  vocab_extra_ids ................................. 0
  vocab_file ...................................... /workspace/model/gpt2-vocab/gpt2-vocab.json
  vocab_size ...................................... None
  weight_decay .................................... 0.01
  weight_decay_incr_style ......................... constant
  world_size ...................................... 1
-------------------- end of arguments ---------------------
setting number of micro-batches to constant 2
> building GPT2BPETokenizer tokenizer ...
 > padded vocab (size: 50257) with 47 dummy tokens (new size: 50304)
> initializing torch distributed ...
> initialized tensor model parallel with size 1
> initialized pipeline model parallel with size 1
> setting random seeds to 1234 ...
> compiling dataset index builder ...
make: Entering directory '/workspace/code/Megatron-LM/megatron/data'
make: Nothing to be done for 'default'.
make: Leaving directory '/workspace/code/Megatron-LM/megatron/data'
>>> done with dataset index builder. Compilation time: 0.221 seconds
> compiling and loading fused kernels ...
Detected CUDA files, patching ldflags
Emitting ninja build file /workspace/code/Megatron-LM/megatron/fused_kernels/build/build.ninja...
Building extension module scaled_upper_triang_masked_softmax_cuda...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
ninja: no work to do.
Loading extension module scaled_upper_triang_masked_softmax_cuda...
Detected CUDA files, patching ldflags
Emitting ninja build file /workspace/code/Megatron-LM/megatron/fused_kernels/build/build.ninja...
Building extension module scaled_masked_softmax_cuda...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
ninja: no work to do.
Loading extension module scaled_masked_softmax_cuda...
Detected CUDA files, patching ldflags
Emitting ninja build file /workspace/code/Megatron-LM/megatron/fused_kernels/build/build.ninja...
Building extension module scaled_softmax_cuda...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
ninja: no work to do.
Loading extension module scaled_softmax_cuda...
>>> done with compiling and loading fused kernels. Compilation time: 1.348 seconds
time to initialize megatron (seconds): 3.520
[after megatron is initialized] datetime: 2023-07-16 10:56:55 
building GPT model ...
 > number of parameters on (tensor, pipeline) model parallel rank (0, 0): 354871296
> learning rate decay style: cosine
 loading checkpoint from /workspace/model/megatron-models/345m at iteration 5000
 checkpoint version 3.0
 > using checkpoint value 0.00015 for learning rate
 > using checkpoint value 1e-05 for minimum learning rate
 > using checkpoint value 6400.0 for warmup iterations
 > using checkpoint value 640000 for total number of iterations
 > using checkpoint value cosine for learning rate decay style
 > using checkpoint value 0.01 for start weight decay
 > using checkpoint value 0.01 for end weight decay
 > using checkpoint value 10000 for total number of weight decay iterations
 > using checkpoint value constant for weight decay incr style
  successfully loaded checkpoint from /workspace/model/megatron-models/345m at iteration 5000
/usr/local/lib/python3.8/dist-packages/torch/distributed/distributed_c10d.py:2603: UserWarning: torch.distributed._all_gather_base is a private function and will be deprecated. Please use torch.distributed.all_gather_into_tensor instead.
  warnings.warn(
(min, max) time across ranks (ms):
    load-checkpoint ................................: (4542.10, 4542.10)
[after model, optimizer, and learning rate scheduler are built] datetime: 2023-07-16 10:57:00 
> building train, validation, and test datasets ...
 > datasets target sizes (minimum size):
    train:      10000
    validation: 120
    test:       20
> building train, validation, and test datasets for GPT ...
Single data path provided for train, valid & test
data_prefix: ['/workspace/data/my-gpt2_text_document']
 > building dataset index ...
    reading sizes...
    reading pointers...
    reading document index...
    creating numpy buffer of mmap...
    creating memory view of numpy buffer...
 > finished creating indexed dataset in 0.003545 seconds
    number of documents: 2456
 > dataset split:
    train:
     document indices in [0, 1719) total of 1719 documents
    validation:
     document indices in [1719, 2210) total of 491 documents
    test:
     document indices in [2210, 2456) total of 246 documents
 > loading doc-idx mapping from /workspace/data/index-cache/3c048217962a127246af1e2322a522e7_doc_idx.npy
 > loading sample-idx mapping from /workspace/data/index-cache/3c048217962a127246af1e2322a522e7_sample_idx.npy
 > loading shuffle-idx mapping from /workspace/data/index-cache/3c048217962a127246af1e2322a522e7_shuffle_idx.npy
    loaded indexed file in 0.006 seconds
    total number of samples: 10411
    total number of epochs: 23
 > loading doc-idx mapping from /workspace/data/index-cache/c96454bddc79ae7749ebb071e254a5ab_doc_idx.npy
 > loading sample-idx mapping from /workspace/data/index-cache/c96454bddc79ae7749ebb071e254a5ab_sample_idx.npy
 > loading shuffle-idx mapping from /workspace/data/index-cache/c96454bddc79ae7749ebb071e254a5ab_shuffle_idx.npy
    loaded indexed file in 0.005 seconds
    total number of samples: 222
    total number of epochs: 2
 > loading doc-idx mapping from /workspace/data/index-cache/c3c889117317b1643e4d407b279b17a5_doc_idx.npy
 > loading sample-idx mapping from /workspace/data/index-cache/c3c889117317b1643e4d407b279b17a5_sample_idx.npy
 > loading shuffle-idx mapping from /workspace/data/index-cache/c3c889117317b1643e4d407b279b17a5_shuffle_idx.npy
    loaded indexed file in 0.005 seconds
    total number of samples: 54
    total number of epochs: 1
> finished creating GPT datasets ...
[after dataloaders are built] datetime: 2023-07-16 10:57:01 
done with setup ...
(min, max) time across ranks (ms):
    model-and-optimizer-setup ......................: (4690.91, 4690.91)
    train/valid/test-data-iterators-setup ..........: (669.89, 669.89)
training ...
[before the start of training step] datetime: 2023-07-16 10:57:01 
[after training is done] datetime: 2023-07-16 10:57:01 
------------------------------------------------------------------------------------------------------------------
 validation loss at the end of training for val data | lm loss value: 5.741563E+00 | lm loss PPL: 3.115511E+02 | 
------------------------------------------------------------------------------------------------------------------
saving checkpoint at iteration    5000 to /workspace/model/megatron-models/345m

  successfully saved checkpoint at iteration    5000 to /workspace/model/megatron-models/345m
-------------------------------------------------------------------------------------------------------------------
 validation loss at the end of training for test data | lm loss value: 6.015863E+00 | lm loss PPL: 4.098796E+02 | 
-------------------------------------------------------------------------------------------------------------------
```
</p></details>



输出权重：

```
> tree -h 345m
345m
├── [4.0K]  iter_0005000
│   └── [4.0K]  mp_rank_00
│       └── [4.6G]  model_optim_rng.pt
├── [   4]  latest_checkpointed_iteration.txt
└── [4.0K]  release
    └── [4.0K]  mp_rank_00
        └── [677M]  model_optim_rng.pt

4 directories, 3 files

> cat 345m/latest_checkpointed_iteration.txt 
5000
```

### 单机多卡(4DP)


模型显存占用：

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.105.17   Driver Version: 525.105.17   CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA A800 80G...  Off  | 00000000:17:00.0 Off |                    0 |
| N/A   49C    P0   183W / 300W |   9661MiB / 81920MiB |     71%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA A800 80G...  Off  | 00000000:31:00.0 Off |                    0 |
| N/A   50C    P0   136W / 300W |   9661MiB / 81920MiB |     67%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   2  NVIDIA A800 80G...  Off  | 00000000:B1:00.0 Off |                    0 |
| N/A   49C    P0   202W / 300W |   9663MiB / 81920MiB |     65%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   3  NVIDIA A800 80G...  Off  | 00000000:CA:00.0 Off |                    0 |
| N/A   50C    P0   227W / 300W |   9663MiB / 81920MiB |     64%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A   3227288      C   /usr/bin/python                  9652MiB |
|    1   N/A  N/A   3227289      C   /usr/bin/python                  9652MiB |
|    2   N/A  N/A   3227290      C   /usr/bin/python                  9652MiB |
|    3   N/A  N/A   3227291      C   /usr/bin/python                  9652MiB |
+-----------------------------------------------------------------------------+
```

模型权重输出：

```
> tree -h /workspace/model/megatron-models/345m-init
/workspace/model/megatron-models/345m-init
├── [4.0K]  iter_0005000
│   └── [4.0K]  mp_rank_00
│       └── [4.6G]  model_optim_rng.pt
└── [   4]  latest_checkpointed_iteration.txt

2 directories, 2 files

> cat /workspace/model/megatron-models/345m-init/latest_checkpointed_iteration.txt 
5000
```


<details><summary>详细输出：</summary><p>

```
sh examples/pretrain_gpt_distributed.sh 
WARNING:torch.distributed.run:
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
*****************************************
using world size: 4, data-parallel-size: 4, tensor-model-parallel size: 1, pipeline-model-parallel size: 1 
using torch.float16 for parameters ...
------------------------ arguments ------------------------
  accumulate_allreduce_grads_in_fp32 .............. False
  adam_beta1 ...................................... 0.9
  adam_beta2 ...................................... 0.999
  adam_eps ........................................ 1e-08
  add_bias_linear ................................. True
  add_position_embedding .......................... True
  adlr_autoresume ................................. False
  adlr_autoresume_interval ........................ 1000
  apply_layernorm_1p .............................. False
  apply_query_key_layer_scaling ................... True
  apply_residual_connection_post_layernorm ........ False
  async_tensor_model_parallel_allreduce ........... True
  attention_dropout ............................... 0.1
  attention_softmax_in_fp32 ....................... False
  barrier_with_L1_time ............................ True
  bert_binary_head ................................ True
  bert_embedder_type .............................. megatron
  bert_load ....................................... None
  bf16 ............................................ False
  bias_dropout_fusion ............................. True
  bias_gelu_fusion ................................ True
  biencoder_projection_dim ........................ 0
  biencoder_shared_query_context_model ............ False
  block_data_path ................................. None
  classes_fraction ................................ 1.0
  clip_grad ....................................... 1.0
  consumed_train_samples .......................... 0
  consumed_valid_samples .......................... 0
  data_cache_path ................................. None
  data_impl ....................................... mmap
  data_parallel_random_init ....................... False
  data_parallel_size .............................. 4
  data_path ....................................... ['/workspace/data/my-gpt2_text_document']
  data_per_class_fraction ......................... 1.0
  data_sharding ................................... True
  dataloader_type ................................. single
  DDP_impl ........................................ local
  decoder_num_layers .............................. None
  decoder_seq_length .............................. None
  dino_bottleneck_size ............................ 256
  dino_freeze_last_layer .......................... 1
  dino_head_hidden_size ........................... 2048
  dino_local_crops_number ......................... 10
  dino_local_img_size ............................. 96
  dino_norm_last_layer ............................ False
  dino_teacher_temp ............................... 0.07
  dino_warmup_teacher_temp ........................ 0.04
  dino_warmup_teacher_temp_epochs ................. 30
  distribute_saved_activations .................... False
  distributed_backend ............................. nccl
  distributed_timeout_minutes ..................... 10
  embedding_path .................................. None
  empty_unused_memory_level ....................... 0
  encoder_num_layers .............................. 24
  encoder_seq_length .............................. 1024
  end_weight_decay ................................ 0.01
  eod_mask_loss ................................... False
  eval_interval ................................... 1000
  eval_iters ...................................... 10
  evidence_data_path .............................. None
  exit_duration_in_mins ........................... None
  exit_interval ................................... None
  exit_on_missing_checkpoint ...................... False
  exit_signal_handler ............................. False
  ffn_hidden_size ................................. 4096
  finetune ........................................ False
  fp16 ............................................ True
  fp16_lm_cross_entropy ........................... False
  fp32_residual_connection ........................ False
  fp8_amax_compute_algo ........................... most_recent
  fp8_amax_history_len ............................ 1
  fp8_e4m3 ........................................ False
  fp8_hybrid ...................................... False
  fp8_interval .................................... 1
  fp8_margin ...................................... 0
  fp8_wgrad ....................................... True
  global_batch_size ............................... 8
  gradient_accumulation_fusion .................... True
  head_lr_mult .................................... 1.0
  hidden_dropout .................................. 0.1
  hidden_size ..................................... 1024
  hysteresis ...................................... 2
  ict_head_size ................................... None
  ict_load ........................................ None
  img_h ........................................... 224
  img_w ........................................... 224
  indexer_batch_size .............................. 128
  indexer_log_interval ............................ 1000
  inference_batch_times_seqlen_threshold .......... 512
  init_method_std ................................. 0.02
  init_method_xavier_uniform ...................... False
  initial_loss_scale .............................. 4294967296
  iter_per_epoch .................................. 1250
  kv_channels ..................................... 64
  layernorm_epsilon ............................... 1e-05
  lazy_mpu_init ................................... None
  load ............................................ /workspace/model/megatron-models/345m-init
  local_rank ...................................... None
  log_batch_size_to_tensorboard ................... False
  log_interval .................................... 100
  log_learning_rate_to_tensorboard ................ True
  log_loss_scale_to_tensorboard ................... True
  log_memory_to_tensorboard ....................... False
  log_num_zeros_in_grad ........................... False
  log_params_norm ................................. False
  log_timers_to_tensorboard ....................... False
  log_validation_ppl_to_tensorboard ............... False
  log_world_size_to_tensorboard ................... False
  loss_scale ...................................... None
  loss_scale_window ............................... 1000
  lr .............................................. 0.00015
  lr_decay_iters .................................. 320000
  lr_decay_samples ................................ None
  lr_decay_style .................................. cosine
  lr_warmup_fraction .............................. 0.01
  lr_warmup_iters ................................. 0
  lr_warmup_samples ............................... 0
  make_vocab_size_divisible_by .................... 128
  mask_factor ..................................... 1.0
  mask_prob ....................................... 0.15
  mask_type ....................................... random
  masked_softmax_fusion ........................... True
  max_position_embeddings ......................... 1024
  max_tokens_to_oom ............................... 12000
  merge_file ...................................... /workspace/model/gpt2-vocab/gpt2-merges.txt
  micro_batch_size ................................ 1
  min_loss_scale .................................. 1.0
  min_lr .......................................... 1e-05
  mmap_warmup ..................................... False
  no_load_optim ................................... None
  no_load_rng ..................................... None
  no_persist_layer_norm ........................... False
  no_save_optim ................................... None
  no_save_rng ..................................... None
  num_attention_heads ............................. 16
  num_channels .................................... 3
  num_classes ..................................... 1000
  num_experts ..................................... None
  num_layers ...................................... 24
  num_layers_per_virtual_pipeline_stage ........... None
  num_workers ..................................... 2
  onnx_safe ....................................... None
  openai_gelu ..................................... False
  optimizer ....................................... adam
  output_bert_embeddings .......................... False
  overlap_p2p_comm ................................ False
  override_opt_param_scheduler .................... False
  params_dtype .................................... torch.float16
  patch_dim ....................................... 16
  perform_initialization .......................... True
  pipeline_model_parallel_size .................... 1
  pipeline_model_parallel_split_rank .............. None
  query_in_block_prob ............................. 0.1
  rampup_batch_size ............................... None
  rank ............................................ 0
  recompute_granularity ........................... None
  recompute_method ................................ None
  recompute_num_layers ............................ 1
  reset_attention_mask ............................ False
  reset_position_ids .............................. False
  retriever_report_topk_accuracies ................ []
  retriever_score_scaling ......................... False
  retriever_seq_length ............................ 256
  retro_add_retriever ............................. False
  retro_cyclic_train_iters ........................ None
  retro_encoder_attention_dropout ................. 0.1
  retro_encoder_hidden_dropout .................... 0.1
  retro_encoder_layers ............................ 2
  retro_num_neighbors ............................. 2
  retro_num_retrieved_chunks ...................... 2
  retro_return_doc_ids ............................ False
  retro_workdir ................................... None
  rotary_percent .................................. 1.0
  sample_rate ..................................... 1.0
  save ............................................ /workspace/model/megatron-models/345m-init
  save_interval ................................... 10000
  scatter_gather_tensors_in_pipeline .............. True
  seed ............................................ 1234
  seq_length ...................................... 1024
  sequence_parallel ............................... False
  sgd_momentum .................................... 0.9
  short_seq_prob .................................. 0.1
  split ........................................... 700,200,100
  squared_relu .................................... False
  standalone_embedding_stage ...................... False
  start_weight_decay .............................. 0.01
  swiglu .......................................... False
  swin_backbone_type .............................. tiny
  tensor_model_parallel_size ...................... 1
  tensorboard_dir ................................. None
  tensorboard_log_interval ........................ 1
  tensorboard_queue_size .......................... 1000
  test_data_path .................................. None
  timing_log_level ................................ 0
  timing_log_option ............................... minmax
  titles_data_path ................................ None
  tokenizer_model ................................. None
  tokenizer_type .................................. GPT2BPETokenizer
  train_data_path ................................. None
  train_iters ..................................... 5000
  train_samples ................................... None
  transformer_impl ................................ local
  transformer_pipeline_model_parallel_size ........ 1
  untie_embeddings_and_output_weights ............. False
  use_checkpoint_args ............................. False
  use_checkpoint_opt_param_scheduler .............. False
  use_contiguous_buffers_in_local_ddp ............. True
  use_cpu_initialization .......................... None
  use_distributed_optimizer ....................... False
  use_flash_attn .................................. False
  use_one_sent_docs ............................... False
  use_ring_exchange_p2p ........................... False
  use_rotary_position_embeddings .................. False
  valid_data_path ................................. None
  variable_seq_lengths ............................ False
  virtual_pipeline_model_parallel_size ............ None
  vision_backbone_type ............................ vit
  vision_pretraining .............................. False
  vision_pretraining_type ......................... classify
  vocab_extra_ids ................................. 0
  vocab_file ...................................... /workspace/model/gpt2-vocab/gpt2-vocab.json
  vocab_size ...................................... None
  weight_decay .................................... 0.01
  weight_decay_incr_style ......................... constant
  world_size ...................................... 4
-------------------- end of arguments ---------------------
setting number of micro-batches to constant 2
> building GPT2BPETokenizer tokenizer ...
 > padded vocab (size: 50257) with 47 dummy tokens (new size: 50304)
> initializing torch distributed ...
> initialized tensor model parallel with size 1
> initialized pipeline model parallel with size 1
> setting random seeds to 1234 ...
> compiling dataset index builder ...
make: Entering directory '/workspace/code/Megatron-LM/megatron/data'
make: Nothing to be done for 'default'.
make: Leaving directory '/workspace/code/Megatron-LM/megatron/data'
>>> done with dataset index builder. Compilation time: 0.211 seconds
> compiling and loading fused kernels ...
Detected CUDA files, patching ldflags
Emitting ninja build file /workspace/code/Megatron-LM/megatron/fused_kernels/build/build.ninja...
Building extension module scaled_upper_triang_masked_softmax_cuda...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
ninja: no work to do.
Loading extension module scaled_upper_triang_masked_softmax_cuda...
Detected CUDA files, patching ldflags
Emitting ninja build file /workspace/code/Megatron-LM/megatron/fused_kernels/build/build.ninja...
Building extension module scaled_masked_softmax_cuda...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
ninja: no work to do.
Loading extension module scaled_masked_softmax_cuda...
Detected CUDA files, patching ldflags
Emitting ninja build file /workspace/code/Megatron-LM/megatron/fused_kernels/build/build.ninja...
Building extension module scaled_softmax_cuda...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
ninja: no work to do.
Loading extension module scaled_softmax_cuda...
>>> done with compiling and loading fused kernels. Compilation time: 3.180 seconds
time to initialize megatron (seconds): 5.824
[after megatron is initialized] datetime: 2023-07-16 11:34:42 
building GPT model ...
 > number of parameters on (tensor, pipeline) model parallel rank (0, 0): 354871296
> learning rate decay style: cosine
WARNING: could not find the metadata file /workspace/model/megatron-models/345m-init/latest_checkpointed_iteration.txt 
    will not load any checkpoints and will start from random
/usr/local/lib/python3.8/dist-packages/torch/distributed/distributed_c10d.py:2603: UserWarning: torch.distributed._all_gather_base is a private function and will be deprecated. Please use torch.distributed.all_gather_into_tensor instead.
  warnings.warn(
/usr/local/lib/python3.8/dist-packages/torch/distributed/distributed_c10d.py:2603: UserWarning: torch.distributed._all_gather_base is a private function and will be deprecated. Please use torch.distributed.all_gather_into_tensor instead.
  warnings.warn(
/usr/local/lib/python3.8/dist-packages/torch/distributed/distributed_c10d.py:2603: UserWarning: torch.distributed._all_gather_base is a private function and will be deprecated. Please use torch.distributed.all_gather_into_tensor instead.
  warnings.warn(
/usr/local/lib/python3.8/dist-packages/torch/distributed/distributed_c10d.py:2603: UserWarning: torch.distributed._all_gather_base is a private function and will be deprecated. Please use torch.distributed.all_gather_into_tensor instead.
  warnings.warn(
(min, max) time across ranks (ms):
    load-checkpoint ................................: (1.15, 1.15)
[after model, optimizer, and learning rate scheduler are built] datetime: 2023-07-16 11:34:42 
> building train, validation, and test datasets ...
 > datasets target sizes (minimum size):
    train:      40000
    validation: 480
    test:       80
> building train, validation, and test datasets for GPT ...
Single data path provided for train, valid & test
data_prefix: ['/workspace/data/my-gpt2_text_document']
 > building dataset index ...
    reading sizes...
    reading pointers...
    reading document index...
    creating numpy buffer of mmap...
    creating memory view of numpy buffer...
 > finished creating indexed dataset in 0.002571 seconds
    number of documents: 2456
 > dataset split:
    train:
     document indices in [0, 1719) total of 1719 documents
    validation:
     document indices in [1719, 2210) total of 491 documents
    test:
     document indices in [2210, 2456) total of 246 documents
 > loading doc-idx mapping from /workspace/data/index-cache/c5fd135211a7052c502ac43f60ef9c1d_doc_idx.npy
 > loading sample-idx mapping from /workspace/data/index-cache/c5fd135211a7052c502ac43f60ef9c1d_sample_idx.npy
 > loading shuffle-idx mapping from /workspace/data/index-cache/c5fd135211a7052c502ac43f60ef9c1d_shuffle_idx.npy
    loaded indexed file in 0.004 seconds
    total number of samples: 40286
    total number of epochs: 89
 > loading doc-idx mapping from /workspace/data/index-cache/1a0ccca19f4fd634c52ef32cb52a4d6f_doc_idx.npy
 > loading sample-idx mapping from /workspace/data/index-cache/1a0ccca19f4fd634c52ef32cb52a4d6f_sample_idx.npy
 > loading shuffle-idx mapping from /workspace/data/index-cache/1a0ccca19f4fd634c52ef32cb52a4d6f_shuffle_idx.npy
    loaded indexed file in 0.003 seconds
    total number of samples: 555
    total number of epochs: 5
 > loading doc-idx mapping from /workspace/data/index-cache/f1583c0457c0ee9c3ecb3c068b3badd8_doc_idx.npy
 > loading sample-idx mapping from /workspace/data/index-cache/f1583c0457c0ee9c3ecb3c068b3badd8_sample_idx.npy
 > loading shuffle-idx mapping from /workspace/data/index-cache/f1583c0457c0ee9c3ecb3c068b3badd8_shuffle_idx.npy
    loaded indexed file in 0.002 seconds
    total number of samples: 108
    total number of epochs: 2
> finished creating GPT datasets ...
[after dataloaders are built] datetime: 2023-07-16 11:34:57 
done with setup ...
(min, max) time across ranks (ms):
    model-and-optimizer-setup ......................: (222.27, 225.22)
    train/valid/test-data-iterators-setup ..........: (14590.46, 14646.38)
training ...
[before the start of training step] datetime: 2023-07-16 11:34:57 
 iteration      100/    5000 | consumed samples:          800 | elapsed time per iteration (ms): 296.8 | learning rate: 3.937E-06 | global batch size:     8 | lm loss: 9.729805E+00 | loss scale: 131072.0 | grad norm: 5.703 | number of skipped iterations:  16 | number of nan iterations:   0 |
[Rank 0] (after 100 iterations) memory (MB) | allocated: 6816.88818359375 | max allocated: 8696.8515625 | reserved: 8786.0 | max reserved: 8786.0
...
 iteration     1000/    5000 | consumed samples:         8000 | elapsed time per iteration (ms): 283.9 | learning rate: 4.608E-05 | global batch size:     8 | lm loss: 5.544420E+00 | loss scale: 65536.0 | grad norm: 2.475 | number of skipped iterations:   0 | number of nan iterations:   0 |
------------------------------------------------------------------------------------------------
 validation loss at iteration 1000 | lm loss value: 6.439325E+00 | lm loss PPL: 6.259840E+02 | 
------------------------------------------------------------------------------------------------
 iteration     1100/    5000 | consumed samples:         8800 | elapsed time per iteration (ms): 291.7 | learning rate: 5.077E-05 | global batch size:     8 | lm loss: 5.352916E+00 | loss scale: 65536.0 | grad norm: 1.765 | number of skipped iterations:   0 | number of nan iterations:   0 |
...
 iteration     2000/    5000 | consumed samples:        16000 | elapsed time per iteration (ms): 292.5 | learning rate: 9.295E-05 | global batch size:     8 | lm loss: 3.812283E+00 | loss scale: 131072.0 | grad norm: 2.040 | number of skipped iterations:   0 | number of nan iterations:   0 |
------------------------------------------------------------------------------------------------
 validation loss at iteration 2000 | lm loss value: 6.188880E+00 | lm loss PPL: 4.873002E+02 | 
------------------------------------------------------------------------------------------------
 iteration     2100/    5000 | consumed samples:        16800 | elapsed time per iteration (ms): 373.8 | learning rate: 9.764E-05 | global batch size:     8 | lm loss: 3.435059E+00 | loss scale: 131072.0 | grad norm: 2.342 | number of skipped iterations:   0 | number of nan iterations:   0 |
...
 iteration     2900/    5000 | consumed samples:        23200 | elapsed time per iteration (ms): 531.2 | learning rate: 1.351E-04 | global batch size:     8 | lm loss: 1.052763E+00 | loss scale: 262144.0 | grad norm: 1.846 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration     3000/    5000 | consumed samples:        24000 | elapsed time per iteration (ms): 290.9 | learning rate: 1.398E-04 | global batch size:     8 | lm loss: 8.640376E-01 | loss scale: 262144.0 | grad norm: 2.122 | number of skipped iterations:   0 | number of nan iterations:   0 |
------------------------------------------------------------------------------------------------
 validation loss at iteration 3000 | lm loss value: 6.826255E+00 | lm loss PPL: 9.217327E+02 | 
------------------------------------------------------------------------------------------------
 iteration     3100/    5000 | consumed samples:        24800 | elapsed time per iteration (ms): 289.2 | learning rate: 1.445E-04 | global batch size:     8 | lm loss: 7.523526E-01 | loss scale: 262144.0 | grad norm: 2.145 | number of skipped iterations:   0 | number of nan iterations:   0 |
...
 iteration     4000/    5000 | consumed samples:        32000 | elapsed time per iteration (ms): 286.2 | learning rate: 1.500E-04 | global batch size:     8 | lm loss: 2.821585E-01 | loss scale: 262144.0 | grad norm: 1.031 | number of skipped iterations:   0 | number of nan iterations:   0 |
------------------------------------------------------------------------------------------------
 validation loss at iteration 4000 | lm loss value: 7.609839E+00 | lm loss PPL: 2.017953E+03 | 
------------------------------------------------------------------------------------------------
 iteration     4100/    5000 | consumed samples:        32800 | elapsed time per iteration (ms): 290.9 | learning rate: 1.500E-04 | global batch size:     8 | lm loss: 2.681582E-01 | loss scale: 262144.0 | grad norm: 1.052 | number of skipped iterations:   0 | number of nan iterations:   0 |
...
 iteration     4800/    5000 | consumed samples:        38400 | elapsed time per iteration (ms): 318.3 | learning rate: 1.500E-04 | global batch size:     8 | lm loss: 1.533272E-01 | loss scale: 131072.0 | grad norm: 0.655 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration     4900/    5000 | consumed samples:        39200 | elapsed time per iteration (ms): 288.0 | learning rate: 1.500E-04 | global batch size:     8 | lm loss: 1.449491E-01 | loss scale: 131072.0 | grad norm: 0.657 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration     5000/    5000 | consumed samples:        40000 | elapsed time per iteration (ms): 287.0 | learning rate: 1.500E-04 | global batch size:     8 | lm loss: 1.350079E-01 | loss scale: 131072.0 | grad norm: 0.706 | number of skipped iterations:   0 | number of nan iterations:   0 |
------------------------------------------------------------------------------------------------
 validation loss at iteration 5000 | lm loss value: 8.208897E+00 | lm loss PPL: 3.673487E+03 | 
------------------------------------------------------------------------------------------------
[after training is done] datetime: 2023-07-16 12:03:49 
saving checkpoint at iteration    5000 to /workspace/model/megatron-models/345m-init
------------------------------------------------------------------------------------------------------------------
 validation loss at the end of training for val data | lm loss value: 8.165035E+00 | lm loss PPL: 3.515845E+03 | 
------------------------------------------------------------------------------------------------------------------
  successfully saved checkpoint at iteration    5000 to /workspace/model/megatron-models/345m-init
-------------------------------------------------------------------------------------------------------------------
 validation loss at the end of training for test data | lm loss value: 8.428315E+00 | lm loss PPL: 4.574786E+03 | 
-------------------------------------------------------------------------------------------------------------------
```
</p></details>



### 模型并行（2TP+2DP）




<details><summary>显存使用：</summary><p>


```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.105.17   Driver Version: 525.105.17   CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA A800 80G...  Off  | 00000000:17:00.0 Off |                    0 |
| N/A   54C    P0   142W / 300W |  8732MiB / 81920MiB |    100%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA A800 80G...  Off  | 00000000:31:00.0 Off |                    0 |
| N/A   53C    P0   210W / 300W |  8732MiB / 81920MiB |    100%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   2  NVIDIA A800 80G...  Off  | 00000000:B1:00.0 Off |                    0 |
| N/A   55C    P0   290W / 300W |  6828MiB / 81920MiB |    100%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   3  NVIDIA A800 80G...  Off  | 00000000:CA:00.0 Off |                    0 |
| N/A   55C    P0   284W / 300W |  7078MiB / 81920MiB |    100%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A   3448098      C   /usr/bin/python                  8732MiB |
|    1   N/A  N/A   3448099      C   /usr/bin/python                  8732MiB |
|    2   N/A  N/A   3448100      C   /usr/bin/python                  6828MiB |
|    3   N/A  N/A   3448101      C   /usr/bin/python                  7078MiB |
+-----------------------------------------------------------------------------+
```

</p></details>



<details><summary>模型权重输出：</summary><p>


```
tree -h 345m-init-mp
345m-init-mp
├── [4.0K]  iter_0002000
│   ├── [4.0K]  mp_rank_00_000
│   │   └── [1.3G]  model_optim_rng.pt
│   ├── [4.0K]  mp_rank_00_001
│   │   └── [1.3G]  model_optim_rng.pt
│   ├── [4.0K]  mp_rank_01_000
│   │   └── [1.3G]  model_optim_rng.pt
│   └── [4.0K]  mp_rank_01_001
│       └── [1.3G]  model_optim_rng.pt
├── [4.0K]  iter_0004000
│   ├── [4.0K]  mp_rank_00_000
│   │   └── [1.3G]  model_optim_rng.pt
│   ├── [4.0K]  mp_rank_00_001
│   │   └── [1.3G]  model_optim_rng.pt
│   ├── [4.0K]  mp_rank_01_000
│   │   └── [1.3G]  model_optim_rng.pt
│   └── [4.0K]  mp_rank_01_001
│       └── [1.3G]  model_optim_rng.pt
├── [4.0K]  iter_0005000
│   ├── [4.0K]  mp_rank_00_000
│   │   └── [1.3G]  model_optim_rng.pt
│   ├── [4.0K]  mp_rank_00_001
│   │   └── [1.3G]  model_optim_rng.pt
│   ├── [4.0K]  mp_rank_01_000
│   │   └── [1.3G]  model_optim_rng.pt
│   └── [4.0K]  mp_rank_01_001
│       └── [1.3G]  model_optim_rng.pt
└── [   4]  latest_checkpointed_iteration.txt

15 directories, 13 files
```

</p></details>




<details><summary>详细日志输出：</summary><p>

```
> sh examples/pretrain_gpt_distributed_with_mp.sh 
WARNING:torch.distributed.run:
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
*****************************************
using world size: 4, data-parallel-size: 1, tensor-model-parallel size: 2, pipeline-model-parallel size: 2 
using torch.float16 for parameters ...
------------------------ arguments ------------------------
  accumulate_allreduce_grads_in_fp32 .............. False
  adam_beta1 ...................................... 0.9
  adam_beta2 ...................................... 0.999
  adam_eps ........................................ 1e-08
  add_bias_linear ................................. True
  add_position_embedding .......................... True
  adlr_autoresume ................................. False
  adlr_autoresume_interval ........................ 1000
  apply_layernorm_1p .............................. False
  apply_query_key_layer_scaling ................... True
  apply_residual_connection_post_layernorm ........ False
  async_tensor_model_parallel_allreduce ........... False
  attention_dropout ............................... 0.1
  attention_softmax_in_fp32 ....................... False
  barrier_with_L1_time ............................ True
  bert_binary_head ................................ True
  bert_embedder_type .............................. megatron
  bert_load ....................................... None
  bf16 ............................................ False
  bias_dropout_fusion ............................. True
  bias_gelu_fusion ................................ True
  biencoder_projection_dim ........................ 0
  biencoder_shared_query_context_model ............ False
  block_data_path ................................. None
  classes_fraction ................................ 1.0
  clip_grad ....................................... 1.0
  consumed_train_samples .......................... 0
  consumed_valid_samples .......................... 0
  data_cache_path ................................. None
  data_impl ....................................... mmap
  data_parallel_random_init ....................... False
  data_parallel_size .............................. 1
  data_path ....................................... ['/workspace/data/my-gpt2_text_document']
  data_per_class_fraction ......................... 1.0
  data_sharding ................................... True
  dataloader_type ................................. single
  DDP_impl ........................................ local
  decoder_num_layers .............................. None
  decoder_seq_length .............................. None
  dino_bottleneck_size ............................ 256
  dino_freeze_last_layer .......................... 1
  dino_head_hidden_size ........................... 2048
  dino_local_crops_number ......................... 10
  dino_local_img_size ............................. 96
  dino_norm_last_layer ............................ False
  dino_teacher_temp ............................... 0.07
  dino_warmup_teacher_temp ........................ 0.04
  dino_warmup_teacher_temp_epochs ................. 30
  distribute_saved_activations .................... False
  distributed_backend ............................. nccl
  distributed_timeout_minutes ..................... 10
  embedding_path .................................. None
  empty_unused_memory_level ....................... 0
  encoder_num_layers .............................. 24
  encoder_seq_length .............................. 1024
  end_weight_decay ................................ 0.01
  eod_mask_loss ................................... False
  eval_interval ................................... 1000
  eval_iters ...................................... 10
  evidence_data_path .............................. None
  exit_duration_in_mins ........................... None
  exit_interval ................................... None
  exit_on_missing_checkpoint ...................... False
  exit_signal_handler ............................. False
  ffn_hidden_size ................................. 4096
  finetune ........................................ False
  fp16 ............................................ True
  fp16_lm_cross_entropy ........................... False
  fp32_residual_connection ........................ False
  fp8_amax_compute_algo ........................... most_recent
  fp8_amax_history_len ............................ 1
  fp8_e4m3 ........................................ False
  fp8_hybrid ...................................... False
  fp8_interval .................................... 1
  fp8_margin ...................................... 0
  fp8_wgrad ....................................... True
  global_batch_size ............................... 16
  gradient_accumulation_fusion .................... True
  head_lr_mult .................................... 1.0
  hidden_dropout .................................. 0.1
  hidden_size ..................................... 1024
  hysteresis ...................................... 2
  ict_head_size ................................... None
  ict_load ........................................ None
  img_h ........................................... 224
  img_w ........................................... 224
  indexer_batch_size .............................. 128
  indexer_log_interval ............................ 1000
  inference_batch_times_seqlen_threshold .......... 512
  init_method_std ................................. 0.02
  init_method_xavier_uniform ...................... False
  initial_loss_scale .............................. 4294967296
  iter_per_epoch .................................. 1250
  kv_channels ..................................... 64
  layernorm_epsilon ............................... 1e-05
  lazy_mpu_init ................................... None
  load ............................................ /workspace/model/megatron-models/345m-init-mp
  local_rank ...................................... None
  log_batch_size_to_tensorboard ................... False
  log_interval .................................... 100
  log_learning_rate_to_tensorboard ................ True
  log_loss_scale_to_tensorboard ................... True
  log_memory_to_tensorboard ....................... False
  log_num_zeros_in_grad ........................... False
  log_params_norm ................................. False
  log_timers_to_tensorboard ....................... False
  log_validation_ppl_to_tensorboard ............... False
  log_world_size_to_tensorboard ................... False
  loss_scale ...................................... None
  loss_scale_window ............................... 1000
  lr .............................................. 0.00015
  lr_decay_iters .................................. 320000
  lr_decay_samples ................................ None
  lr_decay_style .................................. cosine
  lr_warmup_fraction .............................. 0.01
  lr_warmup_iters ................................. 0
  lr_warmup_samples ............................... 0
  make_vocab_size_divisible_by .................... 128
  mask_factor ..................................... 1.0
  mask_prob ....................................... 0.15
  mask_type ....................................... random
  masked_softmax_fusion ........................... True
  max_position_embeddings ......................... 1024
  max_tokens_to_oom ............................... 12000
  merge_file ...................................... /workspace/model/gpt2-vocab/gpt2-merges.txt
  micro_batch_size ................................ 4
  min_loss_scale .................................. 1.0
  min_lr .......................................... 1e-05
  mmap_warmup ..................................... False
  no_load_optim ................................... None
  no_load_rng ..................................... None
  no_persist_layer_norm ........................... False
  no_save_optim ................................... None
  no_save_rng ..................................... None
  num_attention_heads ............................. 16
  num_channels .................................... 3
  num_classes ..................................... 1000
  num_experts ..................................... None
  num_layers ...................................... 24
  num_layers_per_virtual_pipeline_stage ........... None
  num_workers ..................................... 2
  onnx_safe ....................................... None
  openai_gelu ..................................... False
  optimizer ....................................... adam
  output_bert_embeddings .......................... False
  overlap_p2p_comm ................................ False
  override_opt_param_scheduler .................... False
  params_dtype .................................... torch.float16
  patch_dim ....................................... 16
  perform_initialization .......................... True
  pipeline_model_parallel_size .................... 2
  pipeline_model_parallel_split_rank .............. None
  query_in_block_prob ............................. 0.1
  rampup_batch_size ............................... None
  rank ............................................ 0
  recompute_granularity ........................... None
  recompute_method ................................ None
  recompute_num_layers ............................ 1
  reset_attention_mask ............................ False
  reset_position_ids .............................. False
  retriever_report_topk_accuracies ................ []
  retriever_score_scaling ......................... False
  retriever_seq_length ............................ 256
  retro_add_retriever ............................. False
  retro_cyclic_train_iters ........................ None
  retro_encoder_attention_dropout ................. 0.1
  retro_encoder_hidden_dropout .................... 0.1
  retro_encoder_layers ............................ 2
  retro_num_neighbors ............................. 2
  retro_num_retrieved_chunks ...................... 2
  retro_return_doc_ids ............................ False
  retro_workdir ................................... None
  rotary_percent .................................. 1.0
  sample_rate ..................................... 1.0
  save ............................................ /workspace/model/megatron-models/345m-init-mp
  save_interval ................................... 2000
  scatter_gather_tensors_in_pipeline .............. True
  seed ............................................ 1234
  seq_length ...................................... 1024
  sequence_parallel ............................... True
  sgd_momentum .................................... 0.9
  short_seq_prob .................................. 0.1
  split ........................................... 700,200,100
  squared_relu .................................... False
  standalone_embedding_stage ...................... False
  start_weight_decay .............................. 0.01
  swiglu .......................................... False
  swin_backbone_type .............................. tiny
  tensor_model_parallel_size ...................... 2
  tensorboard_dir ................................. None
  tensorboard_log_interval ........................ 1
  tensorboard_queue_size .......................... 1000
  test_data_path .................................. None
  timing_log_level ................................ 0
  timing_log_option ............................... minmax
  titles_data_path ................................ None
  tokenizer_model ................................. None
  tokenizer_type .................................. GPT2BPETokenizer
  train_data_path ................................. None
  train_iters ..................................... 5000
  train_samples ................................... None
  transformer_impl ................................ local
  transformer_pipeline_model_parallel_size ........ 2
  untie_embeddings_and_output_weights ............. False
  use_checkpoint_args ............................. False
  use_checkpoint_opt_param_scheduler .............. False
  use_contiguous_buffers_in_local_ddp ............. True
  use_cpu_initialization .......................... None
  use_distributed_optimizer ....................... False
  use_flash_attn .................................. False
  use_one_sent_docs ............................... False
  use_ring_exchange_p2p ........................... False
  use_rotary_position_embeddings .................. False
  valid_data_path ................................. None
  variable_seq_lengths ............................ False
  virtual_pipeline_model_parallel_size ............ None
  vision_backbone_type ............................ vit
  vision_pretraining .............................. False
  vision_pretraining_type ......................... classify
  vocab_extra_ids ................................. 0
  vocab_file ...................................... /workspace/model/gpt2-vocab/gpt2-vocab.json
  vocab_size ...................................... None
  weight_decay .................................... 0.01
  weight_decay_incr_style ......................... constant
  world_size ...................................... 4
-------------------- end of arguments ---------------------
setting number of micro-batches to constant 4
> building GPT2BPETokenizer tokenizer ...
 > padded vocab (size: 50257) with 175 dummy tokens (new size: 50432)
> initializing torch distributed ...
> initialized tensor model parallel with size 2
> initialized pipeline model parallel with size 2
> setting random seeds to 1234 ...
> compiling dataset index builder ...
make: Entering directory '/workspace/code/Megatron-LM/megatron/data'
make: Nothing to be done for 'default'.
make: Leaving directory '/workspace/code/Megatron-LM/megatron/data'
>>> done with dataset index builder. Compilation time: 0.209 seconds
> compiling and loading fused kernels ...
Detected CUDA files, patching ldflags
Emitting ninja build file /workspace/code/Megatron-LM/megatron/fused_kernels/build/build.ninja...
Building extension module scaled_upper_triang_masked_softmax_cuda...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
ninja: no work to do.
Loading extension module scaled_upper_triang_masked_softmax_cuda...
Detected CUDA files, patching ldflags
Emitting ninja build file /workspace/code/Megatron-LM/megatron/fused_kernels/build/build.ninja...
Building extension module scaled_masked_softmax_cuda...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
ninja: no work to do.
Loading extension module scaled_masked_softmax_cuda...
Detected CUDA files, patching ldflags
Emitting ninja build file /workspace/code/Megatron-LM/megatron/fused_kernels/build/build.ninja...
Building extension module scaled_softmax_cuda...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
ninja: no work to do.
Loading extension module scaled_softmax_cuda...
>>> done with compiling and loading fused kernels. Compilation time: 4.125 seconds
time to initialize megatron (seconds): 7.305
[after megatron is initialized] datetime: 2023-07-16 12:25:38 
building GPT model ...
 > number of parameters on (tensor, pipeline) model parallel rank (1, 0): 102483968
 > number of parameters on (tensor, pipeline) model parallel rank (1, 1): 101437440
 > number of parameters on (tensor, pipeline) model parallel rank (0, 0): 102483968
 > number of parameters on (tensor, pipeline) model parallel rank (0, 1): 101437440
> learning rate decay style: cosine
WARNING: could not find the metadata file /workspace/model/megatron-models/345m-init-mp/latest_checkpointed_iteration.txt 
    will not load any checkpoints and will start from random
/usr/local/lib/python3.8/dist-packages/torch/distributed/distributed_c10d.py:2603: UserWarning: torch.distributed._all_gather_base is a private function and will be deprecated. Please use torch.distributed.all_gather_into_tensor instead.
  warnings.warn(
/usr/local/lib/python3.8/dist-packages/torch/distributed/distributed_c10d.py:2603: UserWarning: torch.distributed._all_gather_base is a private function and will be deprecated. Please use torch.distributed.all_gather_into_tensor instead.
  warnings.warn(
/usr/local/lib/python3.8/dist-packages/torch/distributed/distributed_c10d.py:2603: UserWarning: torch.distributed._all_gather_base is a private function and will be deprecated. Please use torch.distributed.all_gather_into_tensor instead.
  warnings.warn(
/usr/local/lib/python3.8/dist-packages/torch/distributed/distributed_c10d.py:2603: UserWarning: torch.distributed._all_gather_base is a private function and will be deprecated. Please use torch.distributed.all_gather_into_tensor instead.
  warnings.warn(
(min, max) time across ranks (ms):
    load-checkpoint ................................: (6.90, 7.03)
[after model, optimizer, and learning rate scheduler are built] datetime: 2023-07-16 12:25:39 
> building train, validation, and test datasets ...
 > datasets target sizes (minimum size):
    train:      80000
    validation: 960
    test:       160
> building train, validation, and test datasets for GPT ...
Single data path provided for train, valid & test
data_prefix: ['/workspace/data/my-gpt2_text_document']
 > building dataset index ...
    reading sizes...
    reading pointers...
    reading document index...
    creating numpy buffer of mmap...
    creating memory view of numpy buffer...
 > finished creating indexed dataset in 0.003472 seconds
    number of documents: 2456
 > dataset split:
    train:
     document indices in [0, 1719) total of 1719 documents
    validation:
     document indices in [1719, 2210) total of 491 documents
    test:
     document indices in [2210, 2456) total of 246 documents
 > WARNING: could not find index map files, building the indices on rank 0 ...
 > last epoch number of samples (335) is smaller than 80% of number of samples per epoch (452), setting separate_last_epoch to True
 > elasped time to build and save doc-idx mapping (seconds): 0.026260
    using:
     number of documents:       1719
     number of epochs:          177
     sequence length:           1024
     total number of samples:   80118
 > elasped time to build and save sample-idx mapping (seconds): 0.010997
 > building shuffle index with split [0, 79665) and [79665, 80118) ...
 > elasped time to build and save shuffle-idx mapping (seconds): 0.008945
 > loading doc-idx mapping from /workspace/data/index-cache/3780b1a26d534c9d9a1da4ab04f13381_doc_idx.npy
 > loading sample-idx mapping from /workspace/data/index-cache/3780b1a26d534c9d9a1da4ab04f13381_sample_idx.npy
 > loading shuffle-idx mapping from /workspace/data/index-cache/3780b1a26d534c9d9a1da4ab04f13381_shuffle_idx.npy
    loaded indexed file in 0.008 seconds
    total number of samples: 80119
    total number of epochs: 177
 > WARNING: could not find index map files, building the indices on rank 0 ...
 > last epoch number of samples (74) is smaller than 80% of number of samples per epoch (110), setting separate_last_epoch to True
 > elasped time to build and save doc-idx mapping (seconds): 0.005155
    using:
     number of documents:       491
     number of epochs:          9
     sequence length:           1024
     total number of samples:   997
 > elasped time to build and save sample-idx mapping (seconds): 0.004258
 > building shuffle index with split [0, 886) and [886, 997) ...
 > elasped time to build and save shuffle-idx mapping (seconds): 0.005074
 > loading doc-idx mapping from /workspace/data/index-cache/ae23c587dd8002d7491a754ac381df77_doc_idx.npy
 > loading sample-idx mapping from /workspace/data/index-cache/ae23c587dd8002d7491a754ac381df77_sample_idx.npy
 > loading shuffle-idx mapping from /workspace/data/index-cache/ae23c587dd8002d7491a754ac381df77_shuffle_idx.npy
    loaded indexed file in 0.005 seconds
    total number of samples: 998
    total number of epochs: 9
 > loading doc-idx mapping from /workspace/data/index-cache/5eb65247febbee0dc5f98a246cfebfae_doc_idx.npy
 > loading sample-idx mapping from /workspace/data/index-cache/5eb65247febbee0dc5f98a246cfebfae_sample_idx.npy
 > loading shuffle-idx mapping from /workspace/data/index-cache/5eb65247febbee0dc5f98a246cfebfae_shuffle_idx.npy
    loaded indexed file in 0.003 seconds
    total number of samples: 161
    total number of epochs: 3
> finished creating GPT datasets ...
[after dataloaders are built] datetime: 2023-07-16 12:25:43 
done with setup ...
training ...
(min, max) time across ranks (ms):
    model-and-optimizer-setup ......................: (593.04, 601.19)
    train/valid/test-data-iterators-setup ..........: (4026.38, 4324.57)
[before the start of training step] datetime: 2023-07-16 12:25:43 
[W ProcessGroupNCCL.cpp:1692] Warning: 0NCCL_AVOID_RECORD_STREAMS=1 has no effect for point-to-point collectives. (function operator())
[W ProcessGroupNCCL.cpp:1692] Warning: 0NCCL_AVOID_RECORD_STREAMS=1 has no effect for point-to-point collectives. (function operator())
/usr/local/lib/python3.8/dist-packages/torch/distributed/distributed_c10d.py:3071: UserWarning: torch.distributed._reduce_scatter_base is a private function and will be deprecated. Please use torch.distributed.reduce_scatter_tensor instead.
  warnings.warn(
/usr/local/lib/python3.8/dist-packages/torch/distributed/distributed_c10d.py:3071: UserWarning: torch.distributed._reduce_scatter_base is a private function and will be deprecated. Please use torch.distributed.reduce_scatter_tensor instead.
  warnings.warn(
[W ProcessGroupNCCL.cpp:1692] Warning: 0NCCL_AVOID_RECORD_STREAMS=1 has no effect for point-to-point collectives. (function operator())
[W ProcessGroupNCCL.cpp:1692] Warning: 0NCCL_AVOID_RECORD_STREAMS=1 has no effect for point-to-point collectives. (function operator())
[W ProcessGroupNCCL.cpp:1692] Warning: 0NCCL_AVOID_RECORD_STREAMS=1 has no effect for point-to-point collectives. (function operator())
[W ProcessGroupNCCL.cpp:1692] Warning: 0NCCL_AVOID_RECORD_STREAMS=1 has no effect for point-to-point collectives. (function operator())
/usr/local/lib/python3.8/dist-packages/torch/distributed/distributed_c10d.py:3071: UserWarning: torch.distributed._reduce_scatter_base is a private function and will be deprecated. Please use torch.distributed.reduce_scatter_tensor instead.
  warnings.warn(
/usr/local/lib/python3.8/dist-packages/torch/distributed/distributed_c10d.py:3071: UserWarning: torch.distributed._reduce_scatter_base is a private function and will be deprecated. Please use torch.distributed.reduce_scatter_tensor instead.
  warnings.warn(
[W ProcessGroupNCCL.cpp:1692] Warning: 0NCCL_AVOID_RECORD_STREAMS=1 has no effect for point-to-point collectives. (function operator())
[W ProcessGroupNCCL.cpp:1692] Warning: 0NCCL_AVOID_RECORD_STREAMS=1 has no effect for point-to-point collectives. (function operator())
 iteration      100/    5000 | consumed samples:         1600 | elapsed time per iteration (ms): 858.6 | learning rate: 3.984E-06 | global batch size:    16 | lm loss: 9.674668E+00 | loss scale: 262144.0 | grad norm: 3.750 | number of skipped iterations:  15 | number of nan iterations:   0 |
[Rank 3] (after 100 iterations) memory (MB) | allocated: 2029.29443359375 | max allocated: 5373.49560546875 | reserved: 5968.0 | max reserved: 5968.0
[Rank 2] (after 100 iterations) memory (MB) | allocated: 2029.29443359375 | max allocated: 5373.49560546875 | reserved: 5718.0 | max reserved: 5718.0
[Rank 1] (after 100 iterations) memory (MB) | allocated: 2040.25830078125 | max allocated: 7253.22705078125 | reserved: 7622.0 | max reserved: 7622.0
[Rank 0] (after 100 iterations) memory (MB) | allocated: 2040.25830078125 | max allocated: 7253.22705078125 | reserved: 7622.0 | max reserved: 7622.0
 iteration      200/    5000 | consumed samples:         3200 | elapsed time per iteration (ms): 1082.6 | learning rate: 8.672E-06 | global batch size:    16 | lm loss: 8.440352E+00 | loss scale: 262144.0 | grad norm: 3.089 | number of skipped iterations:   0 | number of nan iterations:   0 |
...
 iteration     1000/    5000 | consumed samples:        16000 | elapsed time per iteration (ms): 1109.8 | learning rate: 4.617E-05 | global batch size:    16 | lm loss: 5.200860E+00 | loss scale: 262144.0 | grad norm: 1.947 | number of skipped iterations:   0 | number of nan iterations:   0 |
------------------------------------------------------------------------------------------------
 validation loss at iteration 1000 | lm loss value: 6.329505E+00 | lm loss PPL: 5.608791E+02 | 
------------------------------------------------------------------------------------------------
 iteration     1100/    5000 | consumed samples:        17600 | elapsed time per iteration (ms): 839.6 | learning rate: 5.086E-05 | global batch size:    16 | lm loss: 5.015918E+00 | loss scale: 524288.0 | grad norm: 1.851 | number of skipped iterations:   0 | number of nan iterations:   0 |
...
 iteration     2000/    5000 | consumed samples:        32000 | elapsed time per iteration (ms): 1031.7 | learning rate: 9.295E-05 | global batch size:    16 | lm loss: 2.117708E+00 | loss scale: 262144.0 | grad norm: 3.088 | number of skipped iterations:   0 | number of nan iterations:   0 |
------------------------------------------------------------------------------------------------
 validation loss at iteration 2000 | lm loss value: 6.586888E+00 | lm loss PPL: 7.255198E+02 | 
------------------------------------------------------------------------------------------------
saving checkpoint at iteration    2000 to /workspace/model/megatron-models/345m-init-mp
...
```

</p></details>





### 模型并行（4TP）


显存占用：
```
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A   3895346      C   /usr/bin/python                  4236MiB |
|    1   N/A  N/A   3895347      C   /usr/bin/python                  4176MiB |
|    2   N/A  N/A   3895348      C   /usr/bin/python                  4168MiB |
|    3   N/A  N/A   3895349      C   /usr/bin/python                  4176MiB |
+-----------------------------------------------------------------------------+
```


<details><summary>详细输出</summary><p>


```
> sh pretrain_gpt_distributed_with_4tp.sh
WARNING:torch.distributed.run:
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
*****************************************
using world size: 4, data-parallel-size: 1, tensor-model-parallel size: 4, pipeline-model-parallel size: 1 
using torch.float16 for parameters ...
------------------------ arguments ------------------------
  accumulate_allreduce_grads_in_fp32 .............. False
  adam_beta1 ...................................... 0.9
  adam_beta2 ...................................... 0.999
  adam_eps ........................................ 1e-08
  add_bias_linear ................................. True
  add_position_embedding .......................... True
  adlr_autoresume ................................. False
  adlr_autoresume_interval ........................ 1000
  apply_layernorm_1p .............................. False
  apply_query_key_layer_scaling ................... True
  apply_residual_connection_post_layernorm ........ False
  async_tensor_model_parallel_allreduce ........... False
  attention_dropout ............................... 0.1
  attention_softmax_in_fp32 ....................... False
  barrier_with_L1_time ............................ True
  bert_binary_head ................................ True
  bert_embedder_type .............................. megatron
  bert_load ....................................... None
  bf16 ............................................ False
  bias_dropout_fusion ............................. True
  bias_gelu_fusion ................................ True
  biencoder_projection_dim ........................ 0
  biencoder_shared_query_context_model ............ False
  block_data_path ................................. None
  classes_fraction ................................ 1.0
  clip_grad ....................................... 1.0
  consumed_train_samples .......................... 0
  consumed_valid_samples .......................... 0
  data_cache_path ................................. None
  data_impl ....................................... mmap
  data_parallel_random_init ....................... False
  data_parallel_size .............................. 1
  data_path ....................................... ['/workspace/data/my-gpt2_text_document']
  data_per_class_fraction ......................... 1.0
  data_sharding ................................... True
  dataloader_type ................................. single
  DDP_impl ........................................ local
  decoder_num_layers .............................. None
  decoder_seq_length .............................. None
  dino_bottleneck_size ............................ 256
  dino_freeze_last_layer .......................... 1
  dino_head_hidden_size ........................... 2048
  dino_local_crops_number ......................... 10
  dino_local_img_size ............................. 96
  dino_norm_last_layer ............................ False
  dino_teacher_temp ............................... 0.07
  dino_warmup_teacher_temp ........................ 0.04
  dino_warmup_teacher_temp_epochs ................. 30
  distribute_saved_activations .................... False
  distributed_backend ............................. nccl
  distributed_timeout_minutes ..................... 10
  embedding_path .................................. None
  empty_unused_memory_level ....................... 0
  encoder_num_layers .............................. 24
  encoder_seq_length .............................. 1024
  end_weight_decay ................................ 0.01
  eod_mask_loss ................................... False
  eval_interval ................................... 1000
  eval_iters ...................................... 10
  evidence_data_path .............................. None
  exit_duration_in_mins ........................... None
  exit_interval ................................... None
  exit_on_missing_checkpoint ...................... False
  exit_signal_handler ............................. False
  ffn_hidden_size ................................. 4096
  finetune ........................................ False
  fp16 ............................................ True
  fp16_lm_cross_entropy ........................... False
  fp32_residual_connection ........................ False
  fp8_amax_compute_algo ........................... most_recent
  fp8_amax_history_len ............................ 1
  fp8_e4m3 ........................................ False
  fp8_hybrid ...................................... False
  fp8_interval .................................... 1
  fp8_margin ...................................... 0
  fp8_wgrad ....................................... True
  global_batch_size ............................... 8
  gradient_accumulation_fusion .................... True
  head_lr_mult .................................... 1.0
  hidden_dropout .................................. 0.1
  hidden_size ..................................... 1024
  hysteresis ...................................... 2
  ict_head_size ................................... None
  ict_load ........................................ None
  img_h ........................................... 224
  img_w ........................................... 224
  indexer_batch_size .............................. 128
  indexer_log_interval ............................ 1000
  inference_batch_times_seqlen_threshold .......... 512
  init_method_std ................................. 0.02
  init_method_xavier_uniform ...................... False
  initial_loss_scale .............................. 4294967296
  iter_per_epoch .................................. 1250
  kv_channels ..................................... 64
  layernorm_epsilon ............................... 1e-05
  lazy_mpu_init ................................... None
  load ............................................ /workspace/model/megatron-models/345m-init-4tp
  local_rank ...................................... None
  log_batch_size_to_tensorboard ................... False
  log_interval .................................... 100
  log_learning_rate_to_tensorboard ................ True
  log_loss_scale_to_tensorboard ................... True
  log_memory_to_tensorboard ....................... False
  log_num_zeros_in_grad ........................... False
  log_params_norm ................................. False
  log_timers_to_tensorboard ....................... False
  log_validation_ppl_to_tensorboard ............... False
  log_world_size_to_tensorboard ................... False
  loss_scale ...................................... None
  loss_scale_window ............................... 1000
  lr .............................................. 0.00015
  lr_decay_iters .................................. 320000
  lr_decay_samples ................................ None
  lr_decay_style .................................. cosine
  lr_warmup_fraction .............................. 0.01
  lr_warmup_iters ................................. 0
  lr_warmup_samples ............................... 0
  make_vocab_size_divisible_by .................... 128
  mask_factor ..................................... 1.0
  mask_prob ....................................... 0.15
  mask_type ....................................... random
  masked_softmax_fusion ........................... True
  max_position_embeddings ......................... 1024
  max_tokens_to_oom ............................... 12000
  merge_file ...................................... /workspace/model/gpt2-vocab/gpt2-merges.txt
  micro_batch_size ................................ 2
  min_loss_scale .................................. 1.0
  min_lr .......................................... 1e-05
  mmap_warmup ..................................... False
  no_load_optim ................................... None
  no_load_rng ..................................... None
  no_persist_layer_norm ........................... False
  no_save_optim ................................... None
  no_save_rng ..................................... None
  num_attention_heads ............................. 16
  num_channels .................................... 3
  num_classes ..................................... 1000
  num_experts ..................................... None
  num_layers ...................................... 24
  num_layers_per_virtual_pipeline_stage ........... None
  num_workers ..................................... 2
  onnx_safe ....................................... None
  openai_gelu ..................................... False
  optimizer ....................................... adam
  output_bert_embeddings .......................... False
  overlap_p2p_comm ................................ False
  override_opt_param_scheduler .................... False
  params_dtype .................................... torch.float16
  patch_dim ....................................... 16
  perform_initialization .......................... True
  pipeline_model_parallel_size .................... 1
  pipeline_model_parallel_split_rank .............. None
  query_in_block_prob ............................. 0.1
  rampup_batch_size ............................... None
  rank ............................................ 0
  recompute_granularity ........................... None
  recompute_method ................................ None
  recompute_num_layers ............................ 1
  reset_attention_mask ............................ False
  reset_position_ids .............................. False
  retriever_report_topk_accuracies ................ []
  retriever_score_scaling ......................... False
  retriever_seq_length ............................ 256
  retro_add_retriever ............................. False
  retro_cyclic_train_iters ........................ None
  retro_encoder_attention_dropout ................. 0.1
  retro_encoder_hidden_dropout .................... 0.1
  retro_encoder_layers ............................ 2
  retro_num_neighbors ............................. 2
  retro_num_retrieved_chunks ...................... 2
  retro_return_doc_ids ............................ False
  retro_workdir ................................... None
  rotary_percent .................................. 1.0
  sample_rate ..................................... 1.0
  save ............................................ /workspace/model/megatron-models/345m-init-4tp
  save_interval ................................... 1000
  scatter_gather_tensors_in_pipeline .............. True
  seed ............................................ 1234
  seq_length ...................................... 1024
  sequence_parallel ............................... True
  sgd_momentum .................................... 0.9
  short_seq_prob .................................. 0.1
  split ........................................... 700,200,100
  squared_relu .................................... False
  standalone_embedding_stage ...................... False
  start_weight_decay .............................. 0.01
  swiglu .......................................... False
  swin_backbone_type .............................. tiny
  tensor_model_parallel_size ...................... 4
  tensorboard_dir ................................. None
  tensorboard_log_interval ........................ 1
  tensorboard_queue_size .......................... 1000
  test_data_path .................................. None
  timing_log_level ................................ 0
  timing_log_option ............................... minmax
  titles_data_path ................................ None
  tokenizer_model ................................. None
  tokenizer_type .................................. GPT2BPETokenizer
  train_data_path ................................. None
  train_iters ..................................... 2000
  train_samples ................................... None
  transformer_impl ................................ local
  transformer_pipeline_model_parallel_size ........ 1
  untie_embeddings_and_output_weights ............. False
  use_checkpoint_args ............................. False
  use_checkpoint_opt_param_scheduler .............. False
  use_contiguous_buffers_in_local_ddp ............. True
  use_cpu_initialization .......................... None
  use_distributed_optimizer ....................... False
  use_flash_attn .................................. False
  use_one_sent_docs ............................... False
  use_ring_exchange_p2p ........................... False
  use_rotary_position_embeddings .................. False
  valid_data_path ................................. None
  variable_seq_lengths ............................ False
  virtual_pipeline_model_parallel_size ............ None
  vision_backbone_type ............................ vit
  vision_pretraining .............................. False
  vision_pretraining_type ......................... classify
  vocab_extra_ids ................................. 0
  vocab_file ...................................... /workspace/model/gpt2-vocab/gpt2-vocab.json
  vocab_size ...................................... None
  weight_decay .................................... 0.01
  weight_decay_incr_style ......................... constant
  world_size ...................................... 4
-------------------- end of arguments ---------------------
setting number of micro-batches to constant 4
> building GPT2BPETokenizer tokenizer ...
 > padded vocab (size: 50257) with 431 dummy tokens (new size: 50688)
> initializing torch distributed ...
> initialized tensor model parallel with size 4
> initialized pipeline model parallel with size 1
> setting random seeds to 1234 ...
> compiling dataset index builder ...
make: Entering directory '/workspace/code/Megatron-LM/megatron/data'
make: Nothing to be done for 'default'.
make: Leaving directory '/workspace/code/Megatron-LM/megatron/data'
>>> done with dataset index builder. Compilation time: 0.198 seconds
> compiling and loading fused kernels ...
Detected CUDA files, patching ldflags
Emitting ninja build file /workspace/code/Megatron-LM/megatron/fused_kernels/build/build.ninja...
Building extension module scaled_upper_triang_masked_softmax_cuda...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
ninja: no work to do.
Loading extension module scaled_upper_triang_masked_softmax_cuda...
Detected CUDA files, patching ldflags
Emitting ninja build file /workspace/code/Megatron-LM/megatron/fused_kernels/build/build.ninja...
Building extension module scaled_masked_softmax_cuda...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
ninja: no work to do.
Loading extension module scaled_masked_softmax_cuda...
Detected CUDA files, patching ldflags
Emitting ninja build file /workspace/code/Megatron-LM/megatron/fused_kernels/build/build.ninja...
Building extension module scaled_softmax_cuda...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
ninja: no work to do.
Loading extension module scaled_softmax_cuda...
>>> done with compiling and loading fused kernels. Compilation time: 3.127 seconds
time to initialize megatron (seconds): 5.792
[after megatron is initialized] datetime: 2023-07-16 14:05:13 
building GPT model ...
 > number of parameters on (tensor, pipeline) model parallel rank (2, 0): 89714688
 > number of parameters on (tensor, pipeline) model parallel rank (3, 0): 89714688
 > number of parameters on (tensor, pipeline) model parallel rank (0, 0): 89714688
 > number of parameters on (tensor, pipeline) model parallel rank (1, 0): 89714688
> learning rate decay style: cosine
WARNING: could not find the metadata file /workspace/model/megatron-models/345m-init-4tp/latest_checkpointed_iteration.txt 
    will not load any checkpoints and will start from random
/usr/local/lib/python3.8/dist-packages/torch/distributed/distributed_c10d.py:2603: UserWarning: torch.distributed._all_gather_base is a private function and will be deprecated. Please use torch.distributed.all_gather_into_tensor instead.
  warnings.warn(
/usr/local/lib/python3.8/dist-packages/torch/distributed/distributed_c10d.py:2603: UserWarning: torch.distributed._all_gather_base is a private function and will be deprecated. Please use torch.distributed.all_gather_into_tensor instead.
  warnings.warn(
/usr/local/lib/python3.8/dist-packages/torch/distributed/distributed_c10d.py:2603: UserWarning: torch.distributed._all_gather_base is a private function and will be deprecated. Please use torch.distributed.all_gather_into_tensor instead.
  warnings.warn(
/usr/local/lib/python3.8/dist-packages/torch/distributed/distributed_c10d.py:2603: UserWarning: torch.distributed._all_gather_base is a private function and will be deprecated. Please use torch.distributed.all_gather_into_tensor instead.
  warnings.warn(
(min, max) time across ranks (ms):
    load-checkpoint ................................: (6.91, 7.08)
[after model, optimizer, and learning rate scheduler are built] datetime: 2023-07-16 14:05:14 
> building train, validation, and test datasets ...
 > datasets target sizes (minimum size):
    train:      16000
    validation: 240
    test:       80
> building train, validation, and test datasets for GPT ...
Single data path provided for train, valid & test
data_prefix: ['/workspace/data/my-gpt2_text_document']
 > building dataset index ...
    reading sizes...
    reading pointers...
    reading document index...
    creating numpy buffer of mmap...
    creating memory view of numpy buffer...
 > finished creating indexed dataset in 0.002060 seconds
    number of documents: 2456
 > dataset split:
    train:
     document indices in [0, 1719) total of 1719 documents
    validation:
     document indices in [1719, 2210) total of 491 documents
    test:
     document indices in [2210, 2456) total of 246 documents
 > WARNING: could not find index map files, building the indices on rank 0 ...
 > last epoch number of samples (158) is smaller than 80% of number of samples per epoch (452), setting separate_last_epoch to True
 > elasped time to build and save doc-idx mapping (seconds): 0.008389
    using:
     number of documents:       1719
     number of epochs:          36
     sequence length:           1024
     total number of samples:   16295
 > elasped time to build and save sample-idx mapping (seconds): 0.005391
 > building shuffle index with split [0, 15842) and [15842, 16295) ...
 > elasped time to build and save shuffle-idx mapping (seconds): 0.004106
 > loading doc-idx mapping from /workspace/data/index-cache/09a6ab12e4c21658eefabc732b2e110b_doc_idx.npy
 > loading sample-idx mapping from /workspace/data/index-cache/09a6ab12e4c21658eefabc732b2e110b_sample_idx.npy
 > loading shuffle-idx mapping from /workspace/data/index-cache/09a6ab12e4c21658eefabc732b2e110b_shuffle_idx.npy
    loaded indexed file in 0.007 seconds
    total number of samples: 16296
    total number of epochs: 36
 > WARNING: could not find index map files, building the indices on rank 0 ...
 > last epoch number of samples (19) is smaller than 80% of number of samples per epoch (110), setting separate_last_epoch to True
 > elasped time to build and save doc-idx mapping (seconds): 0.003765
    using:
     number of documents:       491
     number of epochs:          3
     sequence length:           1024
     total number of samples:   332
 > elasped time to build and save sample-idx mapping (seconds): 0.003097
 > building shuffle index with split [0, 221) and [221, 332) ...
 > elasped time to build and save shuffle-idx mapping (seconds): 0.002940
 > loading doc-idx mapping from /workspace/data/index-cache/2d16c1014278a21da8aa7bdfcf0f1e3f_doc_idx.npy
 > loading sample-idx mapping from /workspace/data/index-cache/2d16c1014278a21da8aa7bdfcf0f1e3f_sample_idx.npy
 > loading shuffle-idx mapping from /workspace/data/index-cache/2d16c1014278a21da8aa7bdfcf0f1e3f_shuffle_idx.npy
    loaded indexed file in 0.004 seconds
    total number of samples: 333
    total number of epochs: 3
 > loading doc-idx mapping from /workspace/data/index-cache/f1583c0457c0ee9c3ecb3c068b3badd8_doc_idx.npy
 > loading sample-idx mapping from /workspace/data/index-cache/f1583c0457c0ee9c3ecb3c068b3badd8_sample_idx.npy
 > loading shuffle-idx mapping from /workspace/data/index-cache/f1583c0457c0ee9c3ecb3c068b3badd8_shuffle_idx.npy
    loaded indexed file in 0.003 seconds
    total number of samples: 108
    total number of epochs: 2
> finished creating GPT datasets ...
[after dataloaders are built] datetime: 2023-07-16 14:05:15 
done with setup ...
training ...
(min, max) time across ranks (ms):
    model-and-optimizer-setup ......................: (480.17, 494.89)
    train/valid/test-data-iterators-setup ..........: (965.72, 1189.60)
[before the start of training step] datetime: 2023-07-16 14:05:15 
/usr/local/lib/python3.8/dist-packages/torch/distributed/distributed_c10d.py:3071: UserWarning: torch.distributed._reduce_scatter_base is a private function and will be deprecated. Please use torch.distributed.reduce_scatter_tensor instead.
  warnings.warn(
/usr/local/lib/python3.8/dist-packages/torch/distributed/distributed_c10d.py:3071: UserWarning: torch.distributed._reduce_scatter_base is a private function and will be deprecated. Please use torch.distributed.reduce_scatter_tensor instead.
  warnings.warn(
/usr/local/lib/python3.8/dist-packages/torch/distributed/distributed_c10d.py:3071: UserWarning: torch.distributed._reduce_scatter_base is a private function and will be deprecated. Please use torch.distributed.reduce_scatter_tensor instead.
  warnings.warn(
/usr/local/lib/python3.8/dist-packages/torch/distributed/distributed_c10d.py:3071: UserWarning: torch.distributed._reduce_scatter_base is a private function and will be deprecated. Please use torch.distributed.reduce_scatter_tensor instead.
  warnings.warn(


 iteration      100/    2000 | consumed samples:          800 | elapsed time per iteration (ms): 2946.0 | learning rate: 3.984E-06 | global batch size:     8 | lm loss: 9.654853E+00 | loss scale: 262144.0 | grad norm: 3.700 | number of skipped iterations:  15 | number of nan iterations:   0 |
[Rank 0] (after 100 iterations) memory (MB) | allocated: 1772.68505859375 | max allocated: 3059.67138671875 | reserved: 3394.0 | max reserved: 3394.0
[Rank 2] (after 100 iterations) memory (MB) | allocated: 1794.56005859375 | max allocated: 3081.17138671875 | reserved: 3390.0 | max reserved: 3390.0
[Rank 3] (after 100 iterations) memory (MB) | allocated: 1795.06005859375 | max allocated: 3081.67138671875 | reserved: 3398.0 | max reserved: 3398.0
[Rank 1] (after 100 iterations) memory (MB) | allocated: 1774.18505859375 | max allocated: 3065.67138671875 | reserved: 3398.0 | max reserved: 3398.0

 iteration      200/    2000 | consumed samples:         1600 | elapsed time per iteration (ms): 2435.9 | learning rate: 8.672E-06 | global batch size:     8 | lm loss: 8.503224E+00 | loss scale: 262144.0 | grad norm: 3.291 | number of skipped iterations:   0 | number of nan iterations:   0 |

 iteration      300/    2000 | consumed samples:         2400 | elapsed time per iteration (ms): 2710.7 | learning rate: 1.331E-05 | global batch size:     8 | lm loss: 7.670080E+00 | loss scale: 131072.0 | grad norm: 3.704 | number of skipped iterations:   1 | number of nan iterations:   0 |
 iteration      400/    2000 | consumed samples:         3200 | elapsed time per iteration (ms): 2848.5 | learning rate: 1.800E-05 | global batch size:     8 | lm loss: 6.967275E+00 | loss scale: 131072.0 | grad norm: 2.357 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration      500/    2000 | consumed samples:         4000 | elapsed time per iteration (ms): 2842.7 | learning rate: 2.269E-05 | global batch size:     8 | lm loss: 6.564229E+00 | loss scale: 131072.0 | grad norm: 2.356 | number of skipped iterations:   0 | number of nan iterations:   0 |

 iteration      600/    2000 | consumed samples:         4800 | elapsed time per iteration (ms): 2519.6 | learning rate: 2.737E-05 | global batch size:     8 | lm loss: 6.246951E+00 | loss scale: 131072.0 | grad norm: 2.799 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration      700/    2000 | consumed samples:         5600 | elapsed time per iteration (ms): 2552.9 | learning rate: 3.206E-05 | global batch size:     8 | lm loss: 6.112354E+00 | loss scale: 131072.0 | grad norm: 1.943 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration      800/    2000 | consumed samples:         6400 | elapsed time per iteration (ms): 2701.1 | learning rate: 3.675E-05 | global batch size:     8 | lm loss: 5.886550E+00 | loss scale: 131072.0 | grad norm: 2.176 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration      900/    2000 | consumed samples:         7200 | elapsed time per iteration (ms): 2993.7 | learning rate: 4.144E-05 | global batch size:     8 | lm loss: 5.670854E+00 | loss scale: 131072.0 | grad norm: 2.356 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration     1000/    2000 | consumed samples:         8000 | elapsed time per iteration (ms): 2642.9 | learning rate: 4.612E-05 | global batch size:     8 | lm loss: 5.564718E+00 | loss scale: 131072.0 | grad norm: 2.031 | number of skipped iterations:   0 | number of nan iterations:   0 |
------------------------------------------------------------------------------------------------
 validation loss at iteration 1000 | lm loss value: 6.319978E+00 | lm loss PPL: 5.555609E+02 | 
------------------------------------------------------------------------------------------------
saving checkpoint at iteration    1000 to /workspace/model/megatron-models/345m-init-4tp
  successfully saved checkpoint at iteration    1000 to /workspace/model/megatron-models/345m-init-4tp
(min, max) time across ranks (ms):
    save-checkpoint ................................: (33544.92, 33545.12)
 iteration     1100/    2000 | consumed samples:         8800 | elapsed time per iteration (ms): 3176.9 | learning rate: 5.081E-05 | global batch size:     8 | lm loss: 5.373973E+00 | loss scale: 131072.0 | grad norm: 1.966 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration     1200/    2000 | consumed samples:         9600 | elapsed time per iteration (ms): 2297.0 | learning rate: 5.550E-05 | global batch size:     8 | lm loss: 5.235421E+00 | loss scale: 131072.0 | grad norm: 1.634 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration     1300/    2000 | consumed samples:        10400 | elapsed time per iteration (ms): 2666.7 | learning rate: 6.019E-05 | global batch size:     8 | lm loss: 5.050657E+00 | loss scale: 262144.0 | grad norm: 2.018 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration     1400/    2000 | consumed samples:        11200 | elapsed time per iteration (ms): 2774.4 | learning rate: 6.487E-05 | global batch size:     8 | lm loss: 4.977434E+00 | loss scale: 262144.0 | grad norm: 1.856 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration     1500/    2000 | consumed samples:        12000 | elapsed time per iteration (ms): 2950.4 | learning rate: 6.956E-05 | global batch size:     8 | lm loss: 4.809360E+00 | loss scale: 262144.0 | grad norm: 1.659 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration     1600/    2000 | consumed samples:        12800 | elapsed time per iteration (ms): 2292.5 | learning rate: 7.425E-05 | global batch size:     8 | lm loss: 4.620577E+00 | loss scale: 262144.0 | grad norm: 1.759 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration     1700/    2000 | consumed samples:        13600 | elapsed time per iteration (ms): 2957.8 | learning rate: 7.894E-05 | global batch size:     8 | lm loss: 4.447711E+00 | loss scale: 262144.0 | grad norm: 1.743 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration     1800/    2000 | consumed samples:        14400 | elapsed time per iteration (ms): 2407.2 | learning rate: 8.362E-05 | global batch size:     8 | lm loss: 4.211063E+00 | loss scale: 262144.0 | grad norm: 1.733 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration     1900/    2000 | consumed samples:        15200 | elapsed time per iteration (ms): 2554.1 | learning rate: 8.831E-05 | global batch size:     8 | lm loss: 3.968034E+00 | loss scale: 262144.0 | grad norm: 1.993 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration     2000/    2000 | consumed samples:        16000 | elapsed time per iteration (ms): 3024.0 | learning rate: 9.300E-05 | global batch size:     8 | lm loss: 3.732524E+00 | loss scale: 262144.0 | grad norm: 1.874 | number of skipped iterations:   0 | number of nan iterations:   0 |
------------------------------------------------------------------------------------------------
 validation loss at iteration 2000 | lm loss value: 6.237787E+00 | lm loss PPL: 5.117247E+02 | 
------------------------------------------------------------------------------------------------
saving checkpoint at iteration    2000 to /workspace/model/megatron-models/345m-init-4tp
  successfully saved checkpoint at iteration    2000 to /workspace/model/megatron-models/345m-init-4tp
(min, max) time across ranks (ms):
    save-checkpoint ................................: (52793.38, 52793.87)
[after training is done] datetime: 2023-07-16 15:36:49 
saving checkpoint at iteration    2000 to /workspace/model/megatron-models/345m-init-4tp
------------------------------------------------------------------------------------------------------------------
 validation loss at the end of training for val data | lm loss value: 6.296053E+00 | lm loss PPL: 5.424267E+02 | 
------------------------------------------------------------------------------------------------------------------
  successfully saved checkpoint at iteration    2000 to /workspace/model/megatron-models/345m-init-4tp
-------------------------------------------------------------------------------------------------------------------
 validation loss at the end of training for test data | lm loss value: 6.454095E+00 | lm loss PPL: 6.352988E+02 | 
-------------------------------------------------------------------------------------------------------------------
```

</p></details>


<details><summary>详细输出</summary><p>


```
tree -h /workspace/model/megatron-models/345m-init-4tp
/workspace/model/megatron-models/345m-init-4tp
├── [4.0K]  iter_0001000
│   ├── [4.0K]  mp_rank_00
│   │   └── [1.2G]  model_optim_rng.pt
│   ├── [4.0K]  mp_rank_01
│   │   └── [1.2G]  model_optim_rng.pt
│   ├── [4.0K]  mp_rank_02
│   │   └── [1.2G]  model_optim_rng.pt
│   └── [4.0K]  mp_rank_03
│       └── [1.2G]  model_optim_rng.pt
├── [4.0K]  iter_0002000
│   ├── [4.0K]  mp_rank_00
│   │   └── [1.2G]  model_optim_rng.pt
│   ├── [4.0K]  mp_rank_01
│   │   └── [1.2G]  model_optim_rng.pt
│   ├── [4.0K]  mp_rank_02
│   │   └── [1.2G]  model_optim_rng.pt
│   └── [4.0K]  mp_rank_03
│       └── [1.2G]  model_optim_rng.pt
└── [   4]  latest_checkpointed_iteration.txt

10 directories, 9 files

> cat /workspace/model/megatron-models/345m-init-4tp/latest_checkpointed_iteration.txt 
2000
```

</p></details>




### 模型并行（4PP）


<details><summary>详细输出</summary><p>


```
> nohup sh examples/pretrain_gpt_distributed_with_pp4.sh >gpt-pp4.log  &
WARNING:torch.distributed.run:
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
*****************************************
using world size: 4, data-parallel-size: 1, tensor-model-parallel size: 1, pipeline-model-parallel size: 4 
using torch.float16 for parameters ...
------------------------ arguments ------------------------
  accumulate_allreduce_grads_in_fp32 .............. False
  adam_beta1 ...................................... 0.9
  adam_beta2 ...................................... 0.999
  adam_eps ........................................ 1e-08
  add_bias_linear ................................. True
  add_position_embedding .......................... True
  adlr_autoresume ................................. False
  adlr_autoresume_interval ........................ 1000
  apply_layernorm_1p .............................. False
  apply_query_key_layer_scaling ................... True
  apply_residual_connection_post_layernorm ........ False
  async_tensor_model_parallel_allreduce ........... True
  attention_dropout ............................... 0.1
  attention_softmax_in_fp32 ....................... False
  barrier_with_L1_time ............................ True
  bert_binary_head ................................ True
  bert_embedder_type .............................. megatron
  bert_load ....................................... None
  bf16 ............................................ False
  bias_dropout_fusion ............................. True
  bias_gelu_fusion ................................ True
  biencoder_projection_dim ........................ 0
  biencoder_shared_query_context_model ............ False
  block_data_path ................................. None
  classes_fraction ................................ 1.0
  clip_grad ....................................... 1.0
  consumed_train_samples .......................... 0
  consumed_valid_samples .......................... 0
  data_cache_path ................................. None
  data_impl ....................................... mmap
  data_parallel_random_init ....................... False
  data_parallel_size .............................. 1
  data_path ....................................... ['/workspace/data/my-gpt2_text_document']
  data_per_class_fraction ......................... 1.0
  data_sharding ................................... True
  dataloader_type ................................. single
  DDP_impl ........................................ local
  decoder_num_layers .............................. None
  decoder_seq_length .............................. None
  dino_bottleneck_size ............................ 256
  dino_freeze_last_layer .......................... 1
  dino_head_hidden_size ........................... 2048
  dino_local_crops_number ......................... 10
  dino_local_img_size ............................. 96
  dino_norm_last_layer ............................ False
  dino_teacher_temp ............................... 0.07
  dino_warmup_teacher_temp ........................ 0.04
  dino_warmup_teacher_temp_epochs ................. 30
  distribute_saved_activations .................... False
  distributed_backend ............................. nccl
  distributed_timeout_minutes ..................... 10
  embedding_path .................................. None
  empty_unused_memory_level ....................... 0
  encoder_num_layers .............................. 24
  encoder_seq_length .............................. 1024
  end_weight_decay ................................ 0.01
  eod_mask_loss ................................... False
  eval_interval ................................... 1000
  eval_iters ...................................... 10
  evidence_data_path .............................. None
  exit_duration_in_mins ........................... None
  exit_interval ................................... None
  exit_on_missing_checkpoint ...................... False
  exit_signal_handler ............................. False
  ffn_hidden_size ................................. 4096
  finetune ........................................ False
  fp16 ............................................ True
  fp16_lm_cross_entropy ........................... False
  fp32_residual_connection ........................ False
  fp8_amax_compute_algo ........................... most_recent
  fp8_amax_history_len ............................ 1
  fp8_e4m3 ........................................ False
  fp8_hybrid ...................................... False
  fp8_interval .................................... 1
  fp8_margin ...................................... 0
  fp8_wgrad ....................................... True
  global_batch_size ............................... 8
  gradient_accumulation_fusion .................... True
  head_lr_mult .................................... 1.0
  hidden_dropout .................................. 0.1
  hidden_size ..................................... 1024
  hysteresis ...................................... 2
  ict_head_size ................................... None
  ict_load ........................................ None
  img_h ........................................... 224
  img_w ........................................... 224
  indexer_batch_size .............................. 128
  indexer_log_interval ............................ 1000
  inference_batch_times_seqlen_threshold .......... 512
  init_method_std ................................. 0.02
  init_method_xavier_uniform ...................... False
  initial_loss_scale .............................. 4294967296
  iter_per_epoch .................................. 1250
  kv_channels ..................................... 64
  layernorm_epsilon ............................... 1e-05
  lazy_mpu_init ................................... None
  load ............................................ /workspace/model/megatron-models/345m-init-4pp
  local_rank ...................................... None
  log_batch_size_to_tensorboard ................... False
  log_interval .................................... 100
  log_learning_rate_to_tensorboard ................ True
  log_loss_scale_to_tensorboard ................... True
  log_memory_to_tensorboard ....................... False
  log_num_zeros_in_grad ........................... False
  log_params_norm ................................. False
  log_timers_to_tensorboard ....................... False
  log_validation_ppl_to_tensorboard ............... False
  log_world_size_to_tensorboard ................... False
  loss_scale ...................................... None
  loss_scale_window ............................... 1000
  lr .............................................. 0.00015
  lr_decay_iters .................................. 320000
  lr_decay_samples ................................ None
  lr_decay_style .................................. cosine
  lr_warmup_fraction .............................. 0.01
  lr_warmup_iters ................................. 0
  lr_warmup_samples ............................... 0
  make_vocab_size_divisible_by .................... 128
  mask_factor ..................................... 1.0
  mask_prob ....................................... 0.15
  mask_type ....................................... random
  masked_softmax_fusion ........................... True
  max_position_embeddings ......................... 1024
  max_tokens_to_oom ............................... 12000
  merge_file ...................................... /workspace/model/gpt2-vocab/gpt2-merges.txt
  micro_batch_size ................................ 2
  min_loss_scale .................................. 1.0
  min_lr .......................................... 1e-05
  mmap_warmup ..................................... False
  no_load_optim ................................... None
  no_load_rng ..................................... None
  no_persist_layer_norm ........................... False
  no_save_optim ................................... None
  no_save_rng ..................................... None
  num_attention_heads ............................. 16
  num_channels .................................... 3
  num_classes ..................................... 1000
  num_experts ..................................... None
  num_layers ...................................... 24
  num_layers_per_virtual_pipeline_stage ........... None
  num_workers ..................................... 2
  onnx_safe ....................................... None
  openai_gelu ..................................... False
  optimizer ....................................... adam
  output_bert_embeddings .......................... False
  overlap_p2p_comm ................................ False
  override_opt_param_scheduler .................... False
  params_dtype .................................... torch.float16
  patch_dim ....................................... 16
  perform_initialization .......................... True
  pipeline_model_parallel_size .................... 4
  pipeline_model_parallel_split_rank .............. None
  query_in_block_prob ............................. 0.1
  rampup_batch_size ............................... None
  rank ............................................ 0
  recompute_granularity ........................... None
  recompute_method ................................ None
  recompute_num_layers ............................ 1
  reset_attention_mask ............................ False
  reset_position_ids .............................. False
  retriever_report_topk_accuracies ................ []
  retriever_score_scaling ......................... False
  retriever_seq_length ............................ 256
  retro_add_retriever ............................. False
  retro_cyclic_train_iters ........................ None
  retro_encoder_attention_dropout ................. 0.1
  retro_encoder_hidden_dropout .................... 0.1
  retro_encoder_layers ............................ 2
  retro_num_neighbors ............................. 2
  retro_num_retrieved_chunks ...................... 2
  retro_return_doc_ids ............................ False
  retro_workdir ................................... None
  rotary_percent .................................. 1.0
  sample_rate ..................................... 1.0
  save ............................................ /workspace/model/megatron-models/345m-init-4pp
  save_interval ................................... 1000
  scatter_gather_tensors_in_pipeline .............. True
  seed ............................................ 1234
  seq_length ...................................... 1024
  sequence_parallel ............................... False
  sgd_momentum .................................... 0.9
  short_seq_prob .................................. 0.1
  split ........................................... 700,200,100
  squared_relu .................................... False
  standalone_embedding_stage ...................... False
  start_weight_decay .............................. 0.01
  swiglu .......................................... False
  swin_backbone_type .............................. tiny
  tensor_model_parallel_size ...................... 1
  tensorboard_dir ................................. None
  tensorboard_log_interval ........................ 1
  tensorboard_queue_size .......................... 1000
  test_data_path .................................. None
  timing_log_level ................................ 0
  timing_log_option ............................... minmax
  titles_data_path ................................ None
  tokenizer_model ................................. None
  tokenizer_type .................................. GPT2BPETokenizer
  train_data_path ................................. None
  train_iters ..................................... 2000
  train_samples ................................... None
  transformer_impl ................................ local
  transformer_pipeline_model_parallel_size ........ 4
  untie_embeddings_and_output_weights ............. False
  use_checkpoint_args ............................. False
  use_checkpoint_opt_param_scheduler .............. False
  use_contiguous_buffers_in_local_ddp ............. True
  use_cpu_initialization .......................... None
  use_distributed_optimizer ....................... False
  use_flash_attn .................................. False
  use_one_sent_docs ............................... False
  use_ring_exchange_p2p ........................... False
  use_rotary_position_embeddings .................. False
  valid_data_path ................................. None
  variable_seq_lengths ............................ False
  virtual_pipeline_model_parallel_size ............ None
  vision_backbone_type ............................ vit
  vision_pretraining .............................. False
  vision_pretraining_type ......................... classify
  vocab_extra_ids ................................. 0
  vocab_file ...................................... /workspace/model/gpt2-vocab/gpt2-vocab.json
  vocab_size ...................................... None
  weight_decay .................................... 0.01
  weight_decay_incr_style ......................... constant
  world_size ...................................... 4
-------------------- end of arguments ---------------------
setting number of micro-batches to constant 4
> building GPT2BPETokenizer tokenizer ...
 > padded vocab (size: 50257) with 47 dummy tokens (new size: 50304)
> initializing torch distributed ...
> initialized tensor model parallel with size 1
> initialized pipeline model parallel with size 4
> setting random seeds to 1234 ...
> compiling dataset index builder ...
make: Entering directory '/workspace/code/Megatron-LM/megatron/data'
make: Nothing to be done for 'default'.
make: Leaving directory '/workspace/code/Megatron-LM/megatron/data'
>>> done with dataset index builder. Compilation time: 0.204 seconds
> compiling and loading fused kernels ...
Detected CUDA files, patching ldflags
Emitting ninja build file /workspace/code/Megatron-LM/megatron/fused_kernels/build/build.ninja...
Building extension module scaled_upper_triang_masked_softmax_cuda...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
ninja: no work to do.
Loading extension module scaled_upper_triang_masked_softmax_cuda...
Detected CUDA files, patching ldflags
Emitting ninja build file /workspace/code/Megatron-LM/megatron/fused_kernels/build/build.ninja...
Building extension module scaled_masked_softmax_cuda...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
ninja: no work to do.
Loading extension module scaled_masked_softmax_cuda...
Detected CUDA files, patching ldflags
Emitting ninja build file /workspace/code/Megatron-LM/megatron/fused_kernels/build/build.ninja...
Building extension module scaled_softmax_cuda...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
ninja: no work to do.
Loading extension module scaled_softmax_cuda...
>>> done with compiling and loading fused kernels. Compilation time: 4.538 seconds
time to initialize megatron (seconds): 7.707
[after megatron is initialized] datetime: 2023-07-17 01:14:01 
building GPT model ...
 > number of parameters on (tensor, pipeline) model parallel rank (0, 1): 75577344
 > number of parameters on (tensor, pipeline) model parallel rank (0, 2): 75577344
 > number of parameters on (tensor, pipeline) model parallel rank (0, 3): 127090688
 > number of parameters on (tensor, pipeline) model parallel rank (0, 0): 128137216
> learning rate decay style: cosine
WARNING: could not find the metadata file /workspace/model/megatron-models/345m-init-4pp/latest_checkpointed_iteration.txt 
    will not load any checkpoints and will start from random
/usr/local/lib/python3.8/dist-packages/torch/distributed/distributed_c10d.py:2603: UserWarning: torch.distributed._all_gather_base is a private function and will be deprecated. Please use torch.distributed.all_gather_into_tensor instead.
  warnings.warn(
/usr/local/lib/python3.8/dist-packages/torch/distributed/distributed_c10d.py:2603: UserWarning: torch.distributed._all_gather_base is a private function and will be deprecated. Please use torch.distributed.all_gather_into_tensor instead.
  warnings.warn(
/usr/local/lib/python3.8/dist-packages/torch/distributed/distributed_c10d.py:2603: UserWarning: torch.distributed._all_gather_base is a private function and will be deprecated. Please use torch.distributed.all_gather_into_tensor instead.
  warnings.warn(
/usr/local/lib/python3.8/dist-packages/torch/distributed/distributed_c10d.py:2603: UserWarning: torch.distributed._all_gather_base is a private function and will be deprecated. Please use torch.distributed.all_gather_into_tensor instead.
  warnings.warn(
(min, max) time across ranks (ms):
    load-checkpoint ................................: (6.23, 6.92)
[after model, optimizer, and learning rate scheduler are built] datetime: 2023-07-17 01:14:01 
> building train, validation, and test datasets ...
 > datasets target sizes (minimum size):
    train:      16000
    validation: 240
    test:       80
> building train, validation, and test datasets for GPT ...
Single data path provided for train, valid & test
data_prefix: ['/workspace/data/my-gpt2_text_document']
 > building dataset index ...
    reading sizes...
    reading pointers...
    reading document index...
    creating numpy buffer of mmap...
    creating memory view of numpy buffer...
 > finished creating indexed dataset in 0.002993 seconds
    number of documents: 2456
 > dataset split:
    train:
     document indices in [0, 1719) total of 1719 documents
    validation:
     document indices in [1719, 2210) total of 491 documents
    test:
     document indices in [2210, 2456) total of 246 documents
 > loading doc-idx mapping from /workspace/data/index-cache/09a6ab12e4c21658eefabc732b2e110b_doc_idx.npy
 > loading sample-idx mapping from /workspace/data/index-cache/09a6ab12e4c21658eefabc732b2e110b_sample_idx.npy
 > loading shuffle-idx mapping from /workspace/data/index-cache/09a6ab12e4c21658eefabc732b2e110b_shuffle_idx.npy
    loaded indexed file in 0.004 seconds
    total number of samples: 16296
    total number of epochs: 36
 > loading doc-idx mapping from /workspace/data/index-cache/2d16c1014278a21da8aa7bdfcf0f1e3f_doc_idx.npy
 > loading sample-idx mapping from /workspace/data/index-cache/2d16c1014278a21da8aa7bdfcf0f1e3f_sample_idx.npy
 > loading shuffle-idx mapping from /workspace/data/index-cache/2d16c1014278a21da8aa7bdfcf0f1e3f_shuffle_idx.npy
    loaded indexed file in 0.003 seconds
    total number of samples: 333
    total number of epochs: 3
 > loading doc-idx mapping from /workspace/data/index-cache/f1583c0457c0ee9c3ecb3c068b3badd8_doc_idx.npy
 > loading sample-idx mapping from /workspace/data/index-cache/f1583c0457c0ee9c3ecb3c068b3badd8_sample_idx.npy
 > loading shuffle-idx mapping from /workspace/data/index-cache/f1583c0457c0ee9c3ecb3c068b3badd8_shuffle_idx.npy
    loaded indexed file in 0.002 seconds
    total number of samples: 108
    total number of epochs: 2
> finished creating GPT datasets ...
[after dataloaders are built] datetime: 2023-07-17 01:14:04 
done with setup ...
(min, max) time across ranks (ms):
    model-and-optimizer-setup ......................: (455.86, 465.76)
    train/valid/test-data-iterators-setup ..........: (2100.27, 2154.09)
training ...
[before the start of training step] datetime: 2023-07-17 01:14:04 
[W ProcessGroupNCCL.cpp:1692] Warning: 0NCCL_AVOID_RECORD_STREAMS=1 has no effect for point-to-point collectives. (function operator())
[W ProcessGroupNCCL.cpp:1692] Warning: 0NCCL_AVOID_RECORD_STREAMS=1 has no effect for point-to-point collectives. (function operator())
[W ProcessGroupNCCL.cpp:1692] Warning: 0NCCL_AVOID_RECORD_STREAMS=1 has no effect for point-to-point collectives. (function operator())
[W ProcessGroupNCCL.cpp:1692] Warning: 0NCCL_AVOID_RECORD_STREAMS=1 has no effect for point-to-point collectives. (function operator())
[W ProcessGroupNCCL.cpp:1692] Warning: 0NCCL_AVOID_RECORD_STREAMS=1 has no effect for point-to-point collectives. (function operator())
[W ProcessGroupNCCL.cpp:1692] Warning: 0NCCL_AVOID_RECORD_STREAMS=1 has no effect for point-to-point collectives. (function operator())
[W ProcessGroupNCCL.cpp:1692] Warning: 0NCCL_AVOID_RECORD_STREAMS=1 has no effect for point-to-point collectives. (function operator())
[W ProcessGroupNCCL.cpp:1692] Warning: 0NCCL_AVOID_RECORD_STREAMS=1 has no effect for point-to-point collectives. (function operator())
[Rank 2] (after 100 iterations) memory (MB) | allocated: 1529.77783203125 | max allocated: 4104.89501953125 | reserved: 4170.0 | max reserved: 4170.0
[Rank 0] (after 100 iterations) memory (MB) | allocated: 2524.27783203125 | max allocated: 7643.34033203125 | reserved: 7766.0 | max reserved: 7766.0
[Rank 1] (after 100 iterations) memory (MB) | allocated: 1529.77783203125 | max allocated: 5477.08251953125 | reserved: 5498.0 | max reserved: 5498.0
 iteration      100/    2000 | consumed samples:          800 | elapsed time per iteration (ms): 452.0 | learning rate: 3.984E-06 | global batch size:     8 | lm loss: 9.682818E+00 | loss scale: 262144.0 | grad norm: 3.606 | number of skipped iterations:  15 | number of nan iterations:   0 |
[Rank 3] (after 100 iterations) memory (MB) | allocated: 2513.31787109375 | max allocated: 4390.34521484375 | reserved: 4522.0 | max reserved: 4522.0
 iteration      200/    2000 | consumed samples:         1600 | elapsed time per iteration (ms): 311.9 | learning rate: 8.672E-06 | global batch size:     8 | lm loss: 8.489280E+00 | loss scale: 262144.0 | grad norm: 4.001 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration      300/    2000 | consumed samples:         2400 | elapsed time per iteration (ms): 399.3 | learning rate: 1.336E-05 | global batch size:     8 | lm loss: 7.642234E+00 | loss scale: 262144.0 | grad norm: 4.426 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration      400/    2000 | consumed samples:         3200 | elapsed time per iteration (ms): 382.2 | learning rate: 1.805E-05 | global batch size:     8 | lm loss: 6.942851E+00 | loss scale: 262144.0 | grad norm: 2.279 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration      500/    2000 | consumed samples:         4000 | elapsed time per iteration (ms): 376.9 | learning rate: 2.273E-05 | global batch size:     8 | lm loss: 6.551709E+00 | loss scale: 262144.0 | grad norm: 2.192 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration      600/    2000 | consumed samples:         4800 | elapsed time per iteration (ms): 379.9 | learning rate: 2.742E-05 | global batch size:     8 | lm loss: 6.246989E+00 | loss scale: 262144.0 | grad norm: 2.499 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration      700/    2000 | consumed samples:         5600 | elapsed time per iteration (ms): 344.4 | learning rate: 3.211E-05 | global batch size:     8 | lm loss: 6.110545E+00 | loss scale: 262144.0 | grad norm: 1.982 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration      800/    2000 | consumed samples:         6400 | elapsed time per iteration (ms): 392.7 | learning rate: 3.680E-05 | global batch size:     8 | lm loss: 5.888660E+00 | loss scale: 262144.0 | grad norm: 1.912 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration      900/    2000 | consumed samples:         7200 | elapsed time per iteration (ms): 380.3 | learning rate: 4.148E-05 | global batch size:     8 | lm loss: 5.673041E+00 | loss scale: 262144.0 | grad norm: 2.449 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration     1000/    2000 | consumed samples:         8000 | elapsed time per iteration (ms): 381.1 | learning rate: 4.617E-05 | global batch size:     8 | lm loss: 5.564542E+00 | loss scale: 262144.0 | grad norm: 2.265 | number of skipped iterations:   0 | number of nan iterations:   0 |
------------------------------------------------------------------------------------------------
 validation loss at iteration 1000 | lm loss value: 6.355064E+00 | lm loss PPL: 5.753991E+02 | 
------------------------------------------------------------------------------------------------
saving checkpoint at iteration    1000 to /workspace/model/megatron-models/345m-init-4pp
  successfully saved checkpoint at iteration    1000 to /workspace/model/megatron-models/345m-init-4pp
(min, max) time across ranks (ms):
    save-checkpoint ................................: (32390.20, 32392.97)
 iteration     1100/    2000 | consumed samples:         8800 | elapsed time per iteration (ms): 665.5 | learning rate: 5.086E-05 | global batch size:     8 | lm loss: 5.377799E+00 | loss scale: 524288.0 | grad norm: 1.871 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration     1200/    2000 | consumed samples:         9600 | elapsed time per iteration (ms): 392.0 | learning rate: 5.555E-05 | global batch size:     8 | lm loss: 5.238356E+00 | loss scale: 524288.0 | grad norm: 1.560 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration     1300/    2000 | consumed samples:        10400 | elapsed time per iteration (ms): 381.3 | learning rate: 6.023E-05 | global batch size:     8 | lm loss: 5.050917E+00 | loss scale: 524288.0 | grad norm: 2.267 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration     1400/    2000 | consumed samples:        11200 | elapsed time per iteration (ms): 382.2 | learning rate: 6.492E-05 | global batch size:     8 | lm loss: 4.976177E+00 | loss scale: 524288.0 | grad norm: 1.927 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration     1500/    2000 | consumed samples:        12000 | elapsed time per iteration (ms): 383.2 | learning rate: 6.961E-05 | global batch size:     8 | lm loss: 4.809005E+00 | loss scale: 524288.0 | grad norm: 1.670 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration     1600/    2000 | consumed samples:        12800 | elapsed time per iteration (ms): 379.0 | learning rate: 7.430E-05 | global batch size:     8 | lm loss: 4.620012E+00 | loss scale: 524288.0 | grad norm: 1.684 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration     1700/    2000 | consumed samples:        13600 | elapsed time per iteration (ms): 318.1 | learning rate: 7.898E-05 | global batch size:     8 | lm loss: 4.443442E+00 | loss scale: 524288.0 | grad norm: 1.676 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration     1800/    2000 | consumed samples:        14400 | elapsed time per iteration (ms): 405.2 | learning rate: 8.367E-05 | global batch size:     8 | lm loss: 4.210877E+00 | loss scale: 524288.0 | grad norm: 1.578 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration     1900/    2000 | consumed samples:        15200 | elapsed time per iteration (ms): 379.4 | learning rate: 8.836E-05 | global batch size:     8 | lm loss: 3.972883E+00 | loss scale: 524288.0 | grad norm: 1.674 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration     2000/    2000 | consumed samples:        16000 | elapsed time per iteration (ms): 385.6 | learning rate: 9.305E-05 | global batch size:     8 | lm loss: 3.723653E+00 | loss scale: 524288.0 | grad norm: 1.719 | number of skipped iterations:   0 | number of nan iterations:   0 |
------------------------------------------------------------------------------------------------
 validation loss at iteration 2000 | lm loss value: 6.176825E+00 | lm loss PPL: 4.814607E+02 | 
------------------------------------------------------------------------------------------------
saving checkpoint at iteration    2000 to /workspace/model/megatron-models/345m-init-4pp
  successfully saved checkpoint at iteration    2000 to /workspace/model/megatron-models/345m-init-4pp
(min, max) time across ranks (ms):
    save-checkpoint ................................: (59290.87, 59295.51)
[after training is done] datetime: 2023-07-17 01:28:11 
saving checkpoint at iteration    2000 to /workspace/model/megatron-models/345m-init-4pp
------------------------------------------------------------------------------------------------------------------
 validation loss at the end of training for val data | lm loss value: 6.235375E+00 | lm loss PPL: 5.104922E+02 | 
------------------------------------------------------------------------------------------------------------------
  successfully saved checkpoint at iteration    2000 to /workspace/model/megatron-models/345m-init-4pp
-------------------------------------------------------------------------------------------------------------------
 validation loss at the end of training for test data | lm loss value: 6.396670E+00 | lm loss PPL: 5.998441E+02 | 
-------------------------------------------------------------------------------------------------------------------

```

</p></details>




<details><summary>模型权重输出：</summary><p>


```
> tree -h /workspace/model/megatron-models/345m-init-4pp
/workspace/model/megatron-models/345m-init-4pp
├── [4.0K]  iter_0001000
│   ├── [4.0K]  mp_rank_00_000
│   │   └── [1.7G]  model_optim_rng.pt
│   ├── [4.0K]  mp_rank_00_001
│   │   └── [1009M]  model_optim_rng.pt
│   ├── [4.0K]  mp_rank_00_002
│   │   └── [1009M]  model_optim_rng.pt
│   └── [4.0K]  mp_rank_00_003
│       └── [1.7G]  model_optim_rng.pt
├── [4.0K]  iter_0002000
│   ├── [4.0K]  mp_rank_00_000
│   │   └── [1.7G]  model_optim_rng.pt
│   ├── [4.0K]  mp_rank_00_001
│   │   └── [1009M]  model_optim_rng.pt
│   ├── [4.0K]  mp_rank_00_002
│   │   └── [1009M]  model_optim_rng.pt
│   └── [4.0K]  mp_rank_00_003
│       └── [1.7G]  model_optim_rng.pt
└── [   4]  latest_checkpointed_iteration.txt

10 directories, 9 files

> cat /workspace/model/megatron-models/345m-init-4pp/latest_checkpointed_iteration.txt 
2000
```

</p></details>


显存占用：
```
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A   2630871      C   /usr/bin/python                  8680MiB |
|    1   N/A  N/A   2630872      C   /usr/bin/python                  6408MiB |
|    2   N/A  N/A   2630873      C   /usr/bin/python                  5080MiB |
|    3   N/A  N/A   2630874      C   /usr/bin/python                  5436MiB |
+-----------------------------------------------------------------------------+
```


