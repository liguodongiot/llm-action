


```
docker run -dt --name nvidia_pytorch_env --restart=always --gpus all \
--network=host \
--shm-size 4G \
-v /home/guodong.li/workspace:/workspace \
-w /workspace \
nvcr.io/nvidia/pytorch:23.04-py3 \
/bin/bash

docker exec -it nvidia_pytorch_env bash
```


```
pip install sentencepiece
pip install transformers
pip install deepspeed==0.9.5
pip install einops==0.6.1
pip uninstall -y flash-attn && pip install flash-attn --no-build-isolation
```


<details><summary>训练日志详细输出</summary><p>

```
> bash examples_deepspeed/pretrain_llama_distributed.sh
+ BASE_PATH=./tmp
+ DS_CONFIG=./tmp/deepspeed.json
+ DATASET_1=/workspace/data/gpt2-data/my-gpt2_text_document
+ DATASET_2=/workspace/data/gpt2-data/my-gpt2-1_text_document
+ DATASET='1 /workspace/data/gpt2-data/my-gpt2_text_document 2 /workspace/data/gpt2-data/my-gpt2-1_text_document'
+ CHECKPOINT_PATH=./tmp
+ TOKENIZER_PATH=/workspace/model/llama-tokenizer/tokenizer.model
+ TP=2
+ PP=2
+ ZERO_STAGE=0
+ GPUS_PER_NODE=8
+ MASTER_ADDR=localhost
+ MASTER_PORT=6000
+ NNODES=1
+ NODE_RANK=0
+ HIDDEN_SIZE=2048
+ FFN_HIDDEN_SIZE=5504
+ NUM_LAYERS=24
+ NUM_HEADS=16
+ SEQ_LENGTH=2048
+ MICRO_BATCH_SIZE=4
+ GLOBAL_BATCH_SIZE=32
+ TRAIN_STEPS=2500
+ LR=3e-4
+ MIN_LR=3e-5
+ LR_WARMUP_STEPS=2000
+ WEIGHT_DECAY=0.1
+ GRAD_CLIP=1
+ cat
+ ds_args=
+ ds_args=' --deepspeed '
+ ds_args=' --deepspeed_config=./tmp/deepspeed.json  --deepspeed '
+ ds_args=' --zero-stage=0  --deepspeed_config=./tmp/deepspeed.json  --deepspeed '
+ ds_args=' --deepspeed-activation-checkpointing  --zero-stage=0  --deepspeed_config=./tmp/deepspeed.json  --deepspeed '
+ DISTRIBUTED_ARGS='--nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6000'
+ torchrun --nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6000 pretrain_gpt.py --tensor-model-parallel-size 2 --pipeline-model-parallel-size 2 --num-layers 24 --hidden-size 2048 --ffn-hidden-size 5504 --num-attention-heads 16 --micro-batch-size 4 --global-batch-size 32 --seq-length 2048 --max-position-embeddings 2048 --train-iters 2500 --save ./tmp --load ./tmp --data-path 1 /workspace/data/gpt2-data/my-gpt2_text_document 2 /workspace/data/gpt2-data/my-gpt2-1_text_document --data-impl mmap --tokenizer-type GPTSentencePieceTokenizer --tokenizer-model /workspace/model/llama-tokenizer/tokenizer.model --split 900,50,50 --distributed-backend nccl --lr 3e-4 --lr-decay-style cosine --min-lr 3e-5 --weight-decay 0.1 --clip-grad 1 --lr-warmup-iters 2000 --optimizer adam --adam-beta1 0.9 --adam-beta2 0.95 --log-interval 1 --save-interval 10000 --eval-interval 1000 --eval-iters 10 --fp16 --no-query-key-layer-scaling --attention-dropout 0 --hidden-dropout 0 --use-rotary-position-embeddings --untie-embeddings-and-output-weights --swiglu --normalization rmsnorm --disable-bias-linear --deepspeed-activation-checkpointing --zero-stage=0 --deepspeed_config=./tmp/deepspeed.json --deepspeed
...
--------------------------------------------------
DeepSpeed C++/CUDA extension op report
--------------------------------------------------
...
async_io ............... [NO] ....... [NO]
cpu_adagrad ............ [NO] ....... [OKAY]
cpu_adam ............... [NO] ....... [OKAY]
fused_adam ............. [NO] ....... [OKAY]
fused_lamb ............. [NO] ....... [OKAY]
quantizer .............. [NO] ....... [OKAY]
random_ltd ............. [NO] ....... [OKAY]
...
--------------------------------------------------
DeepSpeed general environment info:
torch install path ............... ['/usr/local/lib/python3.8/dist-packages/torch']
torch version .................... 2.1.0a0+fe05266
deepspeed install path ........... ['/usr/local/lib/python3.8/dist-packages/deepspeed']
deepspeed info ................... 0.9.5, unknown, unknown
torch cuda version ............... 12.1
torch hip version ................ None
nvcc version ..................... 12.1
deepspeed wheel compiled w. ...... torch 2.1, cuda 12.1
using world size: 8, data-parallel-size: 2, tensor-model-parallel size: 2, pipeline-model-parallel size: 2
using torch.float16 for parameters ...
------------------------ arguments ------------------------
  accumulate_allreduce_grads_in_fp32 .............. False
  adam_beta1 ...................................... 0.9
  adam_beta2 ...................................... 0.95
  adam_eps ........................................ 1e-08
  add_bias_linear ................................. False
  add_position_embedding .......................... False
  adlr_autoresume ................................. False
  adlr_autoresume_interval ........................ 1000
  aml_data_download_path .......................... None
  apply_layernorm_1p .............................. False
  apply_query_key_layer_scaling ................... False
  apply_residual_connection_post_layernorm ........ False
  async_tensor_model_parallel_allreduce ........... False
  attention_dropout ............................... 0.0
  attention_softmax_in_fp32 ....................... False
  barrier_with_L1_time ............................ True
  bert_binary_head ................................ True
  bert_embedder_type .............................. megatron
  bert_load ....................................... None
  bf16 ............................................ False
  bias_dropout_fusion ............................. True
  bias_gelu_fusion ................................ False
  biencoder_projection_dim ........................ 0
  biencoder_shared_query_context_model ............ False
  block_data_path ................................. None
  checkpoint_activations .......................... False
  checkpoint_in_cpu ............................... False
  checkpoint_num_layers ........................... 1
  classes_fraction ................................ 1.0
  clip_grad ....................................... 1.0
  compression_training ............................ False
  consumed_train_samples .......................... 0
  consumed_train_tokens ........................... 0
  consumed_valid_samples .......................... 0
  contigious_checkpointing ........................ False
  cpu_optimizer ................................... False
  cpu_torch_adam .................................. False
  create_moe_param_group .......................... False
  curriculum_learning_legacy ...................... False
  data_cache_path ................................. None
  data_efficiency_curriculum_learning ............. False
  data_impl ....................................... mmap
  data_parallel_random_init ....................... False
  data_parallel_size .............................. 2
  data_path ....................................... ['1', '/workspace/data/gpt2-data/my-gpt2_text_document', '2', '/workspace/data/gpt2-data/my-gpt2-1_text_document']
  data_per_class_fraction ......................... 1.0
  data_sharding ................................... True
  dataloader_type ................................. single
  DDP_impl ........................................ local
  decoder_num_layers .............................. None
  decoder_seq_length .............................. None
  deepscale ....................................... False
  deepscale_config ................................ None
  deepspeed ....................................... True
  deepspeed_activation_checkpointing .............. True
  deepspeed_config ................................ ./tmp/deepspeed.json
  deepspeed_mpi ................................... False
  dino_bottleneck_size ............................ 256
  dino_freeze_last_layer .......................... 1
  dino_head_hidden_size ........................... 2048
  dino_local_crops_number ......................... 10
  dino_local_img_size ............................. 96
  dino_norm_last_layer ............................ False
  dino_teacher_temp ............................... 0.07
  dino_warmup_teacher_temp ........................ 0.04
  dino_warmup_teacher_temp_epochs ................. 30
  distribute_checkpointed_activations ............. False
  distribute_saved_activations .................... False
  distributed_backend ............................. nccl
  distributed_timeout_minutes ..................... 10
  ds_inference .................................... False
  ds_pipeline_enabled ............................. True
  embedding_path .................................. None
  embedding_weights_in_fp32 ....................... False
  empty_unused_memory_level ....................... 0
  enable_expert_tensor_parallelism ................ False
  encoder_num_layers .............................. 24
  encoder_seq_length .............................. 2048
  end_weight_decay ................................ 0.1
  eod_mask_loss ................................... False
  eval_interval ................................... 1000
  eval_iters ...................................... 10
  evidence_data_path .............................. None
  exit_duration_in_mins ........................... None
  exit_interval ................................... None
  exit_on_missing_checkpoint ...................... False
  exit_signal_handler ............................. False
  expert_interval ................................. 2
  ffn_hidden_size ................................. 5504
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
  global_batch_size ............................... 32
  gradient_accumulation_fusion .................... True
  head_lr_mult .................................... 1.0
  hidden_dropout .................................. 0.0
  hidden_size ..................................... 2048
  hidden_size_teacher ............................. None
  hysteresis ...................................... 2
  ict_head_size ................................... None
  ict_load ........................................ None
  img_h ........................................... 224
  img_w ........................................... 224
  indexer_batch_size .............................. 128
  indexer_log_interval ............................ 1000
  inference ....................................... False
  inference_batch_times_seqlen_threshold .......... 512
  init_method_std ................................. 0.02
  init_method_xavier_uniform ...................... False
  initial_loss_scale .............................. 4294967296
  iter_per_epoch .................................. 1250
  kd .............................................. False
  kd_alpha_ce ..................................... 1
  kd_beta_ce ...................................... 1
  kd_temp ......................................... 1.0
  kv_channels ..................................... 128
  layernorm_epsilon ............................... 1e-05
  lazy_mpu_init ................................... None
  load ............................................ ./tmp
  load_teacher .................................... None
  local_rank ...................................... None
  log_batch_size_to_tensorboard ................... False
  log_interval .................................... 1
  log_learning_rate_to_tensorboard ................ True
  log_loss_scale_to_tensorboard ................... True
  log_memory_to_tensorboard ....................... False
  log_num_zeros_in_grad ........................... False
  log_optimizer_states_to_tensorboard ............. False
  log_params_norm ................................. False
  log_timers_to_tensorboard ....................... False
  log_validation_ppl_to_tensorboard ............... False
  log_world_size_to_tensorboard ................... False
  loss_scale ...................................... None
  loss_scale_window ............................... 1000
  lr .............................................. 0.0003
  lr_decay_iters .................................. None
  lr_decay_samples ................................ None
  lr_decay_style .................................. cosine
  lr_decay_tokens ................................. None
  lr_warmup_fraction .............................. None
  lr_warmup_iters ................................. 2000
  lr_warmup_samples ............................... 0
  lr_warmup_tokens ................................ None
  make_vocab_size_divisible_by .................... 128
  mask_factor ..................................... 1.0
  mask_prob ....................................... 0.15
  mask_type ....................................... random
  masked_softmax_fusion ........................... True
  max_position_embeddings ......................... 2048
  max_tokens_to_oom ............................... 12000
  memory_centric_tiled_linear ..................... False
  merge_file ...................................... None
  micro_batch_size ................................ 4
  min_loss_scale .................................. 1.0
  min_lr .......................................... 3e-05
  mlp_type ........................................ standard
  mmap_warmup ..................................... False
  moe_eval_capacity_factor ........................ 1.0
  moe_expert_parallel_size ........................ 1
  moe_loss_coeff .................................. 0.1
  moe_min_capacity ................................ 4
  moe_token_dropping .............................. True
  moe_train_capacity_factor ....................... 1.0
  mos ............................................. False
  no_load_lr_state ................................ False
  no_load_optim ................................... None
  no_load_rng ..................................... None
  no_persist_layer_norm ........................... False
  no_pipeline_parallel ............................ False
  no_save_optim ................................... None
  no_save_rng ..................................... None
  normalization ................................... rmsnorm
  num_attention_heads ............................. 16
  num_attention_heads_teacher ..................... None
  num_channels .................................... 3
  num_classes ..................................... 1000
  num_experts ..................................... [1]
  num_experts_switch .............................. None
  num_experts_teacher ............................. [1]
  num_key_value_heads ............................. 16
  num_layers ...................................... 24
  num_layers_per_virtual_pipeline_stage ........... None
  num_layers_teacher .............................. None
  num_workers ..................................... 2
  onnx_safe ....................................... None
  openai_gelu ..................................... False
  optimizer ....................................... adam
  output_bert_embeddings .......................... False
  overlap_p2p_comm ................................ False
  override_opt_param_scheduler .................... False
  params_dtype .................................... torch.float16
  partition_activations ........................... False
  patch_dim ....................................... 16
  perform_initialization .......................... True
  pipeline_model_parallel_size .................... 2
  pipeline_model_parallel_split_rank .............. None
  profile_backward ................................ False
  query_in_block_prob ............................. 0.1
  rampup_batch_size ............................... None
  random_ltd ...................................... False
  rank ............................................ 0
  recompute_granularity ........................... None
  recompute_method ................................ None
  recompute_num_layers ............................ 1
  remote_device ................................... none
  reset_attention_mask ............................ False
  reset_iteration ................................. False
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
  return_data_index ............................... False
  rotary_percent .................................. 1.0
  sample_rate ..................................... 1.0
  save ............................................ ./tmp
  save_interval ................................... 10000
  scatter_gather_tensors_in_pipeline .............. True
  scattered_embeddings ............................ False
  seed ............................................ 1234
  seq_length ...................................... 2048
  sequence_parallel ............................... False
  sgd_momentum .................................... 0.9
  short_seq_prob .................................. 0.1
  skip_train ...................................... False
  split ........................................... 900,50,50
  split_transformers .............................. False
  squared_relu .................................... False
  standalone_embedding_stage ...................... False
  start_weight_decay .............................. 0.1
  swiglu .......................................... True
  swin_backbone_type .............................. tiny
  synchronize_each_layer .......................... False
  tensor_model_parallel_size ...................... 2
  tensorboard_dir ................................. None
  tensorboard_log_interval ........................ 1
  tensorboard_queue_size .......................... 1000
  test_data_path .................................. None
  tile_factor ..................................... 1
  timing_log_level ................................ 0
  timing_log_option ............................... minmax
  titles_data_path ................................ None
  tokenizer_model ................................. /workspace/model/llama-tokenizer/tokenizer.model
  tokenizer_type .................................. GPTSentencePieceTokenizer
  topk ............................................ 1
  train_data_exact_num_epochs ..................... None
  train_data_path ................................. None
  train_desc_path ................................. None
  train_doc_idx_path .............................. None
  train_idx_path .................................. None
  train_iters ..................................... 2500
  train_sample_idx_path ........................... None
  train_samples ................................... None
  train_shuffle_idx_path .......................... None
  train_tokens .................................... None
  transformer_impl ................................ local
  transformer_pipeline_model_parallel_size ........ 2
  untie_embeddings_and_output_weights ............. True
  use_checkpoint_args ............................. False
  use_checkpoint_opt_param_scheduler .............. False
  use_contiguous_buffers_in_local_ddp ............. True
  use_cpu_initialization .......................... None
  use_distributed_optimizer ....................... False
  use_flash_attn .................................. False
  use_one_sent_docs ............................... False
  use_pin_memory .................................. False
  use_ring_exchange_p2p ........................... False
  use_rotary_position_embeddings .................. True
  use_tutel ....................................... False
  valid_data_path ................................. None
  variable_seq_lengths ............................ False
  virtual_pipeline_model_parallel_size ............ None
  vision_backbone_type ............................ vit
  vision_pretraining .............................. False
  vision_pretraining_type ......................... classify
  vocab_extra_ids ................................. 0
  vocab_file ...................................... None
  vocab_size ...................................... None
  weight_decay .................................... 0.1
  weight_decay_incr_style ......................... constant
  world_size ...................................... 8
  zero_allgather_bucket_size ...................... 0.0
  zero_contigious_gradients ....................... False
  zero_reduce_bucket_size ......................... 0.0
  zero_reduce_scatter ............................. False
  zero_stage ...................................... 0
-------------------- end of arguments ---------------------
setting number of micro-batches to constant 4
> building GPTSentencePieceTokenizer tokenizer ...
> padded vocab (size: 32000) with 0 dummy tokens (new size: 32000)
> initializing torch distributed ...
...
> initialized tensor model parallel with size 2
> initialized pipeline model parallel with size 2
> setting random seeds to 1234 ...
> initializing model parallel cuda seeds on global rank 0, model parallel rank 0, and data parallel rank 0 with model parallel seed: 3952 and data parallel seed: 1234
> compiling dataset index builder ...
make: Entering directory '/workspace/code/Megatron-DeepSpeed-llama-20230815/megatron/data'
make: Nothing to be done for 'default'.
make: Leaving directory '/workspace/code/Megatron-DeepSpeed-llama-20230815/megatron/data'
>>> done with dataset index builder. Compilation time: 0.211 seconds
> compiling and loading fused kernels ...
Detected CUDA files, patching ldflags
Emitting ninja build file /workspace/code/Megatron-DeepSpeed-llama-20230815/megatron/fused_kernels/build/build.ninja...
Building extension module scaled_upper_triang_masked_softmax_cuda...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
ninja: no work to do.
Loading extension module scaled_upper_triang_masked_softmax_cuda...
Detected CUDA files, patching ldflags
Emitting ninja build file /workspace/code/Megatron-DeepSpeed-llama-20230815/megatron/fused_kernels/build/build.ninja...
Building extension module scaled_masked_softmax_cuda...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
ninja: no work to do.
Loading extension module scaled_masked_softmax_cuda...
Detected CUDA files, patching ldflags
Emitting ninja build file /workspace/code/Megatron-DeepSpeed-llama-20230815/megatron/fused_kernels/build/build.ninja...
Building extension module scaled_softmax_cuda...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
ninja: no work to do.
Loading extension module scaled_softmax_cuda...
>>> done with compiling and loading fused kernels. Compilation time: 5.323 seconds
time to initialize megatron (seconds): 9.579
[after megatron is initialized] datetime: 2023-08-15 06:31:34
building GPT model ...
[2023-08-15 06:31:34,678] [INFO] [utils.py:785:see_memory_usage] Before Building Model
[2023-08-15 06:31:34,679] [INFO] [utils.py:786:see_memory_usage] MA 0.0 GB         Max_MA 0.73 GB         CA 0.0 GB         Max_CA 1 GB
[2023-08-15 06:31:34,680] [INFO] [utils.py:793:see_memory_usage] CPU Virtual Memory:  used = 53.28 GB, percent = 5.3%
SEED_LAYERS=False BASE_SEED=1234 SEED_FN=None
Using topology: {ProcessCoord(pipe=0, data=0, model=0): 0, ProcessCoord(pipe=0, data=0, model=1): 1, ProcessCoord(pipe=0, data=1, model=0): 2, ProcessCoord(pipe=0, data=1, model=1): 3, ProcessCoord(pipe=1, data=0, model=0): 4, ProcessCoord(pipe=1, data=0, model=1): 5, ProcessCoord(pipe=1, data=1, model=0): 6, ProcessCoord(pipe=1, data=1, model=1): 7}
[2023-08-15 06:31:34,817] [INFO] [module.py:358:_partition_layers] Partitioning pipeline stages with method type:transformer
stage=0 layers=14
     0: _to_float16
     1: EmbeddingPipe
     2: ParallelTransformerLayerPipe
     3: ParallelTransformerLayerPipe
     4: ParallelTransformerLayerPipe
     5: ParallelTransformerLayerPipe
     6: ParallelTransformerLayerPipe
     7: ParallelTransformerLayerPipe
     8: ParallelTransformerLayerPipe
     9: ParallelTransformerLayerPipe
    10: ParallelTransformerLayerPipe
    11: ParallelTransformerLayerPipe
    12: ParallelTransformerLayerPipe
    13: ParallelTransformerLayerPipe
stage=1 layers=15
    14: ParallelTransformerLayerPipe
    15: ParallelTransformerLayerPipe
    16: ParallelTransformerLayerPipe
    17: ParallelTransformerLayerPipe
    18: ParallelTransformerLayerPipe
    19: ParallelTransformerLayerPipe
    20: ParallelTransformerLayerPipe
    21: ParallelTransformerLayerPipe
    22: ParallelTransformerLayerPipe
    23: ParallelTransformerLayerPipe
    24: ParallelTransformerLayerPipe
    25: ParallelTransformerLayerPipe
    26: MixedFusedRMSNorm
    27: LMHeadPipe
    28: float16_to_fp32
  loss: CrossEntropy
 > number of parameters on (tensor, pipeline) model parallel rank (0, 1): 336381952
 > number of parameters on (tensor, pipeline) model parallel rank (1, 0): 336379904
 > number of parameters on (tensor, pipeline) model parallel rank (1, 1): 336381952
[2023-08-15 06:31:34,957] [INFO] [utils.py:785:see_memory_usage] After Building Model
[2023-08-15 06:31:34,958] [INFO] [utils.py:786:see_memory_usage] MA 0.65 GB         Max_MA 0.66 GB         CA 0.68 GB         Max_CA 1 GB
[2023-08-15 06:31:34,958] [INFO] [utils.py:793:see_memory_usage] CPU Virtual Memory:  used = 53.4 GB, percent = 5.3%
 > number of parameters on (tensor, pipeline) model parallel rank (0, 0): 336379904
> learning rate decay style: cosine
DeepSpeed is enabled.
[2023-08-15 06:31:34,960] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed info: version=0.9.5, git-hash=unknown, git-branch=unknown
[2023-08-15 06:31:35,566] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed Flops Profiler Enabled: False
[2023-08-15 06:31:35,566] [INFO] [logging.py:96:log_dist] [Rank 0] Removing param_group that has no 'params' in the client Optimizer
[2023-08-15 06:31:35,566] [INFO] [logging.py:96:log_dist] [Rank 0] Using client Optimizer as basic optimizer
[2023-08-15 06:31:35,567] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed Basic Optimizer = FusedAdam
[2023-08-15 06:31:35,567] [INFO] [logging.py:96:log_dist] [Rank 0] Creating fp16 optimizer with dynamic loss scale
[2023-08-15 06:31:35,631] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed Final Optimizer = FusedAdam
[2023-08-15 06:31:35,631] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed using client LR scheduler
[2023-08-15 06:31:35,631] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed LR Scheduler = <megatron.optimizer_param_scheduler.OptimizerParamScheduler object at 0x7ff39c161e80>
[2023-08-15 06:31:35,631] [INFO] [logging.py:96:log_dist] [Rank 0] step=0, skipped=0, lr=[0.0, 0.0], mom=[(0.9, 0.95), (0.9, 0.95)]
[2023-08-15 06:31:35,632] [INFO] [config.py:960:print] DeepSpeedEngine configuration:
[2023-08-15 06:31:35,632] [INFO] [config.py:964:print]   activation_checkpointing_config  {
    "partition_activations": false,
    "contiguous_memory_optimization": false,
    "cpu_checkpointing": false,
    "number_checkpoints": null,
    "synchronize_checkpoint_boundary": false,
    "profile": false
}
[2023-08-15 06:31:35,632] [INFO] [config.py:964:print]   aio_config ................... {'block_size': 1048576, 'queue_depth': 8, 'thread_count': 1, 'single_submit': False, 'overlap_events': True}
[2023-08-15 06:31:35,632] [INFO] [config.py:964:print]   amp_enabled .................. False
[2023-08-15 06:31:35,632] [INFO] [config.py:964:print]   amp_params ................... False
[2023-08-15 06:31:35,632] [INFO] [config.py:964:print]   autotuning_config ............ {
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
[2023-08-15 06:31:35,633] [INFO] [config.py:964:print]   bfloat16_enabled ............. False
[2023-08-15 06:31:35,633] [INFO] [config.py:964:print]   checkpoint_parallel_write_pipeline  False
[2023-08-15 06:31:35,633] [INFO] [config.py:964:print]   checkpoint_tag_validation_enabled  True
[2023-08-15 06:31:35,633] [INFO] [config.py:964:print]   checkpoint_tag_validation_fail  False
[2023-08-15 06:31:35,633] [INFO] [config.py:964:print]   comms_config ................. <deepspeed.comm.config.DeepSpeedCommsConfig object at 0x7ff348042a00>
[2023-08-15 06:31:35,633] [INFO] [config.py:964:print]   communication_data_type ...... None
[2023-08-15 06:31:35,633] [INFO] [config.py:964:print]   compression_config ........... {'weight_quantization': {'shared_parameters': {'enabled': False, 'quantizer_kernel': False, 'schedule_offset': 0, 'quantize_groups': 1, 'quantize_verbose': False, 'quantization_type': 'symmetric', 'quantize_weight_in_forward': False, 'rounding': 'nearest', 'fp16_mixed_quantize': False, 'quantize_change_ratio': 0.001}, 'different_groups': {}}, 'activation_quantization': {'shared_parameters': {'enabled': False, 'quantization_type': 'symmetric', 'range_calibration': 'dynamic', 'schedule_offset': 1000}, 'different_groups': {}}, 'sparse_pruning': {'shared_parameters': {'enabled': False, 'method': 'l1', 'schedule_offset': 1000}, 'different_groups': {}}, 'row_pruning': {'shared_parameters': {'enabled': False, 'method': 'l1', 'schedule_offset': 1000}, 'different_groups': {}}, 'head_pruning': {'shared_parameters': {'enabled': False, 'method': 'topk', 'schedule_offset': 1000}, 'different_groups': {}}, 'channel_pruning': {'shared_parameters': {'enabled': False, 'method': 'l1', 'schedule_offset': 1000}, 'different_groups': {}}, 'layer_reduction': {'enabled': False}}
[2023-08-15 06:31:35,633] [INFO] [config.py:964:print]   curriculum_enabled_legacy .... False
[2023-08-15 06:31:35,633] [INFO] [config.py:964:print]   curriculum_params_legacy ..... False
[2023-08-15 06:31:35,633] [INFO] [config.py:964:print]   data_efficiency_config ....... {'enabled': False, 'seed': 1234, 'data_sampling': {'enabled': False, 'num_epochs': 1000, 'num_workers': 0, 'curriculum_learning': {'enabled': False}}, 'data_routing': {'enabled': False, 'random_ltd': {'enabled': False, 'layer_token_lr_schedule': {'enabled': False}}}}
[2023-08-15 06:31:35,633] [INFO] [config.py:964:print]   data_efficiency_enabled ...... False
[2023-08-15 06:31:35,633] [INFO] [config.py:964:print]   dataloader_drop_last ......... False
[2023-08-15 06:31:35,633] [INFO] [config.py:964:print]   disable_allgather ............ False
[2023-08-15 06:31:35,633] [INFO] [config.py:964:print]   dump_state ................... False
[2023-08-15 06:31:35,633] [INFO] [config.py:964:print]   dynamic_loss_scale_args ...... {'init_scale': 65536, 'scale_window': 1000, 'delayed_shift': 2, 'consecutive_hysteresis': False, 'min_scale': 1}
[2023-08-15 06:31:35,633] [INFO] [config.py:964:print]   eigenvalue_enabled ........... False
[2023-08-15 06:31:35,633] [INFO] [config.py:964:print]   eigenvalue_gas_boundary_resolution  1
[2023-08-15 06:31:35,633] [INFO] [config.py:964:print]   eigenvalue_layer_name ........ bert.encoder.layer
[2023-08-15 06:31:35,633] [INFO] [config.py:964:print]   eigenvalue_layer_num ......... 0
[2023-08-15 06:31:35,633] [INFO] [config.py:964:print]   eigenvalue_max_iter .......... 100
[2023-08-15 06:31:35,633] [INFO] [config.py:964:print]   eigenvalue_stability ......... 1e-06
[2023-08-15 06:31:35,633] [INFO] [config.py:964:print]   eigenvalue_tol ............... 0.01
[2023-08-15 06:31:35,633] [INFO] [config.py:964:print]   eigenvalue_verbose ........... False
[2023-08-15 06:31:35,633] [INFO] [config.py:964:print]   elasticity_enabled ........... False
[2023-08-15 06:31:35,633] [INFO] [config.py:964:print]   flops_profiler_config ........ {
    "enabled": false,
    "recompute_fwd_factor": 0.0,
    "profile_step": 1,
    "module_depth": -1,
    "top_modules": 1,
    "detailed": true,
    "output_file": null
}
[2023-08-15 06:31:35,633] [INFO] [config.py:964:print]   fp16_auto_cast ............... False
[2023-08-15 06:31:35,633] [INFO] [config.py:964:print]   fp16_enabled ................. True
[2023-08-15 06:31:35,633] [INFO] [config.py:964:print]   fp16_master_weights_and_gradients  False
[2023-08-15 06:31:35,633] [INFO] [config.py:964:print]   global_rank .................. 0
[2023-08-15 06:31:35,633] [INFO] [config.py:964:print]   grad_accum_dtype ............. None
[2023-08-15 06:31:35,633] [INFO] [config.py:964:print]   gradient_accumulation_steps .. 4
[2023-08-15 06:31:35,633] [INFO] [config.py:964:print]   gradient_clipping ............ 0.0
[2023-08-15 06:31:35,633] [INFO] [config.py:964:print]   gradient_predivide_factor .... 1.0
[2023-08-15 06:31:35,634] [INFO] [config.py:964:print]   hybrid_engine ................ enabled=False max_out_tokens=512 inference_tp_size=1 release_inference_cache=False pin_parameters=True tp_gather_partition_size=8
[2023-08-15 06:31:35,634] [INFO] [config.py:964:print]   initial_dynamic_scale ........ 65536
[2023-08-15 06:31:35,634] [INFO] [config.py:964:print]   load_universal_checkpoint .... False
[2023-08-15 06:31:35,634] [INFO] [config.py:964:print]   loss_scale ................... 0
[2023-08-15 06:31:35,634] [INFO] [config.py:964:print]   memory_breakdown ............. False
[2023-08-15 06:31:35,634] [INFO] [config.py:964:print]   mics_hierarchial_params_gather  False
[2023-08-15 06:31:35,634] [INFO] [config.py:964:print]   mics_shard_size .............. -1
[2023-08-15 06:31:35,634] [INFO] [config.py:964:print]   monitor_config ............... tensorboard=TensorBoardConfig(enabled=False, output_path='', job_name='DeepSpeedJobName') wandb=WandbConfig(enabled=False, group=None, team=None, project='deepspeed') csv_monitor=CSVConfig(enabled=False, output_path='', job_name='DeepSpeedJobName') enabled=False
[2023-08-15 06:31:35,634] [INFO] [config.py:964:print]   nebula_config ................ {
    "enabled": false,
    "persistent_storage_path": null,
    "persistent_time_interval": 100,
    "num_of_version_in_retention": 2,
    "enable_nebula_load": true,
    "load_path": null
}
[2023-08-15 06:31:35,634] [INFO] [config.py:964:print]   optimizer_legacy_fusion ...... False
[2023-08-15 06:31:35,634] [INFO] [config.py:964:print]   optimizer_name ............... None
[2023-08-15 06:31:35,634] [INFO] [config.py:964:print]   optimizer_params ............. None
[2023-08-15 06:31:35,634] [INFO] [config.py:964:print]   pipeline ..................... {'stages': 'auto', 'partition': 'best', 'seed_layers': False, 'activation_checkpoint_interval': 0}
[2023-08-15 06:31:35,634] [INFO] [config.py:964:print]   pld_enabled .................. False
[2023-08-15 06:31:35,634] [INFO] [config.py:964:print]   pld_params ................... False
[2023-08-15 06:31:35,634] [INFO] [config.py:964:print]   prescale_gradients ........... False
[2023-08-15 06:31:35,634] [INFO] [config.py:964:print]   scheduler_name ............... None
[2023-08-15 06:31:35,634] [INFO] [config.py:964:print]   scheduler_params ............. None
[2023-08-15 06:31:35,634] [INFO] [config.py:964:print]   sparse_attention ............. None
[2023-08-15 06:31:35,634] [INFO] [config.py:964:print]   sparse_gradients_enabled ..... False
[2023-08-15 06:31:35,634] [INFO] [config.py:964:print]   steps_per_print .............. 1
[2023-08-15 06:31:35,634] [INFO] [config.py:964:print]   train_batch_size ............. 32
[2023-08-15 06:31:35,634] [INFO] [config.py:964:print]   train_micro_batch_size_per_gpu  4
[2023-08-15 06:31:35,634] [INFO] [config.py:964:print]   use_node_local_storage ....... False
[2023-08-15 06:31:35,634] [INFO] [config.py:964:print]   wall_clock_breakdown ......... False
[2023-08-15 06:31:35,634] [INFO] [config.py:964:print]   world_size ................... 2
[2023-08-15 06:31:35,634] [INFO] [config.py:964:print]   zero_allow_untested_optimizer  False
[2023-08-15 06:31:35,634] [INFO] [config.py:964:print]   zero_config .................. stage=0 contiguous_gradients=True reduce_scatter=True reduce_bucket_size=500,000,000 allgather_partitions=True allgather_bucket_size=500,000,000 overlap_comm=False load_from_fp32_weights=True elastic_checkpoint=False offload_param=None offload_optimizer=None sub_group_size=1,000,000,000 cpu_offload_param=None cpu_offload_use_pin_memory=None cpu_offload=None prefetch_bucket_size=50,000,000 param_persistence_threshold=100,000 model_persistence_threshold=sys.maxsize max_live_parameters=1,000,000,000 max_reuse_distance=1,000,000,000 gather_16bit_weights_on_model_save=False stage3_gather_fp16_weights_on_model_save=False ignore_unused_parameters=True legacy_stage1=False round_robin_gradients=False mics_shard_size=-1 mics_hierarchical_params_gather=False memory_efficient_linear=True
[2023-08-15 06:31:35,634] [INFO] [config.py:964:print]   zero_enabled ................. False
[2023-08-15 06:31:35,634] [INFO] [config.py:964:print]   zero_force_ds_cpu_optimizer .. True
[2023-08-15 06:31:35,634] [INFO] [config.py:964:print]   zero_optimization_stage ...... 0
[2023-08-15 06:31:35,634] [INFO] [config.py:950:print_user_config]   json = {
    "train_batch_size": 32,
    "train_micro_batch_size_per_gpu": 4,
    "steps_per_print": 1,
    "zero_optimization": {
        "stage": 0
    },
    "fp16": {
        "enabled": true,
        "auto_cast": false,
        "loss_scale": 0,
        "initial_scale_power": 16,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "consecutive_hysteresis": false,
        "min_loss_scale": 1
    }
}
[2023-08-15 06:31:35,634] [INFO] [engine.py:83:__init__] CONFIG: micro_batches=4 micro_batch_size=4
[W ProcessGroupNCCL.cpp:1692] Warning: 0NCCL_AVOID_RECORD_STREAMS=1 has no effect for point-to-point collectives. (function operator())
[W ProcessGroupNCCL.cpp:1692] Warning: 0NCCL_AVOID_RECORD_STREAMS=1 has no effect for point-to-point collectives. (function operator())
[W ProcessGroupNCCL.cpp:1692] Warning: 0NCCL_AVOID_RECORD_STREAMS=1 has no effect for point-to-point collectives. (function operator())
[W ProcessGroupNCCL.cpp:1692] Warning: 0NCCL_AVOID_RECORD_STREAMS=1 has no effect for point-to-point collectives. (function operator())
[2023-08-15 06:31:36,508] [INFO] [engine.py:138:__init__] RANK=0 STAGE=0 LAYERS=14 [0, 14) STAGE_PARAMS=336379904 (336.380M) TOTAL_PARAMS=1345523712 (1345.524M) UNIQUE_PARAMS=1345523712 (1345.524M)
[2023-08-15 06:31:36,508] [INFO] [engine.py:138:__init__] RANK=1 STAGE=0 LAYERS=14 [0, 14) STAGE_PARAMS=336379904 (336.380M) TOTAL_PARAMS=1345523712 (1345.524M) UNIQUE_PARAMS=1345523712 (1345.524M)
[2023-08-15 06:31:36,508] [INFO] [engine.py:138:__init__] RANK=4 STAGE=1 LAYERS=15 [14, 29) STAGE_PARAMS=336381952 (336.382M) TOTAL_PARAMS=1345523712 (1345.524M) UNIQUE_PARAMS=1345523712 (1345.524M)
[2023-08-15 06:31:36,509] [INFO] [engine.py:138:__init__] RANK=5 STAGE=1 LAYERS=15 [14, 29) STAGE_PARAMS=336381952 (336.382M) TOTAL_PARAMS=1345523712 (1345.524M) UNIQUE_PARAMS=1345523712 (1345.524M)
...
(min, max) time across ranks (ms):
    load-checkpoint ................................: (1.04, 1.37)
[after model, optimizer, and learning rate scheduler are built] datetime: 2023-08-15 06:31:36
> building train, validation, and test datasets ...
 > datasets target sizes (minimum size):
    train:      80000
    validation: 960
    test:       320
> building train, validation, and test datasets for GPT ...
Single data path provided for train, valid & test
 > building dataset index ...
    reading sizes...
    reading pointers...
    reading document index...
    creating numpy buffer of mmap...
    creating memory view of numpy buffer...
 > finished creating indexed dataset in 0.000625 seconds
    number of documents: 2456
 > dataset split:
    train:
     document indices in [0, 2210) total of 2210 documents
    validation:
     document indices in [2210, 2333) total of 123 documents
    test:
     document indices in [2333, 2456) total of 123 documents
...
 > loading doc-idx mapping from /workspace/data/gpt2-data/index-cache/adec68d4b8dc51ef702dcac7f462e93b_doc_idx.npy
 > loading sample-idx mapping from /workspace/data/gpt2-data/index-cache/adec68d4b8dc51ef702dcac7f462e93b_sample_idx.npy
 > loading shuffle-idx mapping from /workspace/data/gpt2-data/index-cache/adec68d4b8dc51ef702dcac7f462e93b_shuffle_idx.npy
    loaded indexed file in 0.002 seconds
    total number of samples: 113
    total number of epochs: 9
 > building dataset index ...
    reading sizes...
    reading pointers...
    reading document index...
    creating numpy buffer of mmap...
    creating memory view of numpy buffer...
 > finished creating indexed dataset in 0.000503 seconds
    number of documents: 2456
 > dataset split:
    train:
     document indices in [0, 2210) total of 2210 documents
    validation:
     document indices in [2210, 2333) total of 123 documents
    test:
     document indices in [2333, 2456) total of 123 documents
 > loading doc-idx mapping from /workspace/data/gpt2-data/index-cache/1a4c392debaeae4aaefe3a720974d6cf_doc_idx.npy
 > loading sample-idx mapping from /workspace/data/gpt2-data/index-cache/1a4c392debaeae4aaefe3a720974d6cf_sample_idx.npy
 > loading shuffle-idx mapping from /workspace/data/gpt2-data/index-cache/1a4c392debaeae4aaefe3a720974d6cf_shuffle_idx.npy
    loaded indexed file in 0.002 seconds
    total number of samples: 53810
    total number of epochs: 191
...
> building indices for blendable datasets ...
 > sample ratios:
   dataset 0, input: 0.333333, achieved: 0.333333
   dataset 1, input: 0.666667, achieved: 0.666667
> elapsed time for building blendable dataset indices: 0.00 (sec)
> size of blendable dataset: 80400 samples
> building indices for blendable datasets ...
 > sample ratios:
   dataset 0, input: 0.333333, achieved: 0.333333
   dataset 1, input: 0.666667, achieved: 0.666667
> elapsed time for building blendable dataset indices: 0.00 (sec)
> size of blendable dataset: 966 samples
> building indices for blendable datasets ...
 > sample ratios:
   dataset 0, input: 0.333333, achieved: 0.334365
   dataset 1, input: 0.666667, achieved: 0.665635
> elapsed time for building blendable dataset indices: 0.00 (sec)
> size of blendable dataset: 323 samples
> finished creating GPT datasets ...
[after dataloaders are built] datetime: 2023-08-15 06:31:38
done with setup ...
(min, max) time across ranks (ms):
    model-and-optimizer-setup ......................: (2449.71, 2515.91)
    train/valid/test-data-iterators-setup ..........: (1430.08, 1704.59)
training ...
[before the start of training step] datetime: 2023-08-15 06:31:38
...
[2023-08-15 06:31:42,452] [INFO] [logging.py:96:log_dist] [Rank 0] step=1, skipped=1, lr=[0.0, 0.0], mom=[(0.9, 0.95), (0.9, 0.95)]
[2023-08-15 06:31:42,452] [INFO] [fused_optimizer.py:363:_update_scale] Reducing dynamic loss scale from 65536 to 32768.0
steps: 1 loss: 10.6554 iter time (s): 4.218 samples/sec: 7.586
 iteration        1/    2500 | consumed samples:           32 | consumed tokens:        65536 | elapsed time per iteration (ms): 4220.8 | learning rate: 0.000E+00 | global batch size:    32 | lm loss: 1.065542E+01 | loss scale: 32768.0 | grad norm: 0.000 | num zeros: 0.0 | actual seqlen:  2048 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 7.582 | TFLOPs: 16.98 |
[2023-08-15 06:31:44,231] [INFO] [logging.py:96:log_dist] [Rank 0] step=2, skipped=1, lr=[1.5e-07, 1.5e-07], mom=[(0.9, 0.95), (0.9, 0.95)]
steps: 2 loss: 10.6521 iter time (s): 1.337 samples/sec: 23.934
 iteration        2/    2500 | consumed samples:           64 | consumed tokens:       131072 | elapsed time per iteration (ms): 1346.3 | learning rate: 1.500E-07 | global batch size:    32 | lm loss: 1.065210E+01 | loss scale: 32768.0 | grad norm: 14.113 | num zeros: 0.0 | actual seqlen:  2048 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 23.769 | TFLOPs: 53.25 |
[2023-08-15 06:31:45,481] [INFO] [logging.py:96:log_dist] [Rank 0] step=3, skipped=1, lr=[3e-07, 3e-07], mom=[(0.9, 0.95), (0.9, 0.95)]
steps: 3 loss: 10.6457 iter time (s): 1.246 samples/sec: 25.683
 iteration        3/    2500 | consumed samples:           96 | consumed tokens:       196608 | elapsed time per iteration (ms): 1250.2 | learning rate: 3.000E-07 | global batch size:    32 | lm loss: 1.064566E+01 | loss scale: 32768.0 | grad norm: 14.740 | num zeros: 0.0 | actual seqlen:  2048 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 25.596 | TFLOPs: 57.34 |
...
steps: 2498 loss: 12.6747 iter time (s): 1.206 samples/sec: 26.538
 iteration     2498/    2500 | consumed samples:        79936 | consumed tokens:    163708928 | elapsed time per iteration (ms): 1210.3 | learning rate: 1.371E-04 | global batch size:    32 | lm loss: 1.267467E+01 | loss scale: 1.0 | grad norm: 2.265 | num zeros: 0.0 | actual seqlen:  2048 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 26.440 | TFLOPs: 59.23 |
[2023-08-15 07:22:30,556] [INFO] [logging.py:96:log_dist] [Rank 0] Overflow detected. Skipping step. Attempted loss scale: 1, reducing to 1
[2023-08-15 07:22:30,556] [INFO] [logging.py:96:log_dist] [Rank 0] step=2499, skipped=1585, lr=[0.0001371, 0.0001371], mom=[(0.9, 0.95), (0.9, 0.95)]
steps: 2499 loss: 13.4013 iter time (s): 1.202 samples/sec: 26.614
 iteration     2499/    2500 | consumed samples:        79968 | consumed tokens:    163774464 | elapsed time per iteration (ms): 1205.4 | learning rate: 1.371E-04 | global batch size:    32 | lm loss: 1.340131E+01 | loss scale: 1.0 | grad norm: 2.265 | num zeros: 0.0 | actual seqlen:  2048 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 26.546 | TFLOPs: 59.47 |
[2023-08-15 07:22:31,763] [INFO] [fused_optimizer.py:363:_update_scale] Reducing dynamic loss scale from 1 to 1
...
[2023-08-15 07:22:31,763] [INFO] [fused_optimizer.py:363:_update_scale] Reducing dynamic loss scale from 1 to 1
[2023-08-15 07:22:31,763] [INFO] [logging.py:96:log_dist] [Rank 0] Overflow detected. Skipping step. Attempted loss scale: 1, reducing to 1
[2023-08-15 07:22:31,764] [INFO] [logging.py:96:log_dist] [Rank 0] step=2500, skipped=1586, lr=[0.0001371, 0.0001371], mom=[(0.9, 0.95), (0.9, 0.95)]
steps: 2500 loss: 12.5495 iter time (s): 1.200 samples/sec: 26.671
 iteration     2500/    2500 | consumed samples:        80000 | consumed tokens:    163840000 | elapsed time per iteration (ms): 1206.4 | learning rate: 1.371E-04 | global batch size:    32 | lm loss: 1.254951E+01 | loss scale: 1.0 | grad norm: 2.265 | num zeros: 0.0 | actual seqlen:  2048 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 26.526 | TFLOPs: 59.42 |
[after training is done] datetime: 2023-08-15 07:22:31
saving checkpoint at iteration    2500 to ./tmp
[2023-08-15 07:22:31,767] [INFO] [logging.py:96:log_dist] [Rank 0] [Torch] Checkpoint global_step2500 is about to be saved!
[2023-08-15 07:22:31,774] [INFO] [torch_checkpoint_engine.py:21:save] [Torch] Saving ./tmp/global_step2500/layer_14-model_01-model_states.pt...
[2023-08-15 07:22:31,774] [INFO] [torch_checkpoint_engine.py:21:save] [Torch] Saving ./tmp/global_step2500/layer_01-model_01-model_states.pt...
[2023-08-15 07:22:31,774] [INFO] [torch_checkpoint_engine.py:33:commit] [Torch] Checkpoint global_step2500 is ready now!
[2023-08-15 07:22:31,774] [INFO] [torch_checkpoint_engine.py:33:commit] [Torch] Checkpoint global_step2500 is ready now!
[2023-08-15 07:22:31,775] [INFO] [torch_checkpoint_engine.py:21:save] [Torch] Saving ./tmp/global_step2500/layer_14-model_00-model_states.pt...
...
[2023-08-15 07:22:32,255] [INFO] [torch_checkpoint_engine.py:21:save] [Torch] Saving ./tmp/global_step2500/layer_27-model_00-model_states.pt...
[2023-08-15 07:22:32,273] [INFO] [torch_checkpoint_engine.py:21:save] [Torch] Saving ./tmp/global_step2500/layer_26-model_01-model_states.pt...
[2023-08-15 07:22:32,273] [INFO] [torch_checkpoint_engine.py:23:save] [Torch] Saved ./tmp/global_step2500/layer_26-model_01-model_states.pt.
[2023-08-15 07:22:32,273] [INFO] [torch_checkpoint_engine.py:21:save] [Torch] Saving ./tmp/global_step2500/layer_27-model_01-model_states.pt...
...
[2023-08-15 07:22:32,448] [INFO] [torch_checkpoint_engine.py:21:save] [Torch] Saving ./tmp/global_step2500/mp_rank_00_model_states.pt...
[2023-08-15 07:22:35,824] [INFO] [torch_checkpoint_engine.py:23:save] [Torch] Saved ./tmp/global_step2500/mp_rank_02_model_states.pt.
[2023-08-15 07:22:35,824] [INFO] [torch_checkpoint_engine.py:33:commit] [Torch] Checkpoint global_step2500 is ready now!
[2023-08-15 07:22:35,987] [INFO] [torch_checkpoint_engine.py:23:save] [Torch] Saved ./tmp/global_step2500/mp_rank_01_model_states.pt.
[2023-08-15 07:22:35,988] [INFO] [torch_checkpoint_engine.py:33:commit] [Torch] Checkpoint global_step2500 is ready now!
[2023-08-15 07:22:36,380] [INFO] [torch_checkpoint_engine.py:23:save] [Torch] Saved ./tmp/global_step2500/mp_rank_03_model_states.pt.
[2023-08-15 07:22:36,380] [INFO] [torch_checkpoint_engine.py:33:commit] [Torch] Checkpoint global_step2500 is ready now!
[2023-08-15 07:22:37,720] [INFO] [torch_checkpoint_engine.py:23:save] [Torch] Saved ./tmp/global_step2500/mp_rank_00_model_states.pt.
[2023-08-15 07:22:37,720] [INFO] [torch_checkpoint_engine.py:33:commit] [Torch] Checkpoint global_step2500 is ready now!
  successfully saved checkpoint at iteration    2500 to ./tmp
Evaluating iter 1/10
...
Evaluating iter 10/10
Evaluating iter 1/10
---------------------------------------------------------------------------------------------------------------------------------------
 validation loss at iteration 2500 on 320-sample draw from validation set | lm loss value: 1.360287E+01 | lm loss PPL: 8.084497E+05 |
---------------------------------------------------------------------------------------------------------------------------------------
Evaluating iter 2/10
...
Evaluating iter 10/10
---------------------------------------------------------------------------------------------------------------------------------
 validation loss at iteration 2500 on 320-sample draw from test set | lm loss value: 1.495915E+01 | lm loss PPL: 3.138158E+06 |
---------------------------------------------------------------------------------------------------------------------------------
```

</p></details>




```
> ls -al -R --block-size=M tmp/

tmp/:
total 1M
drwxr-xr-x  3 root root 1M Aug 15 07:22 .
drwxr-xr-x 13 root root 1M Aug 15 06:07 ..
-rw-r--r--  1 root root 1M Aug 15 06:31 deepspeed.json
drwxr-xr-x  2 root root 1M Aug 15 07:22 global_step2500
-rw-r--r--  1 root root 1M Aug 15 07:22 latest
-rw-r--r--  1 root root 1M Aug 15 07:22 latest_checkpointed_iteration.txt

tmp/global_step2500:
total 17986M
drwxr-xr-x 2 root root    1M Aug 15 07:22 .
drwxr-xr-x 3 root root    1M Aug 15 07:22 ..
-rw-r--r-- 1 root root   63M Aug 15 07:22 layer_01-model_00-model_states.pt
-rw-r--r-- 1 root root   63M Aug 15 07:22 layer_01-model_01-model_states.pt
...
-rw-r--r-- 1 root root   63M Aug 15 07:22 layer_27-model_00-model_states.pt
-rw-r--r-- 1 root root   63M Aug 15 07:22 layer_27-model_01-model_states.pt
-rw-r--r-- 1 root root 3855M Aug 15 07:22 mp_rank_00_model_states.pt
-rw-r--r-- 1 root root 3855M Aug 15 07:22 mp_rank_01_model_states.pt
-rw-r--r-- 1 root root 3855M Aug 15 07:22 mp_rank_02_model_states.pt
-rw-r--r-- 1 root root 3855M Aug 15 07:22 mp_rank_03_model_states.pt
```








