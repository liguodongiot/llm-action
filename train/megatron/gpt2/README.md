
## GPT2 模型训练 

- megatron/tokenizer/file_utils.py
- tools/openwebtext/merge_data.py


## 权重

```
> tree -h megatron
megatron
├── [   8]  latest_checkpointed_iteration.txt
└── [4.0K]  release
    └── [4.0K]  mp_rank_00
        └── [677M]  model_optim_rng.pt

2 directories, 2 files
> cat megatron/latest_checkpointed_iteration.txt 
release
```




## 训练

### 单机单卡
```
#!/bin/bash

# Runs the "345M" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

#CHECKPOINT_PATH=<Specify path>
#VOCAB_FILE=<Specify path to file>/gpt2-vocab.json
#MERGE_FILE=<Specify path to file>/gpt2-merges.txt
#DATA_PATH=<Specify path and file prefix>_text_document

CHECKPOINT_PATH=/workspace/model/megatron-models/345m
VOCAB_FILE=/workspace/model/gpt2-vocab/gpt2-vocab.json
MERGE_FILE=/workspace/model/gpt2-vocab/gpt2-merges.txt
#DATA_PATH=/workspace/data/merged_cleand.json
DATA_PATH=/workspace/data/my-gpt2_text_document
MODEL_PATH=/workspace/model/megatron-models/output

GPT_ARGS="
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size 1 \
    --global-batch-size 2 \
    --lr 0.00015 \
    --train-iters 5000 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --data-impl mmap \
    --split 700,200,100
"

OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10
"

torchrun pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH
```


<details><summary>详细输出</summary><p>

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








