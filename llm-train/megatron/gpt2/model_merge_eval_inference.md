


## 权重合并
```
python tools/checkpoint_util.py \
        --model-type GPT \
        --load-dir /workspace/model/megatron-models/345m-init-mp\
        --save-dir /workspace/model/megatron-models/345m-init-mp-out \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 1
```


<details><summary>详细输出：</summary><p>


```
> tree -h /workspace/model/megatron-models/345m-init-mp-out
/workspace/model/megatron-models/345m-init-mp-out
├── [4.0K]  iter_0005000
│   └── [4.0K]  mp_rank_00
│       └── [1.3G]  model_optim_rng.pt
└── [   4]  latest_checkpointed_iteration.txt

2 directories, 2 files
```

</p></details>


<details><summary>详细输出：</summary><p>


```
> python tools/checkpoint_util.py         --model-type GPT         --load-dir /workspace/model/megatron-models/345m-init-mp        --save-dir /workspace/model/megatron-models/345m-init-mp-out         --target-tensor-parallel-size 1         --target-pipeline-parallel-size 1  
Loaded checkpoint_loader_megatron as the loader.
Loaded checkpoint_saver_megatron as the saver.
Starting saver...
Starting loader...
Namespace(DDP_impl='local', accumulate_allreduce_grads_in_fp32=False, adam_beta1=0.9, adam_beta2=0.999, adam_eps=1e-08, add_bias_linear=True, add_position_embedding=True, adlr_autoresume=False, adlr_autoresume_interval=1000, apply_layernorm_1p=False, apply_query_key_layer_scaling=True, apply_residual_connection_post_layernorm=False, async_tensor_model_parallel_allreduce=False, attention_dropout=0.1, attention_softmax_in_fp32=False, barrier_with_L1_time=True, batch_size=None, bert_binary_head=True, bert_embedder_type='megatron', bert_load=None, bf16=False, bias_dropout_fusion=False, bias_gelu_fusion=False, biencoder_projection_dim=0, biencoder_shared_query_context_model=False, block_data_path=None, checkpoint_activations=False, classes_fraction=1.0, clip_grad=1.0, data_cache_path=None, data_impl='infer', data_parallel_random_init=False, data_path=None, data_per_class_fraction=1.0, data_sharding=True, dataloader_type=None, decoder_num_layers=None, decoder_seq_length=None, dino_bottleneck_size=256, dino_freeze_last_layer=1, dino_head_hidden_size=2048, dino_local_crops_number=10, dino_local_img_size=96, dino_norm_last_layer=False, dino_teacher_temp=0.07, dino_warmup_teacher_temp=0.04, dino_warmup_teacher_temp_epochs=30, distribute_saved_activations=False, distributed_backend='nccl', distributed_timeout_minutes=10, embedding_path=None, empty_unused_memory_level=0, encoder_num_layers=None, encoder_seq_length=None, end_weight_decay=None, eod_mask_loss=False, eval_interval=1000, eval_iters=100, evidence_data_path=None, exit_duration_in_mins=None, exit_interval=None, exit_on_missing_checkpoint=False, exit_signal_handler=False, ffn_hidden_size=None, finetune=False, fp16=False, fp16_lm_cross_entropy=False, fp32_residual_connection=False, fp8_amax_compute_algo='most_recent', fp8_amax_history_len=1, fp8_e4m3=False, fp8_hybrid=False, fp8_interval=1, fp8_margin=0, fp8_wgrad=True, global_batch_size=None, gradient_accumulation_fusion=True, head_lr_mult=1.0, hidden_dropout=0.1, hidden_size=None, hysteresis=2, ict_head_size=None, ict_load=None, img_h=224, img_w=224, indexer_batch_size=128, indexer_log_interval=1000, inference_batch_times_seqlen_threshold=512, init_method_std=0.02, init_method_xavier_uniform=False, initial_loss_scale=4294967296, iter_per_epoch=1250, kv_channels=None, layernorm_epsilon=1e-05, lazy_mpu_init=None, load='/workspace/model/megatron-models/345m-init-mp', local_rank=None, log_batch_size_to_tensorboard=False, log_interval=100, log_learning_rate_to_tensorboard=True, log_loss_scale_to_tensorboard=True, log_memory_to_tensorboard=False, log_num_zeros_in_grad=False, log_params_norm=False, log_timers_to_tensorboard=False, log_validation_ppl_to_tensorboard=False, log_world_size_to_tensorboard=False, loss_scale=None, loss_scale_window=1000, lr=None, lr_decay_iters=None, lr_decay_samples=None, lr_decay_style='linear', lr_warmup_fraction=None, lr_warmup_iters=0, lr_warmup_samples=0, make_vocab_size_divisible_by=128, mask_factor=1.0, mask_prob=0.15, mask_type='random', masked_softmax_fusion=False, max_position_embeddings=None, max_tokens_to_oom=12000, merge_file=None, micro_batch_size=1, min_loss_scale=1.0, min_lr=0.0, mmap_warmup=False, model_parallel_size=None, no_load_optim=True, no_load_rng=True, no_persist_layer_norm=False, no_save_optim=True, no_save_rng=True, num_attention_heads=None, num_channels=3, num_classes=1000, num_experts=None, num_layers=None, num_layers_per_virtual_pipeline_stage=None, num_workers=2, onnx_safe=None, openai_gelu=False, optimizer='adam', output_bert_embeddings=False, overlap_p2p_comm=False, override_opt_param_scheduler=False, patch_dim=16, perform_initialization=False, pipeline_model_parallel_size=1, pipeline_model_parallel_split_rank=None, query_in_block_prob=0.1, rampup_batch_size=None, rank=0, recompute_activations=False, recompute_granularity=None, recompute_method=None, recompute_num_layers=1, reset_attention_mask=False, reset_position_ids=False, retriever_report_topk_accuracies=[], retriever_score_scaling=False, retriever_seq_length=256, retro_add_retriever=False, retro_cyclic_train_iters=None, retro_encoder_attention_dropout=0.1, retro_encoder_hidden_dropout=0.1, retro_encoder_layers=2, retro_num_neighbors=2, retro_num_retrieved_chunks=2, retro_return_doc_ids=False, retro_workdir=None, rotary_percent=1.0, sample_rate=1.0, save=None, save_interval=None, scatter_gather_tensors_in_pipeline=True, seed=1234, seq_length=None, sequence_parallel=False, sgd_momentum=0.9, short_seq_prob=0.1, split='969, 30, 1', squared_relu=False, standalone_embedding_stage=False, start_weight_decay=None, swiglu=False, swin_backbone_type='tiny', tensor_model_parallel_size=1, tensorboard_dir=None, tensorboard_log_interval=1, tensorboard_queue_size=1000, test_data_path=None, timing_log_level=0, timing_log_option='minmax', titles_data_path=None, tokenizer_model=None, tokenizer_type=None, train_data_path=None, train_iters=None, train_samples=None, transformer_impl='local', untie_embeddings_and_output_weights=False, use_checkpoint_args=False, use_checkpoint_opt_param_scheduler=False, use_contiguous_buffers_in_local_ddp=True, use_cpu_initialization=True, use_distributed_optimizer=False, use_flash_attn=False, use_one_sent_docs=False, use_ring_exchange_p2p=False, use_rotary_position_embeddings=False, valid_data_path=None, vision_backbone_type='vit', vision_pretraining=False, vision_pretraining_type='classify', vocab_extra_ids=0, vocab_file=None, vocab_size=None, warmup=None, weight_decay=0.01, weight_decay_incr_style='constant', world_size=1)
Setting num_layers to 24 from checkpoint
Setting hidden_size to 1024 from checkpoint
Setting ffn_hidden_size to 4096 from checkpoint
Setting seq_length to 1024 from checkpoint
Setting num_attention_heads to 16 from checkpoint
Setting kv_channels to 64 from checkpoint
Setting max_position_embeddings to 1024 from checkpoint
Setting add_position_embedding to True from checkpoint
Setting use_rotary_position_embeddings to False from checkpoint
Setting rotary_percent to 1.0 from checkpoint
Setting add_bias_linear to True from checkpoint
Setting swiglu to False from checkpoint
Setting untie_embeddings_and_output_weights to False from checkpoint
Setting apply_layernorm_1p to False from checkpoint
Setting tokenizer_type to GPT2BPETokenizer from checkpoint
Setting padded_vocab_size to 50432 from checkpoint
Setting tensor_model_parallel_size to 2 from checkpoint
Setting pipeline_model_parallel_size to 2 from checkpoint
Checkpoint did not provide arguments virtual_pipeline_model_parallel_size
Checkpoint did not provide arguments num_layers_per_virtual_pipeline_stage
using world size: 4, data-parallel-size: 1, tensor-model-parallel size: 2, pipeline-model-parallel size: 2 
setting global batch size to 1
using torch.float32 for parameters ...
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
  bias_dropout_fusion ............................. False
  bias_gelu_fusion ................................ False
  biencoder_projection_dim ........................ 0
  biencoder_shared_query_context_model ............ False
  block_data_path ................................. None
  classes_fraction ................................ 1.0
  clip_grad ....................................... 1.0
  consumed_train_samples .......................... 0
  consumed_valid_samples .......................... 0
  data_cache_path ................................. None
  data_impl ....................................... infer
  data_parallel_random_init ....................... False
  data_parallel_size .............................. 1
  data_path ....................................... None
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
  eval_iters ...................................... 100
  evidence_data_path .............................. None
  exit_duration_in_mins ........................... None
  exit_interval ................................... None
  exit_on_missing_checkpoint ...................... False
  exit_signal_handler ............................. False
  ffn_hidden_size ................................. 4096
  finetune ........................................ False
  fp16 ............................................ False
  fp16_lm_cross_entropy ........................... False
  fp32_residual_connection ........................ False
  fp8_amax_compute_algo ........................... most_recent
  fp8_amax_history_len ............................ 1
  fp8_e4m3 ........................................ False
  fp8_hybrid ...................................... False
  fp8_interval .................................... 1
  fp8_margin ...................................... 0
  fp8_wgrad ....................................... True
  global_batch_size ............................... 1
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
  iteration ....................................... 5000
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
  lr .............................................. None
  lr_decay_iters .................................. None
  lr_decay_samples ................................ None
  lr_decay_style .................................. linear
  lr_warmup_fraction .............................. None
  lr_warmup_iters ................................. 0
  lr_warmup_samples ............................... 0
  make_vocab_size_divisible_by .................... 128
  mask_factor ..................................... 1.0
  mask_prob ....................................... 0.15
  mask_type ....................................... random
  masked_softmax_fusion ........................... False
  max_position_embeddings ......................... 1024
  max_tokens_to_oom ............................... 12000
  merge_file ...................................... /workspace/model/gpt2-vocab/gpt2-merges.txt
  micro_batch_size ................................ 1
  min_loss_scale .................................. 1.0
  min_lr .......................................... 0.0
  mmap_warmup ..................................... False
  no_load_optim ................................... True
  no_load_rng ..................................... True
  no_persist_layer_norm ........................... False
  no_save_optim ................................... True
  no_save_rng ..................................... True
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
  padded_vocab_size ............................... 50432
  params_dtype .................................... torch.float32
  patch_dim ....................................... 16
  perform_initialization .......................... False
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
  save ............................................ None
  save_interval ................................... None
  scatter_gather_tensors_in_pipeline .............. True
  seed ............................................ 1234
  seq_length ...................................... 1024
  sequence_parallel ............................... False
  sgd_momentum .................................... 0.9
  short_seq_prob .................................. 0.1
  split ........................................... 969, 30, 1
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
  train_iters ..................................... None
  train_samples ................................... None
  transformer_impl ................................ local
  transformer_pipeline_model_parallel_size ........ 2
  untie_embeddings_and_output_weights ............. False
  use_checkpoint_args ............................. False
  use_checkpoint_opt_param_scheduler .............. False
  use_contiguous_buffers_in_local_ddp ............. True
  use_cpu_initialization .......................... True
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
setting number of micro-batches to constant 1
> building GPT2BPETokenizer tokenizer ...
 > padded vocab (size: 50257) with 175 dummy tokens (new size: 50432)
building GPT model ...
 loading checkpoint from /workspace/model/megatron-models/345m-init-mp at iteration 5000
 checkpoint version 3.0
  successfully loaded checkpoint from /workspace/model/megatron-models/345m-init-mp at iteration 5000
building GPT model ...
 loading checkpoint from /workspace/model/megatron-models/345m-init-mp at iteration 5000
 checkpoint version 3.0
  successfully loaded checkpoint from /workspace/model/megatron-models/345m-init-mp at iteration 5000
Overwriting default ffn_hidden_size value None with value from checkpoint 4096.
Overwriting default kv_channels value None with value from checkpoint 64.
Overwriting default micro_batch_size value 1 with value from checkpoint 4.
Overwriting default global_batch_size value None with value from checkpoint 16.
Overwriting default dataloader_type value None with value from checkpoint single.
Overwriting default lr value None with value from checkpoint 0.00015.
Overwriting default lr_decay_style value linear with value from checkpoint cosine.
Overwriting default min_lr value 0.0 with value from checkpoint 1e-05.
Overwriting default load value None with value from checkpoint /workspace/model/megatron-models/345m-init-mp.
Overwriting default fp16 value False with value from checkpoint True.
Overwriting default local_rank value None with value from checkpoint 0.
Overwriting default eval_iters value 100 with value from checkpoint 10.
Overwriting default data_path value None with value from checkpoint ['/workspace/data/my-gpt2_text_document'].
Overwriting default split value 969, 30, 1 with value from checkpoint 700,200,100.
Overwriting default data_impl value infer with value from checkpoint mmap.
Overwriting default world_size value 1 with value from checkpoint 4.
Checkpoint had argument transformer_pipeline_model_parallel_size but new arguments does not have this.
Checkpoint had argument data_parallel_size but new arguments does not have this.
Checkpoint had argument consumed_train_samples but new arguments does not have this.
Checkpoint had argument consumed_valid_samples but new arguments does not have this.
Checkpoint had argument variable_seq_lengths but new arguments does not have this.
Checkpoint had argument padded_vocab_size but new arguments does not have this.
Checkpoint had argument model_type but new arguments does not have this.
Checkpoint had argument allow_transformer_engine but new arguments does not have this.
Checkpoint had argument iteration but new arguments does not have this.
Checkpoint had argument do_train but new arguments does not have this.
Checkpoint had argument do_valid but new arguments does not have this.
Checkpoint had argument do_test but new arguments does not have this.
Checkpoint had argument curr_iteration but new arguments does not have this.
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
  async_tensor_model_parallel_allreduce ........... False
  attention_dropout ............................... 0.1
  attention_softmax_in_fp32 ....................... False
  barrier_with_L1_time ............................ True
  bert_binary_head ................................ True
  bert_embedder_type .............................. megatron
  bert_load ....................................... None
  bf16 ............................................ False
  bias_dropout_fusion ............................. False
  bias_gelu_fusion ................................ False
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
  local_rank ...................................... 0
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
  lr_decay_iters .................................. None
  lr_decay_samples ................................ None
  lr_decay_style .................................. cosine
  lr_warmup_fraction .............................. None
  lr_warmup_iters ................................. 0
  lr_warmup_samples ............................... 0
  make_vocab_size_divisible_by .................... 128
  mask_factor ..................................... 1.0
  mask_prob ....................................... 0.15
  mask_type ....................................... random
  masked_softmax_fusion ........................... False
  max_position_embeddings ......................... 1024
  max_tokens_to_oom ............................... 12000
  merge_file ...................................... /workspace/model/gpt2-vocab/gpt2-merges.txt
  micro_batch_size ................................ 4
  min_loss_scale .................................. 1.0
  min_lr .......................................... 1e-05
  mmap_warmup ..................................... False
  no_load_optim ................................... True
  no_load_rng ..................................... True
  no_persist_layer_norm ........................... False
  no_save_optim ................................... True
  no_save_rng ..................................... True
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
  perform_initialization .......................... False
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
  save ............................................ /workspace/model/megatron-models/345m-init-mp-out
  save_interval ................................... 1
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
  train_iters ..................................... None
  train_samples ................................... None
  transformer_impl ................................ local
  transformer_pipeline_model_parallel_size ........ 1
  untie_embeddings_and_output_weights ............. False
  use_checkpoint_args ............................. False
  use_checkpoint_opt_param_scheduler .............. False
  use_contiguous_buffers_in_local_ddp ............. True
  use_cpu_initialization .......................... True
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
setting number of micro-batches to constant 1
> building GPT2BPETokenizer tokenizer ...
sending embeddings
sending transformer layer 0
sending transformer layer 1
 > padded vocab (size: 50257) with 47 dummy tokens (new size: 50304)
Setting consumed_train_samples to 80000 and consumed_valid_samples to 960
sending transformer layer 2
...
sending transformer layer 11
building GPT model ...
 loading checkpoint from /workspace/model/megatron-models/345m-init-mp at iteration 5000
received embeddings
 > padded vocab (size: 50257) with 47 dummy tokens (new size: 50304)
building GPT model ...
received transformer layer 0
...
received transformer layer 11
 checkpoint version 3.0
  successfully loaded checkpoint from /workspace/model/megatron-models/345m-init-mp at iteration 5000
building GPT model ...
 loading checkpoint from /workspace/model/megatron-models/345m-init-mp at iteration 5000
 checkpoint version 3.0
  successfully loaded checkpoint from /workspace/model/megatron-models/345m-init-mp at iteration 5000
sending transformer layer 12
sending transformer layer 13
sending transformer layer 14
sending transformer layer 15
received transformer layer 12
received transformer layer 13
sending transformer layer 16
sending transformer layer 17
sending transformer layer 18
received transformer layer 14
sending transformer layer 19
sending transformer layer 20
sending transformer layer 21
sending transformer layer 22
received transformer layer 15
received transformer layer 16
received transformer layer 17
sending transformer layer 23
sending final layernorm
received transformer layer 18
Waiting for saver to complete...
received transformer layer 19
received transformer layer 20
received transformer layer 21
received transformer layer 22
received transformer layer 23
received final layernorm
saving checkpoint at iteration    5000 to /workspace/model/megatron-models/345m-init-mp-out
  successfully saved checkpoint at iteration    5000 to /workspace/model/megatron-models/345m-init-mp-out
Done!
```

</p></details>




```
python tools/checkpoint_util.py \
        --model-type GPT \
        --load-dir /workspace/model/megatron-models/345m-init-mp\
        --save-dir /workspace/model/megatron-models/345m-init-mp-out-2tp \
        --target-tensor-parallel-size 2 \
        --target-pipeline-parallel-size 1
```


```
tree -h /workspace/model/megatron-models/345m-init-mp-out-2tp
/workspace/model/megatron-models/345m-init-mp-out-2tp
├── [4.0K]  iter_0005000
│   ├── [4.0K]  mp_rank_00
│   │   └── [680M]  model_optim_rng.pt
│   └── [4.0K]  mp_rank_01
│       └── [680M]  model_optim_rng.pt
└── [   4]  latest_checkpointed_iteration.txt

3 directories, 3 files
```



```
python tools/checkpoint_util.py \
        --model-type GPT \
        --load-dir /workspace/model/megatron-models/345m-init-mp\
        --save-dir /workspace/model/megatron-models/345m-init-mp-out-4pp \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 4
```


```
tree -h /workspace/model/megatron-models/345m-init-mp-out-4pp
/workspace/model/megatron-models/345m-init-mp-out-4pp
├── [4.0K]  iter_0005000
│   ├── [4.0K]  mp_rank_00_000
│   │   └── [489M]  model_optim_rng.pt
│   ├── [4.0K]  mp_rank_00_001
│   │   └── [288M]  model_optim_rng.pt
│   ├── [4.0K]  mp_rank_00_002
│   │   └── [288M]  model_optim_rng.pt
│   └── [4.0K]  mp_rank_00_003
│       └── [485M]  model_optim_rng.pt
└── [   4]  latest_checkpointed_iteration.txt
```


## 模型服务


```
pip install flask -i https://pypi.tuna.tsinghua.edu.cn/simple  --trusted-host pypi.tuna.tsinghua.edu.cn
pip install flask-restful -i https://pypi.tuna.tsinghua.edu.cn/simple  --trusted-host pypi.tuna.tsinghua.edu.cn
```

```
vim examples/run_text_generation_server_345M.sh
```


启动服务：

<details><summary>详细输出：</summary><p>


```
> sh examples/run_text_generation_server_345M.sh
Looking in indexes: https://pypi.org/simple, https://pypi.ngc.nvidia.com
Requirement already satisfied: flask-restful in /usr/local/lib/python3.8/dist-packages (0.3.10)
Requirement already satisfied: aniso8601>=0.82 in /usr/local/lib/python3.8/dist-packages (from flask-restful) (9.0.1)
Requirement already satisfied: pytz in /usr/local/lib/python3.8/dist-packages (from flask-restful) (2023.3)
Requirement already satisfied: six>=1.3.0 in /usr/local/lib/python3.8/dist-packages (from flask-restful) (1.16.0)
Requirement already satisfied: Flask>=0.8 in /usr/local/lib/python3.8/dist-packages (from flask-restful) (2.3.2)
Requirement already satisfied: itsdangerous>=2.1.2 in /usr/local/lib/python3.8/dist-packages (from Flask>=0.8->flask-restful) (2.1.2)
Requirement already satisfied: importlib-metadata>=3.6.0 in /usr/local/lib/python3.8/dist-packages (from Flask>=0.8->flask-restful) (6.3.0)
Requirement already satisfied: Jinja2>=3.1.2 in /usr/local/lib/python3.8/dist-packages (from Flask>=0.8->flask-restful) (3.1.2)
Requirement already satisfied: Werkzeug>=2.3.3 in /usr/local/lib/python3.8/dist-packages (from Flask>=0.8->flask-restful) (2.3.6)
Requirement already satisfied: blinker>=1.6.2 in /usr/local/lib/python3.8/dist-packages (from Flask>=0.8->flask-restful) (1.6.2)
Requirement already satisfied: click>=8.1.3 in /usr/local/lib/python3.8/dist-packages (from Flask>=0.8->flask-restful) (8.1.3)
Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.8/dist-packages (from importlib-metadata>=3.6.0->Flask>=0.8->flask-restful) (3.15.0)
Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.8/dist-packages (from Jinja2>=3.1.2->Flask>=0.8->flask-restful) (2.1.2)
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv
WARNING: You are using pip version 21.2.4; however, version 23.2.1 is available.
You should consider upgrading via the '/usr/bin/python -m pip install --upgrade pip' command.
using world size: 1, data-parallel-size: 1, tensor-model-parallel size: 1, pipeline-model-parallel size: 1 
WARNING: overriding default arguments for tokenizer_type:GPT2BPETokenizer                        with tokenizer_type:GPT2BPETokenizer
setting global batch size to 1
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
  data_impl ....................................... infer
  data_parallel_random_init ....................... False
  data_parallel_size .............................. 1
  data_path ....................................... None
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
  eval_iters ...................................... 100
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
  global_batch_size ............................... 1
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
  load ............................................ /workspace/model/megatron-models/345m-init-mp-out
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
  lr .............................................. None
  lr_decay_iters .................................. None
  lr_decay_samples ................................ None
  lr_decay_style .................................. linear
  lr_warmup_fraction .............................. None
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
  min_lr .......................................... 0.0
  mmap_warmup ..................................... False
  no_load_optim ................................... True
  no_load_rng ..................................... True
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
  out_seq_length .................................. 1024
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
  save ............................................ None
  save_interval ................................... None
  scatter_gather_tensors_in_pipeline .............. True
  seed ............................................ 42
  seq_length ...................................... 1024
  sequence_parallel ............................... False
  sgd_momentum .................................... 0.9
  short_seq_prob .................................. 0.1
  split ........................................... 969, 30, 1
  squared_relu .................................... False
  standalone_embedding_stage ...................... False
  start_weight_decay .............................. 0.01
  swiglu .......................................... False
  swin_backbone_type .............................. tiny
  temperature ..................................... 1.0
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
  top_k ........................................... 0
  top_p ........................................... 0.9
  train_data_path ................................. None
  train_iters ..................................... None
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
setting number of micro-batches to constant 1
> building GPT2BPETokenizer tokenizer ...
 > padded vocab (size: 50257) with 47 dummy tokens (new size: 50304)
> initializing torch distributed ...
> initialized tensor model parallel with size 1
> initialized pipeline model parallel with size 1
> setting random seeds to 42 ...
> compiling dataset index builder ...
make: Entering directory '/workspace/code/Megatron-LM/megatron/data'
make: Nothing to be done for 'default'.
make: Leaving directory '/workspace/code/Megatron-LM/megatron/data'
>>> done with dataset index builder. Compilation time: 0.237 seconds
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
>>> done with compiling and loading fused kernels. Compilation time: 1.632 seconds
building GPT model ...
 > number of parameters on (tensor, pipeline) model parallel rank (0, 0): 354871296
 loading checkpoint from /workspace/model/megatron-models/345m-init-mp-out at iteration 5000
 checkpoint version 3.0
  successfully loaded checkpoint from /workspace/model/megatron-models/345m-init-mp-out at iteration 5000
 * Serving Flask app 'megatron.text_generation_server'
 * Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
 * Running on http://192.168.78.1:5000
Press CTRL+C to quit
request IP: 127.0.0.1
{"prompts": ["Hello world"], "tokens_to_generate": 1}
start time:  2023-07-23 16:00:12.738873
127.0.0.1 - - [23/Jul/2023 16:00:14] "PUT /api HTTP/1.1" 200 -
request IP: 127.0.0.1
{"prompts": ["hello"], "tokens_to_generate": 5}
start time:  2023-07-23 16:02:46.352169
127.0.0.1 - - [23/Jul/2023 16:02:47] "PUT /api HTTP/1.1" 200 -
request IP: 127.0.0.1
{"prompts": ["world"], "tokens_to_generate": 2}
start time:  2023-07-23 16:03:23.518545
127.0.0.1 - - [23/Jul/2023 16:03:23] "PUT /api HTTP/1.1" 200 -
```

</p></details>




```
> curl 'http://localhost:5000/api' -X 'PUT' -H 'Content-Type: application/json; charset=UTF-8'  -d '{"prompts":["Hello world"], "tokens_to_generate":1}'

{"logprobs":null,"segments":[["Hello"," world",","]],"text":["Hello world,"]}
```


<details><summary>详细输出：</summary><p>

        
```
> python tools/text_generation_cli.py localhost:5000
Enter prompt: hello
Enter number of tokens to generate: 5
Megatron Response: 
hello! Until that protagonist receive
Enter prompt: world 
Enter number of tokens to generate: 2
Megatron Response: 
worldboarding-
Enter prompt: 
```

</p></details>



### 4TP


<details><summary>详细输出：</summary><p>


```
> sh examples/run_text_generation_server_345M_4_tensor_parallel.sh 
Looking in indexes: https://pypi.org/simple, https://pypi.ngc.nvidia.com
Requirement already satisfied: flask-restful in /usr/local/lib/python3.8/dist-packages (0.3.10)
Requirement already satisfied: six>=1.3.0 in /usr/local/lib/python3.8/dist-packages (from flask-restful) (1.16.0)
Requirement already satisfied: pytz in /usr/local/lib/python3.8/dist-packages (from flask-restful) (2023.3)
Requirement already satisfied: aniso8601>=0.82 in /usr/local/lib/python3.8/dist-packages (from flask-restful) (9.0.1)
Requirement already satisfied: Flask>=0.8 in /usr/local/lib/python3.8/dist-packages (from flask-restful) (2.3.2)
Requirement already satisfied: blinker>=1.6.2 in /usr/local/lib/python3.8/dist-packages (from Flask>=0.8->flask-restful) (1.6.2)
Requirement already satisfied: Werkzeug>=2.3.3 in /usr/local/lib/python3.8/dist-packages (from Flask>=0.8->flask-restful) (2.3.6)
Requirement already satisfied: importlib-metadata>=3.6.0 in /usr/local/lib/python3.8/dist-packages (from Flask>=0.8->flask-restful) (6.3.0)
Requirement already satisfied: click>=8.1.3 in /usr/local/lib/python3.8/dist-packages (from Flask>=0.8->flask-restful) (8.1.3)
Requirement already satisfied: itsdangerous>=2.1.2 in /usr/local/lib/python3.8/dist-packages (from Flask>=0.8->flask-restful) (2.1.2)
Requirement already satisfied: Jinja2>=3.1.2 in /usr/local/lib/python3.8/dist-packages (from Flask>=0.8->flask-restful) (3.1.2)
Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.8/dist-packages (from importlib-metadata>=3.6.0->Flask>=0.8->flask-restful) (3.15.0)
Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.8/dist-packages (from Jinja2>=3.1.2->Flask>=0.8->flask-restful) (2.1.2)
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv
WARNING: You are using pip version 21.2.4; however, version 23.2.1 is available.
You should consider upgrading via the '/usr/bin/python -m pip install --upgrade pip' command.
WARNING:torch.distributed.run:
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
*****************************************
using world size: 4, data-parallel-size: 1, tensor-model-parallel size: 4, pipeline-model-parallel size: 1 
WARNING: overriding default arguments for tokenizer_type:GPT2BPETokenizer                        with tokenizer_type:GPT2BPETokenizer
setting global batch size to 1
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
  data_impl ....................................... infer
  data_parallel_random_init ....................... False
  data_parallel_size .............................. 1
  data_path ....................................... None
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
  eval_iters ...................................... 100
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
  global_batch_size ............................... 1
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
  lr .............................................. None
  lr_decay_iters .................................. None
  lr_decay_samples ................................ None
  lr_decay_style .................................. linear
  lr_warmup_fraction .............................. None
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
  min_lr .......................................... 0.0
  mmap_warmup ..................................... False
  no_load_optim ................................... True
  no_load_rng ..................................... True
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
  out_seq_length .................................. 1024
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
  save ............................................ None
  save_interval ................................... None
  scatter_gather_tensors_in_pipeline .............. True
  seed ............................................ 42
  seq_length ...................................... 1024
  sequence_parallel ............................... False
  sgd_momentum .................................... 0.9
  short_seq_prob .................................. 0.1
  split ........................................... 969, 30, 1
  squared_relu .................................... False
  standalone_embedding_stage ...................... False
  start_weight_decay .............................. 0.01
  swiglu .......................................... False
  swin_backbone_type .............................. tiny
  temperature ..................................... 1.0
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
  top_k ........................................... 0
  top_p ........................................... 0.9
  train_data_path ................................. None
  train_iters ..................................... None
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
setting number of micro-batches to constant 1
> building GPT2BPETokenizer tokenizer ...
 > padded vocab (size: 50257) with 431 dummy tokens (new size: 50688)
> initializing torch distributed ...
> initialized tensor model parallel with size 4
> initialized pipeline model parallel with size 1
> setting random seeds to 42 ...
> compiling dataset index builder ...
make: Entering directory '/workspace/code/Megatron-LM/megatron/data'
make: Nothing to be done for 'default'.
make: Leaving directory '/workspace/code/Megatron-LM/megatron/data'
>>> done with dataset index builder. Compilation time: 0.185 seconds
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
>>> done with compiling and loading fused kernels. Compilation time: 3.342 seconds
building GPT model ...
 > number of parameters on (tensor, pipeline) model parallel rank (2, 0): 89714688
 > number of parameters on (tensor, pipeline) model parallel rank (3, 0): 89714688
 > number of parameters on (tensor, pipeline) model parallel rank (0, 0): 89714688
 > number of parameters on (tensor, pipeline) model parallel rank (1, 0): 89714688
 loading checkpoint from /workspace/model/megatron-models/345m-init-4tp at iteration 2000
 checkpoint version 3.0
  successfully loaded checkpoint from /workspace/model/megatron-models/345m-init-4tp at iteration 2000
 * Serving Flask app 'megatron.text_generation_server'
 * Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
 * Running on http://192.168.78.1:5000
Press CTRL+C to quit
request IP: 127.0.0.1
{"prompts": ["Hello world"], "tokens_to_generate": 1}
start time:  2023-07-24 08:02:14.824674
127.0.0.1 - - [24/Jul/2023 08:02:18] "PUT /api HTTP/1.1" 200 -

```

</p></details>


<details><summary>详细输出：</summary><p>

```
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A   1844443      C   /usr/bin/python                   788MiB |
|    1   N/A  N/A   1844444      C   /usr/bin/python                   788MiB |
|    2   N/A  N/A   1844445      C   /usr/bin/python                   788MiB |
|    3   N/A  N/A   1844446      C   /usr/bin/python                   788MiB |
+-----------------------------------------------------------------------------+
```
</p></details>


### 2TP+2PP


<details><summary>详细输出：</summary><p>


```
 sh examples/run_text_generation_server_345M_2tp_2dp.sh 
Looking in indexes: https://pypi.org/simple, https://pypi.ngc.nvidia.com
Requirement already satisfied: flask-restful in /usr/local/lib/python3.8/dist-packages (0.3.10)
Requirement already satisfied: six>=1.3.0 in /usr/local/lib/python3.8/dist-packages (from flask-restful) (1.16.0)
Requirement already satisfied: pytz in /usr/local/lib/python3.8/dist-packages (from flask-restful) (2023.3)
Requirement already satisfied: aniso8601>=0.82 in /usr/local/lib/python3.8/dist-packages (from flask-restful) (9.0.1)
Requirement already satisfied: Flask>=0.8 in /usr/local/lib/python3.8/dist-packages (from flask-restful) (2.3.2)
Requirement already satisfied: Jinja2>=3.1.2 in /usr/local/lib/python3.8/dist-packages (from Flask>=0.8->flask-restful) (3.1.2)
Requirement already satisfied: Werkzeug>=2.3.3 in /usr/local/lib/python3.8/dist-packages (from Flask>=0.8->flask-restful) (2.3.6)
Requirement already satisfied: blinker>=1.6.2 in /usr/local/lib/python3.8/dist-packages (from Flask>=0.8->flask-restful) (1.6.2)
Requirement already satisfied: click>=8.1.3 in /usr/local/lib/python3.8/dist-packages (from Flask>=0.8->flask-restful) (8.1.3)
Requirement already satisfied: importlib-metadata>=3.6.0 in /usr/local/lib/python3.8/dist-packages (from Flask>=0.8->flask-restful) (6.3.0)
Requirement already satisfied: itsdangerous>=2.1.2 in /usr/local/lib/python3.8/dist-packages (from Flask>=0.8->flask-restful) (2.1.2)
Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.8/dist-packages (from importlib-metadata>=3.6.0->Flask>=0.8->flask-restful) (3.15.0)
Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.8/dist-packages (from Jinja2>=3.1.2->Flask>=0.8->flask-restful) (2.1.2)
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv
WARNING: You are using pip version 21.2.4; however, version 23.2.1 is available.
You should consider upgrading via the '/usr/bin/python -m pip install --upgrade pip' command.
WARNING:torch.distributed.run:
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
*****************************************
using world size: 4, data-parallel-size: 1, tensor-model-parallel size: 2, pipeline-model-parallel size: 2 
WARNING: overriding default arguments for tokenizer_type:GPT2BPETokenizer                        with tokenizer_type:GPT2BPETokenizer
setting global batch size to 1
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
  data_impl ....................................... infer
  data_parallel_random_init ....................... False
  data_parallel_size .............................. 1
  data_path ....................................... None
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
  eval_iters ...................................... 100
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
  global_batch_size ............................... 1
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
  lr .............................................. None
  lr_decay_iters .................................. None
  lr_decay_samples ................................ None
  lr_decay_style .................................. linear
  lr_warmup_fraction .............................. None
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
  min_lr .......................................... 0.0
  mmap_warmup ..................................... False
  no_load_optim ................................... True
  no_load_rng ..................................... True
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
  out_seq_length .................................. 1024
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
  save ............................................ None
  save_interval ................................... None
  scatter_gather_tensors_in_pipeline .............. True
  seed ............................................ 42
  seq_length ...................................... 1024
  sequence_parallel ............................... False
  sgd_momentum .................................... 0.9
  short_seq_prob .................................. 0.1
  split ........................................... 969, 30, 1
  squared_relu .................................... False
  standalone_embedding_stage ...................... False
  start_weight_decay .............................. 0.01
  swiglu .......................................... False
  swin_backbone_type .............................. tiny
  temperature ..................................... 1.0
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
  top_k ........................................... 0
  top_p ........................................... 0.9
  train_data_path ................................. None
  train_iters ..................................... None
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
setting number of micro-batches to constant 1
> building GPT2BPETokenizer tokenizer ...
 > padded vocab (size: 50257) with 175 dummy tokens (new size: 50432)
> initializing torch distributed ...
> initialized tensor model parallel with size 2
> initialized pipeline model parallel with size 2
> setting random seeds to 42 ...
> compiling dataset index builder ...
make: Entering directory '/workspace/code/Megatron-LM/megatron/data'
make: Nothing to be done for 'default'.
make: Leaving directory '/workspace/code/Megatron-LM/megatron/data'
>>> done with dataset index builder. Compilation time: 0.206 seconds
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
>>> done with compiling and loading fused kernels. Compilation time: 3.460 seconds
building GPT model ...
 > number of parameters on (tensor, pipeline) model parallel rank (0, 0): 102483968
 > number of parameters on (tensor, pipeline) model parallel rank (0, 1): 101437440
 > number of parameters on (tensor, pipeline) model parallel rank (1, 1): 101437440
 > number of parameters on (tensor, pipeline) model parallel rank (1, 0): 102483968
 loading checkpoint from /workspace/model/megatron-models/345m-init-mp at iteration 5000
 checkpoint version 3.0
  successfully loaded checkpoint from /workspace/model/megatron-models/345m-init-mp at iteration 5000
 * Serving Flask app 'megatron.text_generation_server'
 * Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
 * Running on http://192.168.78.1:5000
Press CTRL+C to quit
request IP: 127.0.0.1
{"prompts": ["hello"], "tokens_to_generate": 5}
start time:  2023-07-24 08:11:43.956061
[W ProcessGroupNCCL.cpp:1692] Warning: 0NCCL_AVOID_RECORD_STREAMS=1 has no effect for point-to-point collectives. (function operator())
[W ProcessGroupNCCL.cpp:1692] Warning: 0NCCL_AVOID_RECORD_STREAMS=1 has no effect for point-to-point collectives. (function operator())
[W ProcessGroupNCCL.cpp:1692] Warning: 0NCCL_AVOID_RECORD_STREAMS=1 has no effect for point-to-point collectives. (function operator())
[W ProcessGroupNCCL.cpp:1692] Warning: 0NCCL_AVOID_RECORD_STREAMS=1 has no effect for point-to-point collectives. (function operator())
127.0.0.1 - - [24/Jul/2023 08:11:53] "PUT /api HTTP/1.1" 200 -
```


</p></details>





<details><summary>详细输出：</summary><p>

```
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A   1869409      C   /usr/bin/python                  1222MiB |
|    1   N/A  N/A   1869410      C   /usr/bin/python                  1222MiB |
|    2   N/A  N/A   1869411      C   /usr/bin/python                  1222MiB |
|    3   N/A  N/A   1869412      C   /usr/bin/python                  1222MiB |
+-----------------------------------------------------------------------------+
```
</p></details>






## 模型评估

```
git clone https://github.com/NVIDIA/Megatron-LM.git

git checkout 6ef5bdc
```




<details><summary>详细输出：</summary><p>


```
> sh eval_gpt2_lambada.sh 
using world size: 1, data-parallel-size: 1, tensor-model-parallel size: 1, pipeline-model-parallel size: 1 
setting global batch size to 8
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
  data_impl ....................................... infer
  data_parallel_random_init ....................... False
  data_parallel_size .............................. 1
  data_path ....................................... None
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
  embedding_weights_in_fp32 ....................... False
  empty_unused_memory_level ....................... 0
  encoder_num_layers .............................. 24
  encoder_seq_length .............................. 1024
  end_weight_decay ................................ 0.01
  eod_mask_loss ................................... False
  epochs .......................................... None
  eval_interval ................................... 1000
  eval_iters ...................................... 100
  eval_micro_batch_size ........................... None
  evidence_data_path .............................. None
  exit_duration_in_mins ........................... None
  exit_interval ................................... None
  exit_on_missing_checkpoint ...................... False
  exit_signal_handler ............................. False
  faiss_match ..................................... string
  faiss_topk_retrievals ........................... 100
  faiss_use_gpu ................................... False
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
  keep_last ....................................... False
  kv_channels ..................................... 64
  layernorm_epsilon ............................... 1e-05
  lazy_mpu_init ................................... None
  load ............................................ /workspace/model/megatron-models/345m-init-mp-out
  local_rank ...................................... None
  log_batch_size_to_tensorboard ................... False
  log_interval .................................... 10
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
  lr .............................................. None
  lr_decay_iters .................................. None
  lr_decay_samples ................................ None
  lr_decay_style .................................. linear
  lr_warmup_fraction .............................. None
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
  micro_batch_size ................................ 8
  min_loss_scale .................................. 1.0
  min_lr .......................................... 0.0
  mmap_warmup ..................................... False
  no_load_optim ................................... True
  no_load_rng ..................................... True
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
  overlapping_eval ................................ 32
  override_opt_param_scheduler .................... False
  params_dtype .................................... torch.float16
  patch_dim ....................................... 16
  perform_initialization .......................... True
  pipeline_model_parallel_size .................... 1
  pipeline_model_parallel_split_rank .............. None
  position_embedding_type ......................... learned_absolute
  pretrained_checkpoint ........................... None
  profile ......................................... False
  profile_ranks ................................... [0]
  profile_step_end ................................ 12
  profile_step_start .............................. 10
  qa_data_dev ..................................... None
  qa_data_test .................................... None
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
  save ............................................ None
  save_interval ................................... None
  scatter_gather_tensors_in_pipeline .............. True
  seed ............................................ 1234
  seq_length ...................................... 1024
  sequence_parallel ............................... False
  sgd_momentum .................................... 0.9
  short_seq_prob .................................. 0.1
  skip_train ...................................... False
  split ........................................... 969, 30, 1
  squared_relu .................................... False
  standalone_embedding_stage ...................... False
  start_weight_decay .............................. 0.01
  strict_lambada .................................. True
  swiglu .......................................... False
  swin_backbone_type .............................. tiny
  task ............................................ LAMBADA
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
  train_data ...................................... None
  train_data_path ................................. None
  train_hard_neg .................................. 0
  train_iters ..................................... None
  train_samples ................................... None
  train_with_neg .................................. False
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
  val_av_rank_hard_neg ............................ 30
  val_av_rank_other_neg ........................... 30
  valid_data ...................................... ['/workspace/data/lambada_test.jsonl']
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
setting number of micro-batches to constant 1
> building GPT2BPETokenizer tokenizer ...
 > padded vocab (size: 50257) with 47 dummy tokens (new size: 50304)
> initializing torch distributed ...
> initialized tensor model parallel with size 1
> initialized pipeline model parallel with size 1
> setting random seeds to 1234 ...
> compiling dataset index builder ...
make: Entering directory '/workspace/code/bak/Megatron-LM/megatron/data'
g++ -O3 -Wall -shared -std=c++11 -fPIC -fdiagnostics-color -I/usr/include/python3.8 -I/usr/local/lib/python3.8/dist-packages/pybind11/include helpers.cpp -o helpers.cpython-38-x86_64-linux-gnu.so
make: Leaving directory '/workspace/code/bak/Megatron-LM/megatron/data'
>>> done with dataset index builder. Compilation time: 13.399 seconds
> compiling and loading fused kernels ...
>>> done with compiling and loading fused kernels. Compilation time: 1.411 seconds
building GPT model ...
 > number of parameters on (tensor, pipeline) model parallel rank (0, 0): 354871296
 loading checkpoint from /workspace/model/megatron-models/345m-init-mp-out at iteration 5000
 checkpoint version 3.0
  successfully loaded checkpoint from /workspace/model/megatron-models/345m-init-mp-out at iteration 5000
> building lambada dataset from /workspace/data/lambada_test.jsonl ...
 > found 5153 samples.
> working on iteration: 0
> working on iteration: 10
> working on iteration: 20
> working on iteration: 30
...
> working on iteration: 640
--------------------------------------------------------------------------------------------------------------------
 validation results on LAMBADA | number correct: 0.0000E+00 | total examples: 5.1530E+03 | avg accuracy: 0.0000E+00
--------------------------------------------------------------------------------------------------------------------
done :-)
```

</p></details>











