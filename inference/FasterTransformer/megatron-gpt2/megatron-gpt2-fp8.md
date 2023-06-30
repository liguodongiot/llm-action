

## A800-FP16


```
python examples/pytorch/gpt/utils/megatron_ckpt_convert.py \
>         -head_num 16 \
>         -i /workspace/model/megatron-models/345m/release/ \
>         -o /workspace/model/megatron-models/c-model/345m/ \
>         -t_g 1 \
>         -i_g 1 \
>         --vocab-path /workspace/model/gpt2-vocab/gpt2-vocab.json \
>         --merges-path /workspace/model/gpt2-vocab/gpt2-merges.txt

=============== Argument ===============
saved_dir: /workspace/model/megatron-models/c-model/345m/
in_file: /workspace/model/megatron-models/345m/release/
infer_gpu_num: 1
head_num: 16
trained_tensor_parallel_size: 1
processes: 16
weight_data_type: fp32
load_checkpoints_to_cpu: 1
vocab_path: /workspace/model/gpt2-vocab/gpt2-vocab.json
merges_path: /workspace/model/gpt2-vocab/gpt2-merges.txt
========================================
[INFO] Spent 0:00:05.595219 (h:m:s) to convert the model
```


```
python examples/pytorch/gpt/utils/update_gpt_config.py \
>         --model-dir /workspace/model/megatron-models/c-model/345m/1-gpu \
>         --config-ini-path /workspace/model/megatron-models/c-model/345m/1-gpu/config.ini \
        --max-seq-len 512 \
>         --pipeline-para-size 1 \
>         --tensor-para-size 1 \
>         --max-seq-len 512 \
>         --beam-width 1 \
>         --sampling-top-k 1 \
>         --sampling-top-p 0 \
>         --data-type fp16

```


```
> cat /workspace/model/megatron-models/c-model/345m/1-gpu/config.ini

[ft_instance_hyperparameter]
max_batch_size = 8
max_seq_len = 512
beam_width = 1
top_k = 1
top_p = 0.0
temperature = 1.0
tensor_para_size = 1
pipeline_para_size = 1
data_type = fp16
sparse = 0
int8_mode = 0
enable_custom_all_reduce = 0
model_name = gpt
model_dir = /workspace/model/megatron-models/c-model/345m/1-gpu
repetition_penalty = 1.0
len_penalty = 0.0
beam_search_diversity_rate = 0.0

[request]
request_batch_size = 8
request_output_len = 32
return_log_probs = false
context_log_probs = false
beam_width = 1
top_k = 1
top_p = 0.0
temperature = 1.0
repetition_penalty = 1.0
len_penalty = 0.0
beam_search_diversity_rate = 0.0

[gpt]
model_name = gpt
head_num = 16
size_per_head = 64
inter_size = 4096
num_layer = 24
max_pos_seq_len = 1024
vocab_size = 50304
has_adapters = False
adapter_inter_size = 0
layernorm_eps = 1e-06
start_id = 50256
end_id = 50256
weight_data_type = fp32
tensor_para_size = 1


```

   
```
python examples/pytorch/gpt/lambada_task_example.py \
>        --batch-size 64 \
>        --checkpoint-path /workspace/model/megatron-models/c-model/345m/1-gpu/ \
>        --lib-path /workspace/lib/libth_transformer.so \
>        --lambada-path /workspace/data/lambada_test.jsonl

============== Arguments ===============
checkpoint_path: /workspace/model/megatron-models/c-model/345m/1-gpu/
lib_path: /workspace/lib/libth_transformer.so
config_ini_path: None
lambada_path: /workspace/data/lambada_test.jsonl
output_path: None
batch_size: 64
model_name: gpt
pipeline_para_size: None
data_type: None
sparse: False
int8_mode: None
beam_width: None
sampling_top_k: None
sampling_top_p: None
temperature: None
len_penalty: None
repetition_penalty: None
beam_search_diversity_rate: None
========================================

=============== GPT params ===============
head_num: 16
size_per_head: 64
layer_num: 24
max_seq_len: 1024
tensor_para_size: 1
vocab_size: 50304
start_id: 50256
end_id: 50256
pipeline_para_size: 1
weights_data_type: fp32
has_adapters: False
adapter_inter_size: 0
data_type: fp16
int8_mode: 0
sparse: 0
layernorm_eps: 1e-06
layernorm_type: pre_layernorm
activation_type: gelu
has_positional_encoding: True
has_pre_decoder_layernorm: False
has_post_decoder_layernorm: True
use_attention_linear_bias: False
inter_size: 4096
lib_path: /workspace/lib/libth_transformer.so
========================================
------------------ /workspace/lib/libth_transformer.so
[WARNING] gemm_config.in is not found; using default GEMM algo
[FT][WARNING] Skip NCCL initialization since requested tensor/pipeline parallel sizes are equals to 1.
[FT][INFO] Device NVIDIA A800 80GB PCIe

=========== Inference params ===========
beam_width: 1
top_k: tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32)
top_p: tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
beam_search_diversity_rate: tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
temperature: tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
len_penalty: tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
repetition_penalty: tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
========================================
[INFO] accuracy: 46.7775% (total : 962)
```


## H800-FP32

### 模型转换

```
python examples/pytorch/gpt/utils/megatron_ckpt_convert.py \
-head_num 16 \
-i /workspace/model/megatron-models/345m/release/ \
-o /workspace/model/megatron-models/c-model-fp32 \
-t_g 1 \
-i_g 1 \
--vocab-path /workspace/model/gpt2-vocab/gpt2-vocab.json \
--merges-path /workspace/model/gpt2-vocab/gpt2-merges.txt
```


## H800-FP16

### 模型转换
```
python examples/pytorch/gpt/utils/megatron_ckpt_convert.py \
-head_num 16 \
-i /workspace/model/megatron-models/345m/release/ \
-o /workspace/model/megatron-models/c-model-fp16 \
-t_g 1 \
-i_g 1 \
--vocab-path /workspace/model/gpt2-vocab/gpt2-vocab.json \
--merges-path /workspace/model/gpt2-vocab/gpt2-merges.txt \
-weight_data_type fp16

```



## H800-FP8

### 模型转换

```
python3 examples/pytorch/gpt/utils/megatron_fp8_ckpt_convert.py \
      -i /workspace/model/megatron-models/345m/release \
      -o /workspace/model/megatron-models/c-model-fp8/ \
      -trained_tensor_parallel_size 1 \
      -i_g 1 \
      -head_num 16
      
```






