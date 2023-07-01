

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

运行结果：

```
=============== Argument ===============
saved_dir: /workspace/model/megatron-models/c-model-fp32
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
[INFO] Spent 0:00:02.872906 (h:m:s) to convert the model
```

---


```
python3 examples/pytorch/gpt/gpt_summarization_stat.py \
        --data_type fp32 \
        --lib_path /workspace/FasterTransformer/build/lib/libth_transformer.so \
        --summarize \
         --ft_model_location /workspace/model/megatron-models/c-model-fp32 \
         --hf_model_location /workspace/model/gpt2-tokenizer/gpt2-tokenizer
```
运行结果：
```
python3 examples/pytorch/gpt/gpt_summarization_stat.py \
>         --data_type fp32 \
>         --lib_path /workspace/FasterTransformer/build/lib/libth_transformer.so \
>         --summarize \
>          --ft_model_location /workspace/model/megatron-models/c-model-fp32 \
>          --hf_model_location /workspace/model/gpt2-tokenizer/gpt2-tokenizer
~~~~~~~~~~~~~ args.summarize: True , args.test_hf: False
Found cached dataset cnn_dailymail (/workspace/data/ccdv-data/ccdv/cnn_dailymail/3.0.0/3.0.0/0107f7388b5c6fae455a5661bcd134fc22da53ea75852027040d8d1e997f101f)
100%|██████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 281.70it/s]
top_k: 2
top_p: 0.0
int8_mode: 0
random_seed: 5
temperature: 1
max_seq_len: 1024
max_batch_size: 1
repetition_penalty: 1
vocab_size: 50304
tensor_para_size: 1
pipeline_para_size: 1
lib_path: /workspace/FasterTransformer/build/lib/libth_transformer.so
ckpt_path: /workspace/model/megatron-models/c-model-fp32/1-gpu
hf_config: {'activation_function': 'gelu_new', 'architectures': ['GPT2LMHeadModel'], 'attn_pdrop': 0.1, 'bos_token_id': 50256, 'embd_pdrop': 0.1, 'eos_token_id': 50256, 'initializer_range': 0.02, 'layer_norm_epsilon': 1e-05, 'model_type': 'gpt2', 'n_ctx': 1024, 'n_embd': 768, 'n_head': 12, 'n_layer': 12, 'n_positions': 1024, 'resid_pdrop': 0.1, 'summary_activation': None, 'summary_first_dropout': 0.1, 'summary_proj_to_labels': True, 'summary_type': 'cls_index', 'summary_use_proj': True, 'task_specific_params': {'text-generation': {'do_sample': True, 'max_length': 50}}, 'vocab_size': 50257}
----------------wpe 1024 1024
[WARNING] gemm_config.in is not found; using default GEMM algo
[FT][WARNING] Skip NCCL initialization since requested tensor/pipeline parallel sizes are equals to 1.
[FT][INFO] Device NVIDIA H800
----------args.use_gpt_decoder_ops: False
---------------------------------------------------------
FT Generated :
 Article :  (CNN)James Best, best known for his portrayal of bumbling sheriff Rosco P. Coltrane on TV's "The Dukes of Hazzard," died Monday after a brief illness. He was 88. Best died in hospice in Hickory, North Carolina, of complications from pneumonia, said Steve Latshaw, a longtime friend and Hollywood colleague. Although he'd been a busy actor for decades in theater and in Hollywood, Best didn't become famous until 1979, when "The Dukes of Hazzard's" cornpone charms began beaming into millions of American homes almost every Friday night. For seven seasons, Best's Rosco P. Coltrane chased the moonshine-running Duke boys back and forth across the back roads of fictitious Hazzard County, Georgia, although his "hot pursuit" usually ended with him crashing his patrol car. Although Rosco was slow-witted and corrupt, Best gave him a childlike enthusiasm that got laughs and made him endearing. His character became known for his distinctive "kew-kew-kew" chuckle and for goofy catchphrases such as "cuff 'em and stuff 'em!" upon making an arrest. Among the most popular shows on TV in the early '80s, "The Dukes of Hazzard" ran until 1985 and spawned TV movies, an animated series and video games. Several of Best's "Hazzard" co-stars paid tribute to the late actor on social media. "I laughed and learned more from Jimmie in one hour than from anyone else in a whole year," co-star John Schneider, who played Bo Duke, said on Twitter. "Give Uncle Jesse my love when you see him dear friend." "Jimmy Best was the most constantly creative person I have ever known," said Ben Jones, who played mechanic Cooter on the show, in a Facebook post. "Every minute of his long life was spent acting, writing, producing, painting, teaching, fishing, or involved in another of his life's many passions." Born Jewel Guy on July 26, 1926, in Powderly, Kentucky, Best was orphaned at 3 and adopted by Armen and Essa Best, who renamed him James and raised him in rural Indiana. Best served in the Army during World War II before launching his acting career. In the 1950s and 1960s, he accumulated scores of credits, playing a range of colorful supporting characters in such TV shows as "The Twilight Zone," "Bonanza," "The Andy Griffith Show" and "Gunsmoke." He later appeared in a handful of Burt Reynolds' movies, including "Hooper" and "The End." But Best will always be best known for his "Hazzard" role, which lives on in reruns. "Jimmie was my teacher, mentor, close friend and collaborator for 26 years," Latshaw said. "I directed two of his feature films, including the recent 'Return of the Killer Shrews,' a sequel he co-wrote and was quite proud of as he had made the first one more than 50 years earlier." People we've lost in 2015 . CNN's Stella Chan contributed to this story.

 Highlights :  James Best, who played the sheriff on "The Dukes of Hazzard," died Monday at 88 .
"Hazzard" ran from 1979 to 1985 and was among the most popular shows on TV .

 Summary :   Best was a great actor and an even better friend.
<|endoftext|>.
---------------------------------------------------------
Using the latest cached version of the module from /root/.cache/huggingface/modules/evaluate_modules/metrics/evaluate-metric--rouge/b01e0accf3bd6dd24839b769a5fda24e14995071570870922c71970b3a6ed886 (last modified on Thu Jun 29 16:31:01 2023) since it couldn't be found locally at evaluate-metric--rouge, or remotely on the Hugging Face Hub.
  0%|                                                                                                  | 0/21 [00:00<?, ?it/s]data_point_idx: 1 运行时间： 74.84 毫秒
data_point_idx: 575 运行时间： 254.51 毫秒
 10%|████████▌                                                                                 | 2/21 [00:00<00:03,  6.00it/s]data_point_idx: 1149 运行时间： 202.34 毫秒
 14%|████████████▊                                                                             | 3/21 [00:00<00:03,  5.50it/s]data_point_idx: 1723 运行时间： 236.26 毫秒
 19%|█████████████████▏                                                                        | 4/21 [00:00<00:03,  4.94it/s]data_point_idx: 2297 运行时间： 78.77 毫秒
data_point_idx: 2871 运行时间： 212.88 毫秒
 29%|█████████████████████████▋                                                                | 6/21 [00:01<00:02,  5.77it/s]data_point_idx: 3445 运行时间： 230.37 毫秒
 33%|██████████████████████████████                                                            | 7/21 [00:01<00:02,  5.29it/s]data_point_idx: 4019 运行时间： 56.9 毫秒
data_point_idx: 4593 运行时间： 95.81 毫秒
 43%|██████████████████████████████████████▌                                                   | 9/21 [00:01<00:01,  7.13it/s]data_point_idx: 5167 运行时间： 199.7 毫秒
 48%|██████████████████████████████████████████▍                                              | 10/21 [00:01<00:01,  6.47it/s]data_point_idx: 5741 运行时间： 83.9 毫秒
data_point_idx: 6315 运行时间： 78.78 毫秒
 57%|██████████████████████████████████████████████████▊                                      | 12/21 [00:01<00:01,  7.99it/s]data_point_idx: 6889 运行时间： 61.71 毫秒
data_point_idx: 7463 运行时间： 36.92 毫秒
data_point_idx: 8037 运行时间： 217.59 毫秒
 71%|███████████████████████████████████████████████████████████████▌                         | 15/21 [00:02<00:00,  8.59it/s]data_point_idx: 8611 运行时间： 199.72 毫秒
 76%|███████████████████████████████████████████████████████████████████▊                     | 16/21 [00:02<00:00,  7.60it/s]data_point_idx: 9185 运行时间： 244.61 毫秒
 81%|████████████████████████████████████████████████████████████████████████                 | 17/21 [00:02<00:00,  6.45it/s]data_point_idx: 9759 运行时间： 226.48 毫秒
 86%|████████████████████████████████████████████████████████████████████████████▎            | 18/21 [00:02<00:00,  5.84it/s]data_point_idx: 10333 运行时间： 225.03 毫秒
 90%|████████████████████████████████████████████████████████████████████████████████▌        | 19/21 [00:03<00:00,  5.42it/s]data_point_idx: 10907 运行时间： 79.41 毫秒
data_point_idx: 11481 运行时间： 51.11 毫秒
100%|█████████████████████████████████████████████████████████████████████████████████████████| 21/21 [00:03<00:00,  6.64it/s]
推理耗时列表： [74.84, 254.51, 202.34, 236.26, 78.77, 212.88, 230.37, 56.9, 95.81, 199.7, 83.9, 78.78, 61.71, 36.92, 217.59, 199.72, 244.61, 226.48, 225.03, 79.41, 51.11]
推理耗时列表大小： 21
均值： 149.89
最小值： 36.92
最大值： 254.51
TP50： 199.7
TP90： 236.26
TP99： 252.53
Faster Transformers (total latency: 3.1484735012054443 sec)
rouge1 : 21.230164168943517
--------------!!!!
rouge2 : 5.061263076607168
--------------!!!!
rougeL : 15.151048610040283
--------------!!!!
rougeLsum : 18.847945125211304
--------------!!!!
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

运行结果：

```
=============== Argument ===============
saved_dir: /workspace/model/megatron-models/c-model-fp16
in_file: /workspace/model/megatron-models/345m/release/
infer_gpu_num: 1
head_num: 16
trained_tensor_parallel_size: 1
processes: 16
weight_data_type: fp16
load_checkpoints_to_cpu: 1
vocab_path: /workspace/model/gpt2-vocab/gpt2-vocab.json
merges_path: /workspace/model/gpt2-vocab/gpt2-merges.txt
========================================
[INFO] Spent 0:00:04.113249 (h:m:s) to convert the model
```

---

```
python3 examples/pytorch/gpt/gpt_summarization_stat.py \
        --data_type fp16 \
        --lib_path /workspace/FasterTransformer/build/lib/libth_transformer.so \
        --summarize \
         --ft_model_location /workspace/model/megatron-models/c-model-fp32 \
         --hf_model_location /workspace/model/gpt2-tokenizer/gpt2-tokenizer
```
运行过程：
```
python3 examples/pytorch/gpt/gpt_summarization_stat.py \
>        --lib_path /workspace/FasterTransformer/build/lib/libth_transformer.so \
>         --data_type fp16 \
>         --lib_path /workspace/FasterTransformer/build/lib/libth_transformer.so \
>         --summarize \
>          --ft_model_location /workspace/model/megatron-models/c-model-fp32 \
>          --hf_model_location /workspace/model/gpt2-tokenizer/gpt2-tokenizer
~~~~~~~~~~~~~ args.summarize: True , args.test_hf: False
Found cached dataset cnn_dailymail (/workspace/data/ccdv-data/ccdv/cnn_dailymail/3.0.0/3.0.0/0107f7388b5c6fae455a5661bcd134fc22da53ea75852027040d8d1e997f101f)
100%|██████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 278.88it/s]
top_k: 2
top_p: 0.0
int8_mode: 0
random_seed: 5
temperature: 1
max_seq_len: 1024
max_batch_size: 1
repetition_penalty: 1
vocab_size: 50304
tensor_para_size: 1
pipeline_para_size: 1
lib_path: /workspace/FasterTransformer/build/lib/libth_transformer.so
ckpt_path: /workspace/model/megatron-models/c-model-fp32/1-gpu
hf_config: {'activation_function': 'gelu_new', 'architectures': ['GPT2LMHeadModel'], 'attn_pdrop': 0.1, 'bos_token_id': 50256, 'embd_pdrop': 0.1, 'eos_token_id': 50256, 'initializer_range': 0.02, 'layer_norm_epsilon': 1e-05, 'model_type': 'gpt2', 'n_ctx': 1024, 'n_embd': 768, 'n_head': 12, 'n_layer': 12, 'n_positions': 1024, 'resid_pdrop': 0.1, 'summary_activation': None, 'summary_first_dropout': 0.1, 'summary_proj_to_labels': True, 'summary_type': 'cls_index', 'summary_use_proj': True, 'task_specific_params': {'text-generation': {'do_sample': True, 'max_length': 50}}, 'vocab_size': 50257}
----------------wpe 1024 1024
[WARNING] gemm_config.in is not found; using default GEMM algo
[FT][WARNING] Skip NCCL initialization since requested tensor/pipeline parallel sizes are equals to 1.
[FT][INFO] Device NVIDIA H800
----------args.use_gpt_decoder_ops: False
---------------------------------------------------------
FT Generated :
 Article :  (CNN)James Best, best known for his portrayal of bumbling sheriff Rosco P. Coltrane on TV's "The Dukes of Hazzard," died Monday after a brief illness. He was 88. Best died in hospice in Hickory, North Carolina, of complications from pneumonia, said Steve Latshaw, a longtime friend and Hollywood colleague. Although he'd been a busy actor for decades in theater and in Hollywood, Best didn't become famous until 1979, when "The Dukes of Hazzard's" cornpone charms began beaming into millions of American homes almost every Friday night. For seven seasons, Best's Rosco P. Coltrane chased the moonshine-running Duke boys back and forth across the back roads of fictitious Hazzard County, Georgia, although his "hot pursuit" usually ended with him crashing his patrol car. Although Rosco was slow-witted and corrupt, Best gave him a childlike enthusiasm that got laughs and made him endearing. His character became known for his distinctive "kew-kew-kew" chuckle and for goofy catchphrases such as "cuff 'em and stuff 'em!" upon making an arrest. Among the most popular shows on TV in the early '80s, "The Dukes of Hazzard" ran until 1985 and spawned TV movies, an animated series and video games. Several of Best's "Hazzard" co-stars paid tribute to the late actor on social media. "I laughed and learned more from Jimmie in one hour than from anyone else in a whole year," co-star John Schneider, who played Bo Duke, said on Twitter. "Give Uncle Jesse my love when you see him dear friend." "Jimmy Best was the most constantly creative person I have ever known," said Ben Jones, who played mechanic Cooter on the show, in a Facebook post. "Every minute of his long life was spent acting, writing, producing, painting, teaching, fishing, or involved in another of his life's many passions." Born Jewel Guy on July 26, 1926, in Powderly, Kentucky, Best was orphaned at 3 and adopted by Armen and Essa Best, who renamed him James and raised him in rural Indiana. Best served in the Army during World War II before launching his acting career. In the 1950s and 1960s, he accumulated scores of credits, playing a range of colorful supporting characters in such TV shows as "The Twilight Zone," "Bonanza," "The Andy Griffith Show" and "Gunsmoke." He later appeared in a handful of Burt Reynolds' movies, including "Hooper" and "The End." But Best will always be best known for his "Hazzard" role, which lives on in reruns. "Jimmie was my teacher, mentor, close friend and collaborator for 26 years," Latshaw said. "I directed two of his feature films, including the recent 'Return of the Killer Shrews,' a sequel he co-wrote and was quite proud of as he had made the first one more than 50 years earlier." People we've lost in 2015 . CNN's Stella Chan contributed to this story.

 Highlights :  James Best, who played the sheriff on "The Dukes of Hazzard," died Monday at 88 .
"Hazzard" ran from 1979 to 1985 and was among the most popular shows on TV .

 Summary :   Best was a great actor and an even better friend.
<|endoftext|>.
---------------------------------------------------------
Using the latest cached version of the module from /root/.cache/huggingface/modules/evaluate_modules/metrics/evaluate-metric--rouge/b01e0accf3bd6dd24839b769a5fda24e14995071570870922c71970b3a6ed886 (last modified on Thu Jun 29 16:31:01 2023) since it couldn't be found locally at evaluate-metric--rouge, or remotely on the Hugging Face Hub.
Using the latest cached version of the module from /root/.cache/huggingface/modules/evaluate_modules/metrics/evaluate-metric--rouge/b01e0accf3bd6dd24839b769a5fda24e14995071570870922c71970b3a6ed886 (last modified on Thu Jun 29 16:31:01 2023) since it couldn't be found locally at evaluate-metric--rouge, or remotely on the Hugging Face Hub.
  0%|                                                                                                  | 0/21 [00:00<?, ?it/s]data_point_idx: 1 运行时间： 61.65 毫秒
data_point_idx: 575 运行时间： 203.57 毫秒
 10%|████████▌                                                                                 | 2/21 [00:00<00:02,  7.43it/s]data_point_idx: 1149 运行时间： 160.92 毫秒
 14%|████████████▊                                                                             | 3/21 [00:00<00:02,  6.86it/s]data_point_idx: 1723 运行时间： 66.13 毫秒
data_point_idx: 2297 运行时间： 64.81 毫秒
 24%|█████████████████████▍                                                                    | 5/21 [00:00<00:01,  9.76it/s]data_point_idx: 2871 运行时间： 167.7 毫秒
data_point_idx: 3445 运行时间： 183.41 毫秒
 33%|██████████████████████████████                                                            | 7/21 [00:00<00:01,  7.43it/s]data_point_idx: 4019 运行时间： 56.37 毫秒
data_point_idx: 4593 运行时间： 199.89 毫秒
 43%|██████████████████████████████████████▌                                                   | 9/21 [00:01<00:01,  7.56it/s]data_point_idx: 5167 运行时间： 159.9 毫秒
 48%|██████████████████████████████████████████▍                                              | 10/21 [00:01<00:01,  7.23it/s]data_point_idx: 5741 运行时间： 65.9 毫秒
data_point_idx: 6315 运行时间： 62.1 毫秒
 57%|██████████████████████████████████████████████████▊                                      | 12/21 [00:01<00:00,  9.07it/s]data_point_idx: 6889 运行时间： 47.46 毫秒
data_point_idx: 7463 运行时间： 29.16 毫秒
data_point_idx: 8037 运行时间： 172.23 毫秒
 71%|███████████████████████████████████████████████████████████████▌                         | 15/21 [00:01<00:00, 10.18it/s]data_point_idx: 8611 运行时间： 159.41 毫秒
data_point_idx: 9185 运行时间： 190.56 毫秒
 81%|████████████████████████████████████████████████████████████████████████                 | 17/21 [00:02<00:00,  8.24it/s]data_point_idx: 9759 运行时间： 178.58 毫秒
 86%|████████████████████████████████████████████████████████████████████████████▎            | 18/21 [00:02<00:00,  7.60it/s]data_point_idx: 10333 运行时间： 177.51 毫秒
 90%|████████████████████████████████████████████████████████████████████████████████▌        | 19/21 [00:02<00:00,  7.10it/s]data_point_idx: 10907 运行时间： 199.63 毫秒
 95%|████████████████████████████████████████████████████████████████████████████████████▊    | 20/21 [00:02<00:00,  6.49it/s]data_point_idx: 11481 运行时间： 39.95 毫秒
100%|█████████████████████████████████████████████████████████████████████████████████████████| 21/21 [00:02<00:00,  7.89it/s]
推理耗时列表： [61.65, 203.57, 160.92, 66.13, 64.81, 167.7, 183.41, 56.37, 199.89, 159.9, 65.9, 62.1, 47.46, 29.16, 172.23, 159.41, 190.56, 178.58, 177.51, 199.63, 39.95]
推理耗时列表大小： 21
均值： 126.04
最小值： 29.16
最大值： 203.57
TP50： 159.9
TP90： 199.63
TP99： 202.834
Faster Transformers (total latency: 2.647479772567749 sec)
rouge1 : 21.664839291439336
--------------!!!!
rouge2 : 5.412904663794084
--------------!!!!
rougeL : 15.354753879504434
--------------!!!!
rougeLsum : 18.928251038380523
--------------!!!!
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

执行结果：

```
python3 examples/pytorch/gpt/utils/megatron_fp8_ckpt_convert.py \
> -i /workspace/model/megatron-models/345m/release \
> -o /workspace/model/megatron-models/c-model-fp8-linear/ \
> -trained_tensor_parallel_size 1 \
> -i_g 1 \
> -head_num 16 \
> -weight_data_type fp32 \
> --vocab-path /workspace/model/gpt2-vocab/gpt2-vocab.json \
> --merges-path /workspace/model/gpt2-vocab/gpt2-merges.txt

=============== Argument ===============
saved_dir: /workspace/model/megatron-models/c-model-fp8-linear/
in_file: /workspace/model/megatron-models/345m/release
infer_gpu_num: 1
head_num: 16
trained_tensor_parallel_size: 1
processes: 16
weight_data_type: fp32
load_checkpoints_to_cpu: 1
vocab_path: /workspace/model/gpt2-vocab/gpt2-vocab.json
merges_path: /workspace/model/gpt2-vocab/gpt2-merges.txt
========================================
[INFO] Spent 0:00:04.251986 (h:m:s) to convert the model
```

---

```

```









