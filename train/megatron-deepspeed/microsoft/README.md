

## 测试环境

DeepSpeed v0.9.5

## 验证案例及结果

验证了 GPT-3 预训练的以下情况（同步之前/之后匹配训练/校验曲线、检查点保存/加载工作）：

- 使用 DeepSpeed ZeRO stage 1
- 使用 DeepSpeed ZeRO stage 1 和 Megatron-LM 的张量并行
- 使用 DeepSpeed ZeRO stage 1、Megatron-LM 的张量并行和 DeepSpeed 的流水线并行（即 3D 并行性）

此外，下面是同步前后的性能/收敛性比较。

## Flash attention

- https://github.com/Dao-AILab/flash-attention



## Rotary Positional Embedding (RoPE)




## 模型转换


```
PYTHONPATH=/workspace/code/Megatron-DeepSpeed-llama-20230815 \
python tools/convert_checkpoint/deepspeed_to_megatron.py --target_tp 1 --target_pp 1 \
--input_folder /workspace/code/Megatron-DeepSpeed-llama-20230815/tmp/global_step2500 \
--output_folder /workspace/output/llama-7b-megatron-pretrain


> tree -h workspace/output/llama-7b-megatron-pretrain
workspace/output/llama-7b-megatron-pretrain
├── [  32]  iter_0002500
│   └── [  40]  mp_rank_00
│       └── [ 12G]  model_optim_rng.pt
└── [   4]  latest_checkpointed_iteration.txt




cd /hf/transformers
PYTHONPATH=/workspace/code/Megatron-DeepSpeed-llama-20230815 \
python src/transformers/models/megatron_gpt2/convert_megatron_gpt2_checkpoint.py \
/workspace/output/llama-7b-megatron-pretrain/iter_0002500/mp_rank_00/model_optim_rng.pt
```




## LLaMA-7B


```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.125.06   Driver Version: 525.125.06   CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA A800 80G...  On   | 00000000:34:00.0 Off |                    0 |
| N/A   64C    P0    89W / 300W |  80407MiB / 81920MiB |    100%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA A800 80G...  On   | 00000000:35:00.0 Off |                    0 |
| N/A   66C    P0    93W / 300W |  73999MiB / 81920MiB |    100%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   2  NVIDIA A800 80G...  On   | 00000000:36:00.0 Off |                    0 |
| N/A   66C    P0    91W / 300W |  74149MiB / 81920MiB |    100%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   3  NVIDIA A800 80G...  On   | 00000000:37:00.0 Off |                    0 |
| N/A   68C    P0    98W / 300W |  73987MiB / 81920MiB |    100%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   4  NVIDIA A800 80G...  On   | 00000000:9B:00.0 Off |                    0 |
| N/A   71C    P0   299W / 300W |  59741MiB / 81920MiB |    100%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   5  NVIDIA A800 80G...  On   | 00000000:9C:00.0 Off |                    0 |
| N/A   71C    P0   289W / 300W |  59787MiB / 81920MiB |    100%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   6  NVIDIA A800 80G...  On   | 00000000:9D:00.0 Off |                    0 |
| N/A   69C    P0   262W / 300W |  59829MiB / 81920MiB |    100%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   7  NVIDIA A800 80G...  On   | 00000000:9E:00.0 Off |                    0 |
| N/A   69C    P0   308W / 300W |  59743MiB / 81920MiB |    100%      Default |
|                               |                      |             Enabled* |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      5260      C   /usr/bin/python                 80404MiB |
|    1   N/A  N/A      5261      C   /usr/bin/python                 73996MiB |
|    2   N/A  N/A      5262      C   /usr/bin/python                 74146MiB |
|    3   N/A  N/A      5265      C   /usr/bin/python                 73984MiB |
|    4   N/A  N/A      5266      C   /usr/bin/python                 59738MiB |
|    5   N/A  N/A      5267      C   /usr/bin/python                 59784MiB |
|    6   N/A  N/A      5268      C   /usr/bin/python                 59826MiB |
|    7   N/A  N/A      5269      C   /usr/bin/python                 59740MiB |
+-----------------------------------------------------------------------------+
```



## LLaMA-13B


```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.125.06   Driver Version: 525.125.06   CUDA Version: 12.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA A800 80G...  On   | 00000000:34:00.0 Off |                    0 |
| N/A   62C    P0    89W / 300W |  71639MiB / 81920MiB |    100%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA A800 80G...  On   | 00000000:35:00.0 Off |                    0 |
| N/A   65C    P0    92W / 300W |  77801MiB / 81920MiB |    100%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   2  NVIDIA A800 80G...  On   | 00000000:36:00.0 Off |                    0 |
| N/A   65C    P0    90W / 300W |  77847MiB / 81920MiB |    100%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   3  NVIDIA A800 80G...  On   | 00000000:37:00.0 Off |                    0 |
| N/A   69C    P0   209W / 300W |  77761MiB / 81920MiB |    100%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   4  NVIDIA A800 80G...  On   | 00000000:9B:00.0 Off |                    0 |
| N/A   70C    P0   236W / 300W |  58423MiB / 81920MiB |     99%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   5  NVIDIA A800 80G...  On   | 00000000:9C:00.0 Off |                    0 |
| N/A   71C    P0   156W / 300W |  63465MiB / 81920MiB |     99%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   6  NVIDIA A800 80G...  On   | 00000000:9D:00.0 Off |                    0 |
| N/A   66C    P0   228W / 300W |  57581MiB / 81920MiB |    100%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   7  NVIDIA A800 80G...  On   | 00000000:9E:00.0 Off |                    0 |
| N/A   66C    P0   174W / 300W |  57079MiB / 81920MiB |     99%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
```





