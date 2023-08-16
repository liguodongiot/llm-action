

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




