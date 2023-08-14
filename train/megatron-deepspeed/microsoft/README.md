



## 验证案例及结果

验证了 GPT-3 预训练的以下情况（同步之前/之后匹配训练/校验曲线、检查点保存/加载工作）：

- 使用 DeepSpeed ZeRO stage 1
- 使用 DeepSpeed ZeRO stage 1 和 Megatron-LM 的张量并行
- 使用 DeepSpeed ZeRO stage 1、Megatron-LM 的张量并行和 DeepSpeed 的流水线并行（即 3D 并行性）

此外，下面是同步前后的性能/收敛性比较。

## Flash attention



## Rotary Positional Embedding (RoPE)


## 
