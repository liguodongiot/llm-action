



https://github.com/flashinfer-ai/flashinfer
https://flashinfer.ai/whl/cu124/torch2.6/flashinfer-python/




用FlashInfer加速大语言模型推理中的自注意力操作：https://zhuanlan.zhihu.com/p/681506469




FlashInfer优化了分组自注意力，融合旋转位置编码的自注意力 和 量化自注意力 操作。


- 使用CUDA Cores的传统GQA实现会被算力所限制。FlashInfer提出使用预填充阶段的自注意力内核（使用Tensor Cores来实现）用于GQA的解码自注意力操作
- 融合旋转位置编码的自注意力
- 量化自注意力 KV-Cache 4bit
- FlashInfer中PageAttention实现预取（prefetch）了页表结构的索引，最小化page大小对于算子性能的影响。



RoPE 需要 sin/cos 等计算，不能使用 Tensor Cores加速。





FlashInfer中DeepSeek MLA的内核设计：https://zhuanlan.zhihu.com/p/25920092499