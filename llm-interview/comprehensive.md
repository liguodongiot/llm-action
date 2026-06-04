


- 解决显存不足的方法有哪些？

训练：


a.减小 Batch Size 与梯度累加（Gradient Accumulation）
  
  直接减小单次输入模型的 Batch Size 可以显著降低显存占用。为了不影响模型收敛效果，可以使用梯度累加技术，即执行多次前向和反向传播?>后再更新一次参数，从而在逻辑上实现大 Batch Size 的效果。


b.混合精度训练（Mixed Precision Training）

  使用 FP16（半精度）或 BF16 代替传统的 FP32（单精度）进行训练。这不仅能将模型权重和激活值的显存占用减少一半，还能加速计算。

c.梯度检查点（Gradient Checkpointing / Activation Recomputation）

  在正向传播时不保存所有的中间激活值（Activations），而是在反向传播需要用到时，重新计算这些激活值。这是一种“用时间换空间”的策略，能大幅减少显存占用。

推理：

a.模型量化（Quantization）
  
  将模型的权重从 FP16 压缩到 INT8 甚至 INT4（如 GPTQ、AWQ、GGUF/GGML 等格式）。这可以在几乎不损失太多生成质量的前提下，将显存需求降低 2 倍到 4 倍。

b.KV Cache 优化（PagedAttention）

  在 LLM 推理时，KV Cache（键值缓存）会随着上下文长度的增加而占用大量显存，且容易产生内存碎片。使用 vLLM 等推理框架中的 PagedAttention 技术，可以像操作系统管理虚拟内存一样分页管理 KV Cache，极大提高显存利用率。

c.FlashAttention
  
  这是一种硬件感知且内存高效的注意力机制算法。它通过优化 GPU 的 SRAM 和 HBM 之间的读写操作，不仅加快了推理速度，还显著降低了长上下文时的显存占用。

- 



