



- Mastering LLM Techniques: Inference Optimization: https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/
- 7 Ways To Speed Up Inference of Your Hosted LLMs：https://betterprogramming.pub/speed-up-llm-inference-83653aa24c47
- How to Speed Up LLM Training with Distributed Systems?：https://www.appypie.com/blog/llm-training-with-distributed-systems
- 详谈大模型训练和推理优化技术：https://wjn1996.blog.csdn.net/article/details/130764843
- LLM 盛行，如何优雅地训练大模型？：https://cloud.tencent.com/developer/article/2321394?areaId=106001



底延迟、高吞吐量 （权衡）

首Token延迟，平均Token延迟


- 模型量化
	- AutoAWQ(W4A16)、SmoothQuant（W8A8），参考：TensorRT-LLM
	- GPTQ、bitsandbytes(LLM.int8())，参考：hf transformers
	- KV Cache 量化 （降低显存，从而进一步增加处理的batch大小）
- 分布式并行推理
	- 张量并行推理（降低延迟）
	- 流水线并行推理（提高吞吐量）
- 模型服务化调度优化：
	- 动态batch，参考：inference triton server
	- continuous batching,参考：vllm
- 投机采样
	- 使用一个小模型来做草稿，然后使用大模型做纠正检查。参考：flexflow server
- 模型编译优化：
	- AI编译前端优化：图算融合、内存分配、常量折叠、公共子表达式消除、死代码消除、代数化简
	- AI编译后端优化：算子融合、循环优化
- 显存优化：
	- 通过 PagedAttention 对 KV Cache 的有效管理，参考：vllm
	- CPU Offloading是将张量保存在CPU内存中，并且在计算时仅将张量复制到GPU。
- 低精度浮点数优化：
	- FP8（NVIDIA H系列GPU开始支持FP8，兼有FP16的稳定性和INT8的速度），Nvidia Transformer Engine 兼容 FP8 框架，主要利用这种精度进行 GEMM（通用矩阵乘法）计算，同时以 FP16 或 FP32 高精度保持主权重和梯度。  MS-AMP (使用FP8进行训练)
	- FP16 / BF16 



前端优化：输入计算图，关注计算图整体拓扑结构，而不关心算子的具体实现。在 AI 编译器的前端优化，对算子节点进行融合、消除、化简等操作，使计算图的计算和存储开销最小。
后端优化：关注算子节点的内部具体实现，针对具体实现使得性能达到最优。重点关心节点的输入，输出，内存循环方式和计算的逻辑。






简单的来说：

- 就是使用一个小模型来做草稿，然后使用大模型做纠正检查。
- 小模型的参数量要远小于原模型参数量一个级别才效果明显。
- 小模型和原模型的tokenizer最好一模一样，不然会增加额外的解码、编码时间。






## 对比下Flash Attention和Paged Attention


PagedAttention是应用在推理时，用分块内存和共享内存优化了KV cache的存储。减少了单个序列的显存，从而可以增大batch size，获得了更大的吞吐量。总的说，目标是推理时，减少显存，增大吞吐量。 这个团队之前有篇类似的工作FlexGen，推理时通过对KV cache进行高效的CPU卸载，提升了吞吐量。

Flash Attention则是训练和推理都可以使用，通过分块计算和kernel融合，减少了HBM访问次数，实现了计算加速，同时减少了显存占用。理论上，Flash Attention和Paged Attention可以组合使用。






