



https://github.com/microsoft/DeepSpeed/blob/master/blogs/deepspeed-fp6/03-05-2024/README-Chinese.md






INT4量化技术的挑战：

虽然这些技术可以减小模型大小和参数存储量，但由于过拟合问题, 它们在更一般的许多任务中往往表现不佳，包括代码生成和摘要等更多生成任务。


FP6的突破：

FP6数据格式在当前AI硬件的高效支持中存在挑战。该格式在各种任务的性能和灵活性方面均表现出色。

为了提高FP6在当前主流AI硬件上的执行效率，我们提出了一种4+2新颖的FP6 GPU kernel方案。这一创新使FP6成为提高LLMs效率的有效途径。





## 开创性的全栈GPU KERNEL设计


运行前比特层级的数据排布转换。用以解决权重具有不规则位宽时不友好的内存访问挑战，实现GPU内存的最优访问；

运行时的高效SIMT计算。用以最小化权重反量化的运行时开销；

全栈的高效流水线设计。其SIMT计算、Tensor Core计算和GPU内存访问进行高效调度，最大程度提升性能。


FP6 kernel在NVIDIA A100 GPU上进行（因decoder的矩阵形状狭长而导致参数矩阵的访存成为瓶颈的）矩阵乘法时，处理速度比FP16 cuBLAS基准提高了2.1倍。



FP6服务LLM


## FP6服务LLM



FP6量化为模型推理提供了两个关键好处：
它使大型语言模型（LLMs）能够在更少的GPU上部署——例如，LLaMA-70b在单个A100-80G GPU上就能以FP6形式运行，而FP16模型至少需要两个GPU。

此外，它显著加快了小batch之下内存访问为瓶颈的线性层计算。

此外，FP6量化减少了模型权重的GPU内存需求，允许同时服务更多查询，从而提高了服务吞吐量。





较长解码场景中内存访问瓶颈增强的两个因素如下：

首先，KV缓存的内存使用随序列长度增加而增加，减少了可容纳的batch大小并导致线性层的矩阵计算瓶颈变为参数的访存。

其次，在DeepSpeed-FastGen的prefill-decoding-mixed-batch技术背景下，对于decoding较长的情况，用于和decoding进行mixed-batching的prefill切块会相对不足，这导致纯粹用于decoding的batch频率增加，进一步加剧了访存的瓶颈。



在GEMM因batch较大或有充足的GPU内存而使得瓶颈变为Tensor Core计算时，我们的仅限权重的量化kernel可能无法保持其性能优势，尤其是与厂商的优化库如cuBlas相比。然而，我们系统的低内存占用仍是一个关键优势。目前的支持限于非混合专家（Non-MoE）结构，我们正在努力将支持扩展到MoE结构。此外，当前系统仅与FP16输入模型兼容，因为当前实现的FP6 kernel仅支持处理FP16的激活。








