

# 源码解读

- https://www.armcvai.cn/2024-10-30/llm-smoothquant.html
- https://www.armcvai.cn/2024-10-31/smoothquant-inplement.html


# SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models

- https://huggingface.co/mit-han-lab/opt-6.7b-smoothquant







- 英特尔模型压缩库-SQ：https://github.com/intel/neural-compressor/blob/master/docs/source/smooth_quant.md
- 英特尔模型压缩库-仅权重量化方法：https://github.com/intel/neural-compressor/blob/master/docs/source/quantization_weight_only.md



## 概要

大语言模型 (LLM) 显示出了出色的性能，但属于计算和内存密集型。 量化可以减少内存并加速推理。 
然而，现有方法无法同时保持准确性和硬件效率。 我们提出了 SmoothQuant，这是一种免训练、同时保持精度的通用训练后量化 (PTQ) 解决方案，
可为 LLM 实现 8 位权重、8 位激活 (W8A8) 量化。 

基于权重易于量化而激活却不易量化的事实，SmoothQuant 通过使用数学上等效的变换平滑激活异常值，从而将量化难度从激活离线迁移到权重

SmoothQuant 支持 LLM 中所有矩阵乘法的权重和激活的 INT8 量化，包括 OPT、BLOOM、GLM、MT-NLG 和 LLaMA 系列。
我们证明了 LLM 的加速速度高达 1.56 倍，内存减少了 2 倍，而精度损失可以忽略不计。 
SmoothQuant 支持在单个节点内提供 530B 的 LLM 服务。 我们的工作提供了一个交钥匙解决方案，可以降低硬件成本并使LLM普及。






## 结论 

我们提出了 SmoothQuant，这是一种准确且高效的训练后量化方法，可为高达 530B 参数的 LLM 实现无损 8 位权重和激活量化。 
SmoothQuant 能够对 LLM 中的所有 GEMM 的权重和激活进行量化，与混合精度激活量化基线相比，这显著减少了推理延迟和内存使用量。 
我们将 SmoothQuant 集成到 PyTorch 和 FasterTransformer 中，获得高达 1.56 倍的推理加速并将内存占用减半。 
SmoothQuant 通过提供了一种交钥匙解决方案来降低服务成本，使LLM的应用普及。




