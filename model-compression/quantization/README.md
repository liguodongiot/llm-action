
## 简介

- https://docs.nvidia.com/deeplearning/tensorrt/tensorflow-quantization-toolkit/docs/docs/qat.html
- https://github.com/HuangOwen/Awesome-LLM-Compression

一文总结当下常用的大型 transformer 效率优化方案
- https://zhuanlan.zhihu.com/p/604118644
- https://lilianweng.github.io/posts/2023-01-10-inference-optimization/

在深度神经网络上应用量化策略有两种常见的方法：

- 训练后量化（PTQ）：首先需要模型训练至收敛，然后将其权重的精度降低。与训练过程相比，量化操作起来往往代价小得多；
- 量化感知训练 (QAT)：在预训练或进一步微调期间应用量化。QAT 能够获得更好的性能，但需要额外的计算资源，还需要使用具有代表性的训练数据。


值得注意的是，理论上的最优量化策略与实际在硬件内核上的表现存在着客观的差距。**由于 GPU 内核对某些类型的矩阵乘法（例如 INT4 x FP16）缺乏支持，并非下面所有的方法都会加速实际的推理过程**。





## Post Training Quantization(PTQ)

- ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers
  - https://www.deepspeed.ai/tutorials/model-compression/
  - 集成在Deepspeed
- SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models
  - https://github.com/mit-han-lab/smoothquant
  - 已经集成在[FasterTransformer](https://github.com/NVIDIA/FasterTransformer)
- GPTQ: Accurate Post-training Compression for Generative Pretrained Transformers
- AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration
  - https://github.com/mit-han-lab/llm-awq

## Quantization Aware Training（QAT）

- LLM-QAT: Data-Free Quantization Aware Training for Large Language Models
  - https://github.com/facebookresearch/LLM-QAT


