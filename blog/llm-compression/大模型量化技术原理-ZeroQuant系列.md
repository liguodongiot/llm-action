
近年来，随着Transformer、MOE架构的提出，使得深度学习模型轻松突破上万亿规模参数，从而导致模型变得越来越大，因此，我们需要一些大模型压缩技术来降低模型部署的成本，并提升模型的推理性能。
模型压缩主要分为如下几类：

-   剪枝（Pruning）
-   知识蒸馏（Knowledge Distillation）
-   量化

本系列将针对大模型的一些常见训练后量化方案（GPTQ、LLM.int8()、SmoothQuant、AWQ等）进行讲述。

- [大模型量化概述](https://www.zhihu.com/question/627484732/answer/3261671478)
- [大模型量化技术原理-GPTQ、LLM.int8()]()
- [大模型量化技术原理-SmoothQuant]()
- [大模型量化技术原理-AWQ、AutoAWQ]()
- [大模型量化技术原理-SpQR]()
- [大模型量化技术原理-ZeroQuant系列]()

而本文主要针对大模型量化技术 ZeroQuant 系列进行讲述。

- **ZeroQuant**: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers
- **ZeroQuant-V2**: Exploring Post-training Quantization in LLMs from Comprehensive Study to Low Rank Compensation
- **ZeroQuant-FP**: A Leap Forward in LLMs Post-Training W4A8 Quantization Using Floating-Point Formats
- **ZeroQuant-HERO**: Hardware-Enhanced Robust Optimized Post-Training Quantization Framework for W8A8 Transformers


## ZeroQuant




## ZeroQuant-V2



## ZeroQuant-FP


## ZeroQuant-HERO













参考文档：

- https://github.com/microsoft/DeepSpeed
- ZeroQuant: https://arxiv.org/pdf/2206.01861.pdf
- ZeroQuant-V2: https://arxiv.org/abs/2303.08302
- ZeroQuant-FP: https://arxiv.org/pdf/2307.09782.pdf
- ZeroQuant-HERO: https://arxiv.org/pdf/2310.17723.pdf
