



## mtp

https://zhuanlan.zhihu.com/p/18056041194

DeepSeek-V3 Technical Report: https://arxiv.org/pdf/2412.19437
Better & faster large language models via multi-token prediction:https://arxiv.org/pdf/2404.19737


具体来说，在训练阶段，一次生成多个后续token，可以一次学习多个位置的label，进而有效提升样本的利用效率，提升训练速度；在推理阶段通过一次生成多个token，实现成倍的推理加速来提升推理性能。

