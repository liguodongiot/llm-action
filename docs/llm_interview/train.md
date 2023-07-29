



> 使用bf16和fp16进行半精度训练的优缺点

> DeepSpeed的特点是什么？各个 ZeRO Stage 都有什么用？



> 流水线并行能与DeepSpeed ZeRO 2/3一起训练吗？

PP + ZeRO 2/3 不推荐一起训练。 PP需要累积梯度（accumulate gradients），但 ZeRO2 需要对梯度进行分块（chunk）。 即使能够实现，也没有真正的性能提升。

将两者结合使用来提高效率并不容易，PP + ZeRO 2 实际上比 ZeRO2（无 PP）更慢且内存效率低。如果用户内存不足，用户可以使用 ZeRO3 代替 ZeRO2 + PP。而正因为如此，在 DeepSpeed 中， PP + ZeRO 2/3 之间不兼容。使用将 PP + ZeRO 1 进行组合使用。

即使该方法效率不高，但是 ColossalAI 为了支持更多的并行训练方法。ColossalAI 还是提供了 ZeRO 3 + PP + TP 一起组合的方案。

参考：
- https://github.com/microsoft/DeepSpeed/issues/1110
- https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/runtime/pipe/engine.py#L71
- https://github.com/hpcaitech/ColossalAI/issues/682
- https://github.com/hpcaitech/ColossalAI/pull/477


## 微调

> 介绍下 LoRA、AdaLoRA、QLoRA 这几种高效微调方法及其特点

  
> 在LoRA中，A和B低秩矩阵的初始化方法，对A采用高斯初始化，对B采用零矩阵初始化，目的是让训练刚开始时BA的值为0，这样不会给模型带来额外的噪声。那么，对A做零矩阵初始化，对B做高斯初始化行不行呢？反正看起来只要让初始化为0就行？

当前作者还没有发现转换初始化方式产生的显著区别，只要这两者中任意一者为0，另一者不为0即可。

参考：https://github.com/microsoft/LoRA/issues/98

> 介绍下 Prefix Tuning、Prompt Tuning、P-Tuning、P-Tuning v2 这四种高效微调方法的区别与联系？

1. Prompt Tuning和P-Tuning都是只在Embbedding层加入虚拟Token。而 Prefix Tuning、P-Tuning v2 会在每一层都加入虚拟Token，从而引入了更多的可训练参数；通过加入到更深层结构中的Prompt，能给模型预测带来更直接的影响。
2. P-Tuning通过 LSTM + MLP 去编码这些virtual token，再输入到模型，可以让模型收敛更快。
3. Prefix Tuning 为了防止直接更新 Prefix 的参数（virtual token）导致训练不稳定和性能下降的情况，在Prefix层前面加了MLP结构，训练完成后，只保留Prefix的参数。





## RLHF

> RLHF 完整训练过程是什么？RL建模过程中涉及到几个模型？
> RLHF 过程中RM随着训练过程的进行，得分越来越高，效果就一定好吗？

> 

- 
