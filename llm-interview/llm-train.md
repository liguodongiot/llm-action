



> 使用bf16和fp16进行半精度训练的优缺点



## DeepSpeed


> DeepSpeed的特点是什么？各个 ZeRO Stage 都有什么用？



> 流水线并行能与DeepSpeed ZeRO 2/3一起训练吗？


https://www.zhihu.com/question/652836990/answer/3468210626

PP + ZeRO 2/3 不推荐一起训练。 PP需要累积梯度（accumulate gradients），但 ZeRO2 需要对梯度进行分块（chunk）。 即使能够实现，也没有真正的性能提升。

将两者结合使用来提高效率并不容易，PP + ZeRO 2 实际上比 ZeRO2（无 PP）更慢且内存效率低。如果用户内存不足，用户可以使用 ZeRO3 代替 ZeRO2 + PP。而正因为如此，在 DeepSpeed 中， PP + ZeRO 2/3 之间不兼容。使用将 PP + ZeRO 1 进行组合使用。

即使该方法效率不高，但是 ColossalAI 为了支持更多的并行训练方法。ColossalAI 还是提供了 ZeRO 3 + PP + TP 一起组合的方案。

参考：
- https://github.com/microsoft/DeepSpeed/issues/1110
- https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/runtime/pipe/engine.py#L71
- https://github.com/hpcaitech/ColossalAI/issues/682
- https://github.com/hpcaitech/ColossalAI/pull/477






---





