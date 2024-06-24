


> 模型训练通常关注的性能指标有哪些？

在模型训练过程中，通常关注的性能指标如下：

| 指标名称  | 单位                 |  指标含义       |
| :---- | :----------------- | ----- |
| 吞吐率   | samples/s、tokens/s | 单位时间（例如1s）内处理的Token数/训练样本数  |
| 单步时间  | s                  |   执行一个step所花费的时间  |
| 线性度、加速比   | values             | 单卡训练扩展到多卡，单机拓展到集群的效率度量指标  |
| 内存占用  | 百分比                | -  |
| 带宽占比  | 百分比                | -  |
| 训练效率  | tokens/day         | -  |
| 浮点运算  | TFLOPS           | 每秒浮点运算次数，是计算设备的计算性能指标  |
| 模型算力利用率（Model FLOPs Utilization， MFU）| 百分比  | 模型一次前反向计算消耗的矩阵算力与机器算力的比值  |
| 硬件算力利用率（Hardware FLOPs Utilization， HFU）| 百分比 | 考虑重计算后，模型一次前反向计算消耗的矩阵算力与机器算力的比值  |


在计算性能指标时，通常的优先级排序为：**吞吐率 > 单步迭代时间 > 线性度 > 内存占用 > 带宽占用 > 训练效率 > 浮点计算次数每秒 > 算力利用率**。




> 混合精度训练使用半精度训练的优缺点

半精度训练优点：跑得快+省显存

半精度训练缺点：精度（下溢+舍入误差）的问题。

使用了半精度训练，一般会采用一些捆绑的技术来弥补半精度的缺点。

- FP32 权重备份：对权重备份一份float32版本的版本，在梯度更新的时候避免float16精度不够而发生舍入误差导致的无效梯度更新，但是这样会占用额外的权重的内存，不过这些显存在一些情况下并不致命也就是了。

- loss scale：由于下溢的问题，也就是训练后期，梯度会很小，float16 容易产生 underflow，所以可以对loss做scale操作，毕竟loss的scale也会作用在梯度上（链式法则），这样一个大的scale比每个梯度都scale下要划算很多。


layer norm的层可能会完全使用float32，因为需要计算一组值的均值和方差，而这需要进行加法和除法运算，所以float16可能会出岔子。

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


## PyTorch











