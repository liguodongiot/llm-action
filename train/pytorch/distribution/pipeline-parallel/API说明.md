



## torch.distributed.pipeline.sync.Pipe

包装任意的 nn.Sequential 模块以使用同步流水线并行进行训练。 


如果模块需要大量内存并且不适合单个 GPU，则流水线并行是一种用于训练的有用技术。该实现基于 torchgpipe 论文。

Pipe 将流水线并行与checkpointing相结合，以减少训练所需的峰值内存；同时，最大限度地减少设备利用率不足的问题。

您应该将所有模块放置在适当的设备上，并将它们包装到定义所需执行顺序的 nn.Sequential 模块中。 



如果模块不包含任何参数/buffers，则假定该模块应在 CPU 上执行，并且该模块的适当输入张量在执行之前已移动到 CPU。 此行为可以被 WithDevice 包装器覆盖，该包装器可用于显式指定模块应在哪个设备上运行。



参数说明：







