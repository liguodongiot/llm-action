



## torch.distributed.pipeline.sync.Pipe

包装任意的 nn.Sequential 模块以使用同步流水线并行进行训练。 


如果模块需要大量内存并且不适合单个 GPU，则流水线并行是一种用于训练的有用技术。该实现基于 torchgpipe 论文。

Pipe 将流水线并行与checkpointing相结合，以减少训练所需的峰值内存；同时，最大限度地减少设备利用率不足的问题。

您应该将所有模块放置在适当的设备上，并将它们包装到定义所需执行顺序的 nn.Sequential 模块中。 



如果模块不包含任何参数/buffers，则假定该模块应在 CPU 上执行，并且该模块的适当输入张量在执行之前已移动到 CPU。 此行为可以被 WithDevice 包装器覆盖，该包装器可用于显式指定模块应在哪个设备上运行。



参数说明：

module (nn.Sequential) – 使用流水线进行并行的sequential模块。 sequential中的每个模块都必须在单个设备上具有其所有参数。 序列中的每个模块必须是 nn.Module 或 nn.Sequential（在单个设备上组合多个sequential模块）

chunks (int) – 微批次数量（默认值：1）

checkpoint (str) – 何时启用检查点，always、except_last或never之一（默认值： except_last）。 never：完全禁用检查点，except_last：启用除最后一个之外的所有微批次的检查点，always：启用所有微批次的检查点。

deferred_batch_norm (bool) – 是否使用延迟 BatchNorm 移动统计（默认值：False）。 如果设置为 True，我们将跟踪多个微批次的统计信息，以更新每个小批次的运行统计信息。



```
# Need to initialize RPC framework first.
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'
torch.distributed.rpc.init_rpc('worker', rank=0, world_size=1)

# Build pipe.
fc1 = nn.Linear(16, 8).cuda(0)
fc2 = nn.Linear(8, 4).cuda(1)
model = nn.Sequential(fc1, fc2)
model = Pipe(model, chunks=8)
input = torch.rand(16, 16).cuda(0)
output_rref = model(input)
```


注：仅当 Pipe 的检查点参数为“never”时，才可以使用 torch.nn.parallel.DistributedDataParallel 包装 Pipe 模型。


Pipe 目前仅支持节点内（intra-node）流水线，未来将扩展支持节点间（inter-node ）流水线。 forward 函数返回一个 RRef 以允许将来进行节点间（inter-node ）流水线传输，其中输出可能位于远程主机上。 对于节点内流水线，您可以使用 local_value() 在本地检索输出。



警告： Pipe 是实验性的，可能会发生变化。



### `forward(*inputs)`



## 跳过连接

某些模型（例如 ResNeXt）不是完全顺序的，并且层之间具有跳跃连接。 天真的实现作为管道并行性的一部分意味着我们需要通过多个 GPU 复制某些层的输出，直到我们最终到达跳跃连接层所在的 GPU。 为了避免这种复制开销，我们提供了以下 API 来在模型的不同层中存储和弹出张量。








