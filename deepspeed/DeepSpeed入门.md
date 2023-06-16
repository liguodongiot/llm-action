


## DeepSpeed 

通过简单三步将Pytorch DDP模型训练改造 DeepSpeed DP 模型训练。

第一步：**初始化DeepSpeed引擎**:
```
model_engine, optimizer, _, _ = deepspeed.initialize(args=cmd_args,
                                                     model=model,
                                                     model_parameters=params)
```
deepspeed.initialize确保在底层适当地完成了所需的分布式数据并行或混合精度训练所需的所有设置。



第二步：**初始化分布式环境**:
```
deepspeed.init_distributed()
```

DeepSpeed将在其初始化期间自动初始化分布式环境，因此，可以不使用此函数。


第三步，**模型训练**

使用三个简单的API来进行前向传播（callable object）、反向传播（backward）和权重更新（step）来训练模型。

```
for step, batch in enumerate(data_loader):
    #forward() method
    loss = model_engine(batch)

    #runs backpropagation
    model_engine.backward(loss)

    #weight update
    model_engine.step()
```

- Gradient Averaging: 在分布式数据并行训练中，backward 确保在对一个 train_batch_size 进行训练后，梯度在数据并行进程间进行平均。
- Loss Scaling: 在FP16/混合精度训练中, DeepSpeed 引擎会自动处理缩放损失,以避免梯度中的精度损失。
- Learning Rate Scheduler: 当使用 DeepSpeed 的学习率调度器(在ds_config.json文件中指定)时, DeepSpeed 会在每次训练步骤(执行model_engine.step()时)调用调度器的step()方法。当不使用DeepSpeed的学习率调度器时:
  -  如果调度期望在每次训练步骤都执行, 那么用户可以在初始化 DeepSpeed 引擎时将调度器传递给 deepspeed.initialize, 让 DeepSpeed 进行管理、更新或保存/恢复。
  -  如果调度应该在任何其它间隔（例如训练周期）执行，则用户在初始化期间不应将调度传递给 DeepSpeed，必须显式地管理它。





## 多节点环境变量

当在多个节点上进行训练时，我们发现支持传播用户定义的环境变量非常有用。

默认情况下，DeepSpeed 将传播所有设置的 NCCL 和 PYTHON 相关环境变量。

如果您想传播其它变量，可以在名为 .deepspeed_env 的文件中指定它们，该文件包含一个行分隔的 VAR=VAL 条目列表。

DeepSpeed 启动器将查找你执行的本地路径以及你的主目录（~/）。

以一个具体的例子来说明，有些集群需要在训练之前设置特殊的 NCCL 变量。

用户可以简单地将这些变量添加到其主目录中的 `.deepspeed_env` 文件中，该文件如下所示：
```
NCCL_IB_DISABLE=1
NCCL_SOCKET_IFNAME=eth0
```
DeepSpeed 然后会确保在启动每个进程时在整个训练工作的每个节点上设置这些环境变量。


## 兼容MPI 

如上所述，DeepSpeed 提供了自己的并行启动器来帮助启动多节点/多GPU训练作业。如果您喜欢使用MPI（例如: mpirun）启动训练作业，则我们提供对此的支持。

需要注意的是，DeepSpeed 仍将使用 torch 分布式 NCCL 后端，而不是 MPI 后端。

要使用 mpirun + DeepSpeed （使用 mpirun 作为启动器后端）启动你的训练作业，您只需要安装 mpi4py Python 包。DeepSpeed 将使用它来发现 MPI 环境，并将必要的状态（例如 world size、rank 等）传递给 torch 分布式后端。

如果你正在使用模型并行，流水线并行或者在调用 deepspeed.initialize(..) 之前需要使用 torch.distributed 调用，我们为你提供了额外的 DeepSpeed API 调用以支持相同的 MPI。请将您的初始 torch.distributed.init_process_group(..) 调用替换为：deepspeed.init_distributed()








