


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


