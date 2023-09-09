


# DDP

- https://zhuanlan.zhihu.com/p/343951042



## 数据并行

当一张 GPU 可以存储一个模型时，可以采用数据并行得到更准确的梯度或者加速训练，即每个 GPU 复制一份模型，将一批样本分为多份输入各个模型并行计算。因为求导以及加和都是线性的，数据并行在数学上也有效。

## DP


model = nn.DataParallel(model)



## DDP


DDP 通过 Reducer 来管理梯度同步。为了提高通讯效率， Reducer 会将梯度归到不同的桶里（按照模型参数的 reverse order， 因为反向传播需要符合这样的顺序），一次归约一个桶。其中，桶的大小为参数 bucket_cap_mb 默认为 25，可根据需要调整。


可以看到每个进程里，模型参数都按照倒序放在桶里，每次归约一个桶。


DDP 通过在构建时注册 autograd hook 进行梯度同步。反向传播时，当一个梯度计算好后，相应的 hook 会告诉 DDP 可以用来归约。

当一个桶里的梯度都可以了，Reducer 就会启动异步 allreduce 去计算所有进程的平均值。allreduce 异步启动使得 DDP 可以边计算边通信，提高效率。

当所有桶都可以了，Reducer 会等所有 allreduce 完成，然后将得到的梯度写到 param.grad。




### DDP+MP

DDP与流水线并行。







## launch



通过 launch.py 启动，在 8 个 GPU 节点上，每个 GPU 一个进程：
```
python /home/guodong.li/virtual-venv/megatron-ds-venv-py310-cu117/lib/python3.10/site-packages/torch/distributed/launch.py --nnode=1 --node_rank=0 --nproc_per_node=8 example.py --local_world_size=8

python /home/guodong.li/virtual-venv/megatron-ds-venv-py310-cu117/lib/python3.10/site-packages/torch/distributed/launch.py --nnode=1 --node_rank=0 --nproc_per_node=1 example.py --local_world_size=1

```




## torch elastic/torchrun


```
torchrun --nnodes=2 --nproc_per_node=8 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29400 elastic_ddp.py
```

我们在两台主机上运行 DDP 脚本，每台主机运行 8 个进程，也就是说，我们在 16 个 GPU 上运行它。 

请注意，所有节点上的 $MASTER_ADDR 必须相同。

这里torchrun将启动8个进程，并在启动它的节点上的每个进程上调用elastic_ddp.py，
但用户还需要应用slurm等集群管理工具才能在2个节点上实际运行此命令。


例如，在启用 SLURM 的集群上，我们可以编写一个脚本来运行上面的命令并将 MASTER_ADDR 设置为：

```
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
```

然后我们可以使用 SLURM 命令运行此脚本：

```
srun --nodes=2 ./torchrun_script.sh
```

当然，这只是一个例子； 您可以选择自己的集群调度工具来启动torchrun作业。



- 详细启动命令：https://pytorch.org/docs/stable/elastic/quickstart.html




```
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29400 elastic_ddp.py
```


```
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=10.xx.2.46:29400 multigpu_torchrun.py --batch_size 32  10 5
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=10.xx.2.46:29400 multigpu_torchrun.py --batch_size 32  10 5
```





## 官方文档
- DDP 教程：https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
- DDP 设计：https://pytorch.org/docs/master/notes/ddp.html
- DDP 示例：
	- https://github.com/pytorch/examples/tree/main/distributed/ddp (官方没有更新了)
	- https://github.com/pytorch/examples/tree/main/distributed/ddp-tutorial-series




# FSDP



- GETTING STARTED WITH FULLY SHARDED DATA PARALLEL(FSDP)：https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
- ADVANCED MODEL TRAINING WITH FULLY SHARDED DATA PARALLEL (FSDP)：https://pytorch.org/tutorials/intermediate/FSDP_adavnced_tutorial.html










# miniGPT

- https://pytorch.org/tutorials/intermediate/ddp_series_minGPT.html

用于训练的文件：

- trainer.py ：包含 Trainer 类，该类使用提供的数据集在模型上运行分布式训练迭代。
- model.py ：定义模型架构。
- char_dataset.py ：包含字符级数据集的 Dataset 类。
- gpt2_train_cfg.yaml ：包含数据、模型、优化器和训练运行的配置。
- main.py ：训练作业的入口点。 它设置 DDP 进程组，读取所有配置并运行训练作业。








