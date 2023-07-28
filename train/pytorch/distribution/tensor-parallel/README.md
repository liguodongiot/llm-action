

```
from utils import cleanup, setup, ToyModel
model = ToyModel().cuda(0)
model
```

```
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A   1521759      C   /opt/conda/bin/python            3820MiB |
+-----------------------------------------------------------------------------+

```


```
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A   4112660      C   /opt/conda/bin/python            1382MiB |
|    1   N/A  N/A   4112661      C   /opt/conda/bin/python            1382MiB |
|    2   N/A  N/A   4112662      C   /opt/conda/bin/python            1370MiB |
|    3   N/A  N/A   4112663      C   /opt/conda/bin/python            1370MiB |
+-----------------------------------------------------------------------------+




```




Tensor Parallelism(TP) 建立在 DistributedTensor(DTensor) 之上，并提供多种并行格式：Rowwise、Colwise 和 Pairwise Parallelism。


Rowwise，对模块的行进行分区。假设输入是分片的 DTensor，则输出是仿制的 DTensor。

Colwise，对张量或模块的列进行分区。 假设输入是仿制的 DTensor，则输出是分片的 DTensor。

PairwiseParallel 将 colwise 和 rowwise 样式串联为固定对，就像 [Megatron-LM](https://arxiv.org/abs/1909.08053) 所做的那样。 我们假设输入和输出都需要复制 DTensor。

PairwiseParallel 目前仅支持 `nn.Multihead Attention`、`nn.Transformer` 或偶数层 `MLP`。

由于 Tensor Parallelism 是建立在 DTensor 之上的，因此我们需要使用 DTensor 指定模块的输入和输出位置，以便它可以在前后与模块进行预期的交互。 以下是用于输入/输出准备的函数：

torch.distributed.tensor.parallel.style.make_input_replicate_1d(input, device_mesh=None)












---

DeviceMesh 设备网格















