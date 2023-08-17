




## megatron.training

### pretrain

```
构建数据集
from megatron.data.gpt_dataset import build_train_valid_test_datasets


预训练函数

from megatron.training import pretrain

Main training program.
This function will run the followings in the order provided:
    1) initialize Megatron.
    2) setup model, optimizer and lr schedule using the model_provider.
    3) call train_val_test_data_provider to get train/val/test datasets.
    4) train the modle using the forward_step_func.

```

#### 初始化 Megatron

1. 定义模型的切割框架
2. 在此框架上，初始化进程，分配GPU，设置进程组（DP/TP/PP）




### megatron.initialize

- `initialize_megatron`:


- `_initialize_distributed`: 
1. 设置分布式环境：初始化进程，分配GPU，并设置进程大组（group）
2. 制定DP/TP/PP分组策略，设置进程子组（subgroup）
3. 设置DeepSpeed ZeRO-R，对activation进行优化





### megatron.core.parallel_state

- initialize_model_parallel：初始化模型数据并行组

假设我们总共有 16 个 GPU，用 g0 ... g15 表示，我们使用 2 个 GPU 来张量并行，使用 4 个 GPU 来流水线并行。 当前函数将创建 8 个张量模型并行组、4 个流水线模型并行组和 8 个数据并行组，如下所示：

- 8 个数据并行组：[g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15] ]
- 8 个张量模型并行组：[g0，g1]，[g2，g3]，[g4，g5]，[g6，g7]，[g8，g9]，[g10，g11]，[g12，g13]，[g14，g15]
- 4 个管道模型并行组：[g0、g4、g8、g12]、[g1、g5、g9、g13]、[g2、g6、g10、g14]、[g3、g7、g11、g15]




