



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



## 初始化 Megatron

1. 定义模型的切割框架
2. 在此框架上，初始化进程，分配GPU，设置进程组（DP/TP/PP）






### megatron.initialize

- `initialize_megatron`:


- `_initialize_distributed`: 
1. 设置分布式环境：初始化进程，分配GPU，并设置进程大组（group）
2. 制定DP/TP/PP分组策略，设置进程子组（subgroup）
3. 设置DeepSpeed ZeRO-R，对activation进行优化





### megatron.core.parallel_state


- initialize_model_parallel


