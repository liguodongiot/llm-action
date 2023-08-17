

## pretrain_gpt

### train_valid_test_datasets_provider

### model_provider

### data_post_process



---


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

初始化 Megatron

1. 定义模型的切割框架
2. 在此框架上，初始化进程，分配GPU，设置进程组（DP/TP/PP）



---


### setup_model_and_optimizer

Model, optimizer, and learning rate.

设置模型及优化器




### build_train_valid_test_data_iterators

构建预训练数据迭代器


### train 

训练



### get_model: 构建模型





---


## megatron.initialize

### initialize_megatron


### _initialize_distributed

1. 设置分布式环境：初始化进程，分配GPU，并设置进程大组（group）
2. 制定DP/TP/PP分组策略，设置进程子组（subgroup）
3. 设置DeepSpeed ZeRO-R，对activation进行优化

- 初始化进程组：torch.distributed.init_process_group
- 设置张量模型并行、流水线模型并行和数据并行通信器：mpu.initialize_model_parallel



## megatron.core.parallel_state

### initialize_model_parallel：初始化模型数据并行组

假设我们总共有 16 个 GPU，用 g0 ... g15 表示，我们使用 2 个 GPU 来张量并行，使用 4 个 GPU 来流水线并行。 当前函数将创建 8 个张量模型并行组、4 个流水线模型并行组和 8 个数据并行组，如下所示：

- 8 个数据并行组：[g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15]
- 8 个张量模型并行组：[g0，g1]，[g2，g3]，[g4，g5]，[g6，g7]，[g8，g9]，[g10，g11]，[g12，g13]，[g14，g15]
- 4 个管道模型并行组：[g0、g4、g8、g12]、[g1、g5、g9、g13]、[g2、g6、g10、g14]、[g3、g7、g11、g15]

---



## megatron.model.gpt_model


### GPTModel




## megatron.model.module


### MegatronModule

Megatron 针对 torch.nn.Module 的特定扩展以支持流水线。



### Float16Module




## megatron.model.transformer

### ParallelTransformer

### ParallelTransformerLayer

一个 transformer 层

Transformer 层接受大小为 [s, b, h] 的输入并返回相同大小的输出。


### ParallelAttention

并行自注意力层抽象类。


### ParallelMLP

并行 MLP 层。


















## megatron.model.language_model


### TransformerLanguageModel

Transformer 语言模型


### Embedding

语言模型 Embedding




## megatron.core.tensor_parallel.cross_entropy

### _VocabParallelCrossEntropy

计算交叉熵。






## megatron.core.tensor_parallel.layers


### VocabParallelEmbedding


### ColumnParallelLinear

列并行线性层

### RowParallelLinear

行并行线性层


### _initialize_affine_weight_gpu

初始化 GPU 上模型并行的仿射权重。




## megatron.core.tensor_parallel.mappings


### copy_to_tensor_model_parallel_region

### _CopyToModelParallelRegion

将输入传递到模型并行区域。

### gather_from_tensor_model_parallel_region
### _GatherFromModelParallelRegion

从模型并行区域收集输入拼接在一起。


### _ScatterToModelParallelRegion 

分割输入并仅将相应的chuck保留到rank中



### _ReduceFromModelParallelRegion

All-reduce来自模型并行区域的输入




## megatron.core.model_parallel_config


### ModelParallelConfig

Megatron Core 基础配置



---


## megatron.training


### build_train_valid_test_datasets(build_train_valid_test_datasets_provider)

构建预训练数据集





## megatron.data.data_samplers

### build_pretraining_data_loader(dataset, consumed_samples)

给定一个数据集构建数据加载器







