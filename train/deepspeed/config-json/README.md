- https://www.deepspeed.ai/docs/config-json/


## Batch Size 相关的参数


train_batch_size 必须等于 train_micro_batch_size_per_gpu * gradient_accumulation * gpu数量



### train_batch_size


### train_micro_batch_size_per_gpu


### gradient_accumulation_steps


## Optimizer 参数

- type：优化器名称。 DeepSpeed 原生支持 Adam、AdamW、OneBitAdam、Lamb 和 OneBitLamb 优化器，同时，也可以从 torch 中导入其他优化器。
   - https://deepspeed.readthedocs.io/en/latest/optimizers.html#optimizers
   - https://pytorch.org/docs/stable/optim.html
- params：用于实例化优化器的参数字典。参数名称必须与优化器构造函数签名匹配（例如，Adam）。
   - https://pytorch.org/docs/stable/optim.html#algorithms
   - https://pytorch.org/docs/stable/generated/torch.optim.Adam.html

Adam 优化器示例：

```
"optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.001,
      "betas": [
        0.8,
        0.999
      ],
      "eps": 1e-8,
      "weight_decay": 3e-7
    }
  }
```



## Scheduler 参数

当执行 model_engine.step() 时，DeepSpeed 在每个训练步骤调用 scheduler 的 step() 方法。

- type：学习率调度器名，DeepSpeed 提供了 LRRangeTest、OneCycle、WarmupLR、WarmupDecayLR 学习率调度器的实现。
   - https://deepspeed.readthedocs.io/en/latest/schedulers.html
- params：用于实例化调度器的参数字典。参数名称应与调度程序构造函数签名匹配。

scheduler 示例：

```
 "scheduler": {
      "type": "WarmupLR",
      "params": {
          "warmup_min_lr": 0,
          "warmup_max_lr": 0.001,
          "warmup_num_steps": 1000
      }
  }
```

## 通讯选项


### communication_data_type


### prescale_gradients


### gradient_predivide_factor


### sparse_gradients


## FP16 训练选项

- 注意：此模式不能与下述 amp 模式结合使用。






```
"fp16": {
    "enabled": true,
    "auto_cast": false,
    "loss_scale": 0,
    "initial_scale_power": 16,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "consecutive_hysteresis": false,
    "min_loss_scale": 1
}
```





## BFLOAT16 训练选项

- 注意：此模式不能与下述amp模式结合使用。
- 注意：该模式不能与上述fp16模式结合使用。



```
"bf16": {
   "enabled": true
 }
```



## 自动混合精度 (AMP) 训练选项

注意：该模式不能与上述fp16模式结合使用。 此外，该模式目前与 ZeRO 不兼容。

```
"amp": {
    "enabled": true,
    ...
    "opt_level": "O1",
    ...
}
```


## 梯度裁剪(Gradient Clipping)



## 针对 FP16 训练的 ZeRO 优化


## 参数卸载（Parameter offloading）


启用和配置 ZeRO 优化，将参数卸载到 CPU/NVMe。 仅适用于 ZeRO 阶段 3。

- 注意，如果"device"的值未指定或不支持，则会触发断言。

```
 "offload_param": {
    "device": "[cpu|nvme]",
    "nvme_path": "/local_nvme",
    "pin_memory": [true|false],
    "buffer_count": 5,
    "buffer_size": 1e8,
    "max_in_cpu": 1e9
  }
```

## 优化器卸载

启用和配置 ZeRO 优化，将优化器计算卸载到 CPU 并将优化器状态卸载到 CPU/NVMe。 

CPU 卸载适用于 ZeRO 阶段 1、2、3。NVMe 卸载仅适用于 ZeRO 阶段 3。

- 注意，如果"device"的值未指定或不支持，则会触发断言。


```
 "offload_optimizer": {
    "device": "[cpu|nvme]",
    "nvme_path": "/local_nvme",
    "pin_memory": [true|false],
    "buffer_count": 4,
    "fast_init": false
  }
```

## Activation Checkpointing

```
"activation_checkpointing": {
    "partition_activations": false,
    "cpu_checkpointing": false,
    "contiguous_memory_optimization": false,
    "number_checkpoints": null,
    "synchronize_checkpoint_boundary": false,
    "profile": false
    }
```


## 稀疏注意力（Sparse Attention）

```
"sparse_attention": {
 "mode": "fixed",
 "block": 16,
 "different_layout_per_head": true,
 "num_local_blocks": 4,
 "num_global_blocks": 1,
 "attention": "bidirectional",
 "horizontal_global_attention": false,
 "num_different_global_patterns": 4,
 "num_random_blocks": 0,
 "local_window_blocks": [4],
 "global_block_indices": [0],
 "global_block_end_indices": None,
 "num_sliding_window_blocks": 3
}
```


## Flops 分析器（Flops Profiler）

- detailed：是否打印详细的模型配置。
- output_file：输出文件的路径。 如果没有，Profiler 将打印到标准输出。


```
{
  "flops_profiler": {
    "enabled": false,
    "profile_step": 1,
    "module_depth": -1,
    "top_modules": 1,
    "detailed": true,
    "output_file": null,
    }
}
```



## 压缩（Compression）

### Layer Reduction

### 权重量化（Weight Quantization）


### 激活量化（Activation Quantization）

### 稀疏剪枝（Sparse Pruning）

### 头剪枝(Head Pruning)



### 通道剪枝（Channel Pruning）


## Checkpoint 选项

```
"checkpoint": {
    "tag_validation"="Warn",
    "load_universal"=false,
    "use_node_local_storage"=false,
    "parallel_write":{
        "pipeline_stage": false
    }
}
```



## 数据类型选项

```
"data_types": {
    "grad_accum_dtype"=["fp32"|"fp16"|"bf16"]
    }
}
```



## Data Efficiency







