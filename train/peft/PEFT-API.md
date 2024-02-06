



- prepare_model_for_int8_training 用途(deprecated, 直接使用prepare_model_for_kbit_training)：

```
将所有非 int8 模块转换为全精度 (fp32) 以确保稳定性

在输入嵌入层添加一个前向钩子（forward hook）来计算输入隐藏状态的梯度

启用梯度检查点以提高内存效率的训练

```


- prepare_model_for_kbit_training 用途：

```
处理量化模型以用于训练。

1-转换 layernorm 为 fp32
2-使输出嵌入层计算梯度
3-转换lm head 为 fp32

```