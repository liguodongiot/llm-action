




## 定义模型

在本教程中，我们将在两个 GPU 上拆分 Transformer 模型，并使用流水线并行来训练模型。 

该模型与使用 nn.Transformer 和 TorchText 进行序列到序列建模教程中使用的模型完全相同，但分为两个阶段。 

参数数量最多的是 nn.TransformerEncoder 层。 

nn.TransformerEncoder 本身由 nlayers 个 nn.TransformerEncoderLayer 组成。 因此，我们的重点是 nn.TransformerEncoder，并拆分模型，使 nn.TransformerEncoderLayer 的一半位于一个 GPU 上，另一半位于另一个 GPU 上。 为此，我们将编码器和解码器部分提取到单独的模块中，然后构建一个代表原始 Transformer 模块的 nn.Sequential。







