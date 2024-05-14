

## 定义模型

PositionalEncoding 模块注入一些有关序列中标记的相对或绝对位置的信息。 

位置编码与嵌入具有相同的维度，因此可以将两者相加。 在这里，我们使用不同频率的正弦和余弦函数。






在本教程中，我们将在两个 GPU 上拆分 Transformer 模型，并使用流水线并行来训练模型。 

除此之外，我们还使用分布式数据并行来训练该流水线的两个副本。 

我们有一个进程驱动跨 GPU 0 和 1 的流水线，另一个进程驱动跨 GPU 2 和 3 的流水线。

然后，这两个进程都使用分布式数据并行来训练两个副本。 该模型与使用 nn.Transformer 和 TorchText 进行序列到序列建模教程中使用的模型完全相同，但分为两个阶段。 

参数数量最多的是 nn.TransformerEncoder 层。 nn.TransformerEncoder 本身由 nlayers 个 nn.TransformerEncoderLayer 组成。 因此，我们的重点是 nn.TransformerEncoder，并拆分模型，使 nn.TransformerEncoderLayer 的一半位于一个 GPU 上，另一半位于另一个 GPU 上。 

为此，我们将编码器和解码器部分提取到单独的模块中，然后构建一个代表原始 Transformer 模块的 nn.Sequential。




