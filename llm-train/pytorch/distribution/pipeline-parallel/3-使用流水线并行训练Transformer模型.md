




## 定义模型

在本教程中，我们将在两个 GPU 上拆分 Transformer 模型，并使用流水线并行来训练模型。 

该模型与使用 nn.Transformer 和 TorchText 进行序列到序列建模教程中使用的模型完全相同，但分为两个阶段。 

参数数量最多的是 nn.TransformerEncoder 层。 

nn.TransformerEncoder 本身由 nlayers 个 nn.TransformerEncoderLayer 组成。 因此，我们的重点是 nn.TransformerEncoder，并拆分模型，使 nn.TransformerEncoderLayer 的一半位于一个 GPU 上，另一半位于另一个 GPU 上。 为此，我们将编码器和解码器部分提取到单独的模块中，然后构建一个代表原始 Transformer 模块的 nn.Sequential。







## 模型流水线并行初始化

为了演示使用流水线并行训练大型 Transformer 模型，我们适当扩展了 Transformer 层。 我们使用 4096 的嵌入维度、4096 的隐藏大小、16 个注意力头和 12 个 Transformer 层 (nn.TransformerEncoderLayer)。 这将创建一个具有约 14 亿个参数的模型。

我们需要初始化 RPC 框架，因为 Pipe 通过 RRef 依赖于 RPC 框架，这允许将来扩展到跨主机流水线。 

由于我们使用单个进程来驱动多个 GPU，因此我们只需要使用单个工作线程来初始化 RPC 框架。

然后使用一个 GPU 上的 8 个Transformer 层和另一个 GPU 上的 8 个Transformer 层来初始化流水线。


注意：

出于效率目的，我们确保传递给 Pipe 的 nn.Sequential 仅包含两个元素（对应于两个 GPU），这允许 Pipe 仅处理两个分区并避免任何跨分区开销。








