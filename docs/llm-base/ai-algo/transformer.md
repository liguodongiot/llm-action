
原理：
- [x] Transformer模型详解（图解最完整版）：https://zhuanlan.zhihu.com/p/338817680
- [x] OpenAI ChatGPT（一）：十分钟读懂 Transformer：https://zhuanlan.zhihu.com/p/600773858

源码：
- OpenAI ChatGPT（一）：Tensorflow实现Transformer：https://zhuanlan.zhihu.com/p/603243890
- GPT （一）transformer原理和代码详解：https://zhuanlan.zhihu.com/p/632880248
- Transformer源码详解（Pytorch版本）：https://zhuanlan.zhihu.com/p/398039366








Transformer是编码器－解码器架构的一个实践，尽管在实际情况中编码器或解码器可以单独使用。

在Transformer中，多头自注意力用于表示输入序列和输出序列，不过解码器必须通过掩蔽机制来保留自回归属性。

Transformer中的残差连接和层规范化是训练非常深度模型的重要工具。

Transformer模型中基于位置的前馈网络使用同一个多层感知机，作用是对所有序列位置的表示进行转换。




---
Add & Norm 层，Add 表示残差连接 (Residual Connection) 用于防止网络退化，Norm 表示 Layer Normalization，用于对每一层的激活值进行归一化。




## Transformer的输入表示

Transformer 中除了单词的Embedding，还需要使用位置Embedding 表示单词出现在句子中的位置。因为 Transformer不采用RNN结构，而是使用全局信息，不能利用单词的顺序信息，而这部分信息对于NLP来说非常重要。所以Transformer中使用位置Embedding保存单词在序列中的相对或绝对位置。


位置Embedding用 PE 表示，  PE 的维度与单词Embedding相同。 PE 可以通过训练得到，也可以使用某种公式计算得到。在Transformer中采用了后者。








