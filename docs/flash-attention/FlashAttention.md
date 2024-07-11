




- https://github.com/Dao-AILab/flash-attention

- FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
- FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning


Flash Attention的主要目的是加速和节省内存，主要贡献包括：

计算softmax时候不需要全量input数据，可以分段计算；

反向传播的时候，不存储attention matrix (N^2的矩阵)，而是只存储softmax归一化的系数。




Online Softmax+Tiling+Recompute



Online Softmax

Online normalizer calculation for softmax（https://arxiv.org/abs/1805.02867）


FlashAttention-2




Flash Decoding

https://crfm.stanford.edu/2023/10/12/flashdecoding.html

 Flash-Decoding 可以显著加快推理过程中的注意力，使长序列的生成速度提高 8 倍。

 主要思想是尽可能快地并行加载Key和Value，然后分别重新缩放并组合结果以维持正确的注意力输出。
















