




- https://github.com/Dao-AILab/flash-attention

- FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
- FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning


Flash Attention的主要目的是加速和节省内存，主要贡献包括：

计算softmax时候不需要全量input数据，可以分段计算；

反向传播的时候，不存储attention matrix (N^2的矩阵)，而是只存储softmax归一化的系数。


