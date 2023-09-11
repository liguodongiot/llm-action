
- LLM推理优化技术综述：KVCache、PageAttention、FlashAttention、MQA、GQA：https://zhuanlan.zhihu.com/p/655325832


## KVCache


## PageAttention



PageAttention是目前kv cache优化的重要技术手段，目前最炙手可热的大模型推理加速项目VLLM的核心就是PageAttention技术。

在缓存中，这些 KV cache 都很大，并且大小是动态变化的，难以预测。已有的系统中，由于显存碎片和过度预留，浪费了60%-80%的显存。

PageAttention提供了一种技术手段解决显存碎片化的问题，从而可以减少显存占用，提高KV cache可使用的显存空间，提升推理性能。



---


首先，PageAttention命名的灵感来自OS系统中虚拟内存和分页的思想。可以实现在不连续的空间存储连续的kv键值。

另外，因为所有键值都是分布存储的，需要通过分页管理彼此的关系。序列的连续逻辑块通过 block table 映射到非连续物理块。

另外，同一个prompt生成多个输出序列，可以共享计算过程中的attention键值，实现copy-on-write机制，即只有需要修改的时候才会复制，从而大大降低显存占用。





## FlashAttention

Flash attention推理加速技术是利用GPU硬件非均匀的存储器层次结构实现内存节省和推理加速，它的论文标题是“FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness”。意思是通过合理的应用GPU显存实现IO的优化，从而提升资源利用率，提高性能。


首先我们要了解一个硬件机制，计算速度越快的硬件往往越昂贵且体积越小，Flash attention的核心原理是尽可能地合理应用SRAM内存计算资源。

A100 GPU有40-80GB的高带宽内存(HBM)，带宽为1.5-2.0 TB/s，而每108个流处理器有192KB的SRAM，带宽估计在19TB/s左右。

也就是说，存在一种优化方案是利用SRAM远快于HBM的性能优势，将密集计算尽放在SRAM，减少与HBM的反复通信，实现整体的IO效率最大化。比如可以将矩阵计算过程，softmax函数尽可能在SRAM中处理并保留中间结果，全部计算完成后再写回HBM，这样就可以减少HBM的写入写出频次，从而提升整体的计算性能。

如何有效分割矩阵的计算过程，涉及到flash attention的核心计算逻辑Tiling算法。



### v1



### v2







## MQA/GQA


MQA，全称 Multi Query Attention, 而 GQA 则是前段时间 Google 提出的 MQA 变种，全称 Group-Query Attention。MHA（Multi-head Attention）是标准的多头注意力机制，h个Query、Key 和 Value 矩阵。MQA 让所有的头之间共享同一份 Key 和 Value 矩阵，每个头只单独保留了一份 Query 参数，从而大大减少 Key 和 Value 矩阵的参数量。GQA将查询头分成N组，每个组共享一个Key 和 Value 矩阵。


GQA以及MQA都可以实现一定程度的Key value的共享，从而可以使模型体积减小，GQA是MQA和MHA的折中方案。这两种技术的加速原理是（1）减少了数据的读取（2）减少了推理过程中的KV Cache。需要注意的是GQA和MQA需要在模型训练的时候开启，按照相应的模式生成模型。










