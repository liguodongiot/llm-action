



Reducing Activation Recomputation in Large Transformer Models：https://arxiv.org/pdf/2205.05198

**选择性激活重计算**（selective activation recomputation），是一种策略，即只对那些**占用大量内存但重新计算成本不高的Transformer层的部分激活进行存储和重计算**。例如，在自注意力机制中，某些操作（如: $QK^T$矩阵乘法、softmax、softmax dropout和对V的注意力）会产生较大的激活，但每个输入元素所需的浮点运算次数却相对较低。通过选择性地存储这些激活，可以在使用较少内存的同时，以较低的计算开销重新计算未存储的激活。



通过结合使用序列并行性（sequence parallelism）和张量并行性（tensor parallelism），以及选择性激活重计算，论文中的方法能够在减少5倍激活内存需求的同时，将由激活重计算引起的执行时间开销降低90%以上。这使得在大规模参数的语言模型上训练变换器模型变得更加高效。





