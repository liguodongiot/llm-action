



KVCache 量化的值计算还是使用 fp16 计算是比较常见的方式。


KVCache 量化与离线的参数一次性量化好可以不停使用不同，KVCache 的数据是随时生成的，随后马上就进行量化过程的再进行存储和后续使用，虽然也是一次性量化好的，但是这个一次性是在整个推理过程中的，是占有一部分在线资源和延时的。





KVQuant



在处理长序列和较大的批量时，激活内存成为了主要的性能瓶颈，特别是在模型权重已经量化到较低精度的情况下。对于LLaMA-7B模型，当序列长度达到128K时，KV缓存成为了主要的瓶颈。此外，如果模型权重已经被量化，即使在 32K 的序列长度下，KV 缓存也是主要的瓶颈。





在 QAQ 和 KIVI 中，作者也都发现 Key cache 和 Value cache 之间存在很大的区别，Key cache 量化难度远高于 value 。 KVQuant 采用的方法是在 Key 前面的 rope 前就完成量化，实际使用时则反量化之后再做一次 rope ；QAQ 是采用了混合 bit 数和全精度保留 outlier 的量化策略，KIVI 则是用两种不同的量化分组粒度策略完成不同的量化过程。





KVQuant：
 per-channel 的 Key 和 per-token 的 Value 这样的组合




------

KIVI

 KIVI 和 KVQuant 里用的 per-channel quantization for Key cache 是一个蛮重要的 recipe（然后应该也算新鲜？）。帖子里提到的「有很多更小 block 的 quant 方法」的确没错，但这个 per-channel 在乎的是沿着哪个 dimension 组 block，而不是说具体 block size 多大多小。然后因为 per-channel quant 需要跨多个 token，所以怎么在 autoregressive 的背景下 quant 也是一个需要考虑的挑战；我们搞 FP16 residual 一定程度上也是为了方便这个。我感觉这些楼主应该是理解的（毕竟帖子里也提到了异常值的分布作为 per-channel 的 motivation）


--------

IntactKV

（1）IntactKV中pivot token的现象在同期工作Massive Activations in Large Language Models中得到了更加细致的研究，
这篇论文里有详细分析LLM中pivot token上超大outlier产生的来源、作用和重要性；

（2）IntactKV的校准过程还是采用和其他PTQ方法相同的MSE loss，区别是我们没有进行逐层优化，而是直接端到端地优化所有层的MSE loss，因为IntactKV是基于已经量化好的模型做校准，所以内存开销也不会很大，7B模型在单卡H800上只需要训练10min


现在KVQuant的最新版本里也采用了和IntactKV相同的思路保持首token的KV cache无损，另外IntactKV也支持权重量化和激活值量化，与目前的多种LLM主流量化方法兼容，更加详细的介绍可以参考我们的知乎文章 一个小技巧轻松提升量化精度！IntactKV：保持关键词元无损的大语言模型量化方法

-----------

QAQ

对于文中提到的「震荡（？）」一点，我们想表达的意思是：我们使用微分来表达量化前后的误差，直观来看，Key cache得到的式子显著含有更高阶次，因此需要更小心的量化策略，问题定义如Figure 1所示，详细数学推导请见论文。


Key cache 的分布和 value cache 的完全不同，几篇文章中都着重提到了。这两块是和 softmax 函数直接相关的，一个在 softmax 前一个在后，softmax 又是非线性的，带来了区别就很正常。
