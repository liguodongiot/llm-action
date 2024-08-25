
Text embedding就是将文本转成一组固定维度的向量表示。我们所熟知的word embedding是以token为基本单位，而text embedding则是以文本为基本单位的。 

**理想的text embedding应该尽可能保留文本的语义信息，相同语义但不同表述方式的文本可以被映射到同一个位置，而不同语义的文本在向量空间应该保持对应的距离。**


Text embedding能用来做些什么？它可以用在Classification, Clustering, Pair Classification, Reranking, Retrieval, STS, Summarization, Bitext Mining等任务中。如何评价它的好坏，也可以通过在这些任务的表现来评估。


如何从语言模型中获得Text embedding呢？以目前主流的Transformer-Encoder为例，有以下几种不同方式。


a) 将模型最后一层[CLS]位置的向量表征直接作为句向量。
b) 将模型最后一层[CLS]位置的向量表征，再经过MLP层得到的向量。
c) 将模型最后一层所有位置的向量表征，再经过一个Pooling层得到的向量。（大部分情况下采用的是mean pooling，在有些情况下也会使用max pooling等其他方式）  
d) 将模型最后一层所有位置的向量表征，再经过一个Pooling层跟MLP层得到的向量。






