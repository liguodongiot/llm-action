




- Mesh-Tensorflow: 广义分布式: https://zhuanlan.zhihu.com/p/342223356


在深度学习中，由于数据量和计算量的浩大，往往会使用到分布式计算。而最常用的分布式模式是SPMD(Single-Program-Multiple-Data)，即数据并行，这种模式相当于在数据的batch维去做拆分；然后，进行并行。

Mesh-Tensorflow对这种模式做了泛化，即除了batch维外的其他维度也可做并行。






---


Mesh-Tensorflow的灵感来自于目前广泛使用的数据并行, 数据并行可以看做是把 tensors 和 operations 在 batch 这个维度上进行分割。 Mesh-Tensorflow则顺势把这个点子推广到所有维度。


Mesh-Tensorflow 看定义了一套DSL语法，用于描述模型的维度和布局，你用它重写你的整个Model后，它自动帮你把模型和数据分割到多个TPU上。

Mesh-Tensorflow看起来很复杂和精巧，比 recomputation, pipeline parallelism 等技巧要更复杂更自动化，那它是否就能解决问题呢？

我觉得它侵入性比普通的库（例如GPipe）更强，你需要用Mesh-Tensorflow的语法重写你的整个模型，仔细思考维度，说实话，这个精神负担挺重的(想起了C++)。况且，它目前还没有实现并行的卷积操作，因此对于CNN网络并没有卵用，暂时只适合 Language Model 这个领域.

