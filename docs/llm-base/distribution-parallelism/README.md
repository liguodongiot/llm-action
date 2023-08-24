



- One weird trick for parallelizing convolutional neural networks
  - 不同的层适合用不同的并行方式，具体的，卷积层数据比参数大，适合数据并行，全连接层参数比数据大，适合模型并行。
- Exploring Hidden Dimensions in Parallelizing Convolutional Neural Networks
  - 这篇文章在抽象上更进一步，发现数据并行，模型并行都只是张量切分方式的不同罢了，有的是切数据，有的是切模型，而且对于多维张量，在不同的维度上切分，效果也不同，譬如在sample, channel, width, length等维度都可以切分。
  - 其次，不同的切分方式，都是一种构型（configuration)，不同的构型会导致不同的效果，所以寻找最优的并行方式，其实就是在构型空间里面搜索最优的构型而已，问题形式化成一个搜索问题。
  - 最后，引入了代价模型来衡量每个构型的优劣，并提出了一系列对搜索空间剪枝的策略，并实现了原型系统。

- BEYOND DATA AND MODEL PARALLELISM FOR DEEP NEURAL NETWORKS（FlexFlow）
  - 主要是提出了execution simulator来完善cost model。
  
- Supporting Very Large Models using Automatic Dataflow Graph Partitioning（Tofu）
  - tofu 提出了一套DSL，方便开发者描述张量的划分策略，使用了类似poly的integer interval analysis来描述并行策略，同样，并行策略的搜索算法上也做了很多很有特色的工作。
  - Tofu与所有其它工作的不同之处在于，它的关注点是operator的划分，其它工作的关注点是tensor的划分，二者当然是等价的。
  - 不过，我认为关注点放在tensor的划分上更好一些，这不需要用户修改operator的实现，Tofu需要在DSL里描述operator的实现方式。
  - 
- Mesh-TensorFlow: Deep Learning for Supercomputers
  - Mesh-TensorFlow的作者和GShard的作者几乎是重叠的，Mesh-TensorFlow甚至可以被看作GShard的前身。
  - Mesh-TensorFlow的核心理念也是beyond batch splitting，数据并行是batch splitting，模型并行是张量其它维度的切分。这篇文章把集群的加速卡抽象成mesh结构，提出了一种把张量切分并映射到这个mesh结构的办法。



- Unity: Accelerating DNN Training Through Joint Opt of Algebraic Transform and Parallelization：https://zhuanlan.zhihu.com/p/560247608
  - Unity 在 FlexFlow、TASO 和 MetaFlow 的基础上，提出在并行计算图（PCG）中代数变换和并行化的统一表示（OP，Operator）和共优化（图替代，Substitution）方法，可以同时考虑分布式训练中的计算、并行和通信过程。对于共优化，Unity 使用一个多级搜索算法来高效搜索性能最好的图替代组合以及相应的硬件放置策略。




- https://github.com/DicardoX/Individual_Paper_Notes
- https://jeongseob.github.io/readings_mlsys.html
- https://paperswithcode.com/methods/category/distributed-methods





