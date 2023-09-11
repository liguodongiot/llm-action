

- https://www.usenix.org/conference/osdi22/presentation/zheng-lianmin



- alpa: https://www.zhihu.com/question/414549247/answer/3044074132


Alpa 则是先通过动态规划来决定模型怎么切分成 stage，每个 stage 能分到哪些卡。然后在每个 stage 内部，再通过

整数线性规划

的方式来决定每个 op 是如何切分到这个 stage 的多个卡上的。这是一个自动优化的过程。


旧有的系统往往 focus 在 inter-op，intra-op 和自动并行策略搜索的一个或者两个点，而 alpa 兼顾了所有。比如 GShard 提出了 intra-op 的方式，GPipe 提出 inter-op 的方式，而 Megatron-LM v2 通过结合 inter 和 intra op 的方式，通过人工指定的并行策略来支持分布式训练 GPT 模型。微软提出的 ZeRO 模型（deepspeed）试图通过自动的策略，通过多个层级步骤，来优化数据并行中的显存使用。Alpa 我们前面介绍了，首先做 inter-op 的自动切分，然后做 intra-op 的层级调度方式，来达到兼顾所有的优化策略。



Alpa 高度依赖 JAX，它魔改了 XLA （JAX 底层通过 XLA 执行）中的 GSPMD，拿到 XLA 的计算图后，自动对 op 进行切分，生成对应的程序，在每个 worker 上执行。



Alpa 确实是大语言模型并行训练的 SOTA 工作，在理论上突破它还是有相当难度。





- 用ILP和DP自动探索DL分布式策略——Alpa:https://zhuanlan.zhihu.com/p/487588274

Alpa，可以在用户无任何干预的情况下，在可接受的时间内，自动对DL模型做分布式策略寻优。它需要两个输入：1. computation graph 2: device cluster


Alpa将分布式策略归结为两类：

Intra-Operator Parallelism：将Tensor按某些维度切裂，放到不同Device上计算的并行方式
Inter-Operator Parallelism：与Intra-Operator相对的并行方式
Data Parallelism和Operator Partitioning属于1，Pipeline Parallelism属于2。简单回顾一下这两类并行方式的特点：Intra-op Parallelism可充分利用带宽，切分带来的通信基本属于高效的集合通信。而Inter-op Parallelism若切点寻找的合适，则通信较小，但同步版本的策略无可避免的会引来Bubble。所以，可以利用cluster的非对称特性，将Intra-op Parallelism映射到高带宽互联的devices上；将Inter-op Parallelism映射到低带宽互联的devices上。如此组合，就能释放更大的算力。Alpa会自动探索这些策略及组合情况





之所以要做这两种区分，目的是为了在不同的level上做策略搜索，然后将二者组合起来，生成一统Data、Operator和Pipeline并行方式的执行计划。总结起来就是以下两个点：

将并行度视为两级：intra-operator和inter-operator，构建分级的解空间
在两级空间中分别探索最优解，然后提供高效的Runtime将二者编排起来



Alpa的Workflow

Alpa工作在DL编译层（基于XLA，在HLO上进行策略探索），所以Paper中也称之为——自动生成分布式策略的DL编译器。因此，它使用优化Pass对HLO IR做策略探索，改图，以及其他优化。在Alpa的workflow中，有三个pass起了重要作用:

Inter-op pass：不仅要将计算图切分成多个stage，还要对device cluster进行相应的切分，并组织成mesh分配给stage，所以产出stage-mesh pairs。这一过程使用动态规划（Dynamic Programming）完成，优化目标是追求端到端最小latency

Intra-op pass：对单个stage-mesh pair做intra-op的切分探索，并把的结果将reports给上一级的inter-op pass。这一过程使用整数线性规划（Integer Linear Programming）完成，优化目标是最小化execution cost

Runtime Orchestration pass：做Runtime缝合的事情。比如stage调度，通信优化等。显然这个pass和策略搜索没什么关系
所以在宏观上做DP，微观上ILP，Inter-op和Intra-op两个pass不断迭代最终获得最佳方案。


并行策略模块

Intra-Operator并行策略探索
Alpa使用ILP对单个stage-mesh pair求解。这里有两个假设：a. device mesh中所有的设备具有完全相同的计算能力；b. mesh一定是二维的



其他方面

还有两项值得注意，一个是关于Cost的获取，另一个是缩图剪枝。和很多文献中的cost model相同，Alpa也采用事先预估cost的方式，并假设所有的op的computation cost均为0（这和Tofu相同，即只考虑通信开销）；在搜索空间方面，能剪枝则剪枝，例如将一些operator和它的operand融合，从而减少计算图的节点规模，是有必要的。


Inter-Operator并行策略搜索


上面已经提到，ILP针对单个stage-mesh pair做优化，那么如何将整个computation graph和device mesh cluster切成一个个stage-mesh pair呢？这就是Inter-Operator pass的工作了。该pass使用DP解决划分问题，目标是优化整体端到端latency。

不过，这里还是要做一个假设：假设computation graph的拓扑按照user define的顺序排列，这就是要把排序限定在固定的linear顺序。否则，动态规划算法不是很好建模。








- 【论文赏读】Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning: https://zhuanlan.zhihu.com/p/571836701




Part 1. Introduction

已有并行工作的局限：要么被限制在单个并行方法 (PipeDream)，要么依赖于对模型和集群规格的强假设 (DAPPLE, Tofu)。

自动混合并行的搜索空间较复杂，多并行策略的实现不够灵活。例如，若数据并行和 OP 并行结合，则添加一个 worker 需要额外分配多个设备；同时，最优的流水线策略依赖于每个 stage 数据和 OP 并行的决策，以及设备分配。因此，先前工作大部分局限于将数据并行和至多一类并行方法结合；
Key observation：可以将不同并行策略组织为一个多级空间，并将这些并行策略映射到计算集群中的多级结构。不同并行策略对通信带宽要求不同，这与集群中不同位置设备间的带宽不同相一致；
多级并行策略的好处：1) intra-op 并行的硬件利用率更高，但会在每 iter 带来 OP 切片间的更多通信，以及划分 OP 后的合并；2) inter-op 并行仅在相邻 stages 间通信，若适当划分则轻量，但会导致设备的 idle 时间。Alpa 将 intra-op 映射到高带宽设备，将 inter-op 映射到距离较远的低带宽设备，并分级近似最优优化，全局非最优但性能较好；



Part 2. Intra-op Parallelism



Intra-op pass 中，模型图被表示为 XLA 的 HLO 格式，将常见的 DL OPs 总结为 80 个不到的原语 OPs，所以可以人工枚举每个原语 OPs 可能的并行策略。


整数线性规划 ILP Formulation：
cost 模型：类似 Tofu，由于 Alpa 对大 OP 进行均匀切分，所有的并行方法有相同的算数复杂性；同时，小 OP 设备间存在的部分重复计算开销可忽略。因此，Alpa 仅关注通信开销和 reshard 开销，而非计算开销。在 intra-op pass 中，不对通信开销和 reshard 开销进行 profile，而是用通信量 / 带宽来评估；
目标函数：最小化图内所有节点的通信开销和所有边的 reshard 开销之和。


图简化：Alpa 将轻量 OP 合并，通过 BFS 确定并合并到深度最深的 OP。原因是推迟到最后的 depth 才计算，防止跨分支 merge 时出现依赖关系紊乱（例如分支 A 上已经执行到了，但分支 B 还没，此时执行 merged OP 显然不合理）。此外，和 ZeRO 类似，将 all-reduce 拆分为 reduce-scatter 和 all-gather，在 mesh 内共享优化器状态、梯度和参数，以节约内存。




Part 3. Inter-op Parallelism



不进行 OP 划分或 replicate，而是将不同 OPs 组织为 stages，分配到由 cluster mesh 切分的不同 device meshes 上执行，映射为 stage-mesh pair。Alpa 使用同步 1F1B 调度作为 pipeline 策略，该方法保留了同步一致性，且相较于 GPipe，保持 pipeline 延迟相同的同时，峰值内存开销更低。
Pipeline stages 间进行设备间的点对点通信，所需通信量远小于 intra-op parallelism；由于 stages 间的数据依赖，会导致部分设备在 fp/bp 过程中存在 idle time；
先前工作假设每个 stage 的设备是预先分配好的，且所有 stages 有固定的 intra-op plan；
Inter-op pass 基于模型图的拓扑顺序（简单实现为 model IR 中 users 如何定义每个 OP 的顺序）进行模型图的线性化。




Part 4. Runtime Orchestration


Workflow：Alpa 依次设计了 inter-op pass、intra-op pass 和 runtime orchestration 三类 pass。
给定 Jax IR 的模型描述和集群配置，inter-op pass 将 IR 切分为多个 stages，将设备集群切分为多个 device meshes，并通过动态规划 (DP) 将 stages 分配到 meshes，并对每个 stage-mesh pair 激活 intra-op pass。
被激活后，intra-op pass 通过整数线性规划 (ILP) 来最小化 stage 执行开销，优化 stage-mesh pair 的 intra-op plan，并将 plan 和开销反馈给 inter-op pass。
随后，inter-op pass 基于 intra-op plan 编译并执行 stage-mesh plan，获取精确的 stage latency，运行 stage 所需的内存，以及存储中间激励所需的内存，并基于 cost 和内存情况通过 DP 最小化 pipeline 的 end-to-end latency，以获取 stages 和 meshes 的最优划分方案。


在 stage level，Alpa 基于 XLA 和 GSPMD 进行编译，为每个 stage-mesh pair 生成并行可执行文件，并在需要的时候插入通信原语来处理 mesh 内通信；在 intra-op level，Alpa 实现了一个并行编排 pass，来解决 cross-mesh 的 stages 间通信，并为 inter-op 并行执行生成静态指令。
Cross-mesh Resharding：现有的人工 pipeline 系统要求 stages 有相同的 DP 和 MP 并行度，并将 stages 间通信简单实现为相同 meshes 间的 P2P send/recv。 Alpa 中，包含两个相邻 stages 的设备 meshes 可能有不同的 mesh shapes，且 stages 间通信的 tensor 可能有不同的 sharding specs，将这类通信模式定义为 cross-mesh resharding，一个多对多的多播问题。给定 sender 和 receiver mesh 上 tensors 的 sharding specs，Alpa 生成一个通信方案，从两步迭代来解决 cross-mesh sharding：1) Alpa 计算 src 和 dst mesh 间 tiles (tensor partitions) 的对应关系，生成 src 和 dst 设备间的 P2P send/recv 原语；2) 识别 dst tensor 的 sharding spec 中是否包含 replication，若包含则 tensor 仅需要向 dst mesh 传一次，并通过 all-gather 在 mesh 内的 devices 上以更高的带宽交换，该方法叫 local all-gather。由于 stages 间的通信量较小，Alpa 未考虑 cross-mesh resharding 的优化。
生成 pipeline 执行指令：最后一步，Alpa 生成静态执行指令来 launch 训练，使用一个 MPMD-style 的 runtime 来编排 inter-op 并行执行，为每个设备 mesh 单独生成静态执行指令集，包括 stage 内 tensors 的内存分配和去分配，stages 间按 cross-mesh resharding plan 的 tensors 通信，同步和计算等。根据 user 选择的 pipeline schedule，Alpa 使用一个 driver process 来在执行前提前生成指令并分发给各 worker，以避免 runtime 时 driver-worker 的协同开销。




- Alpa/Parax @OSDI 2022 : https://zhuanlan.zhihu.com/p/521211578


Alpa认为，不同的并行技术是有不同的带宽要求的，这也和分布式系统的结构相符合，因此，在不同的系统层次使用不同的并行技术，是Alpa的重要observation。基于这一observation，Alpa提出iter-operator和intra-operator并行。算子内并行主要考虑数据和模型划分，算子间并行主要考虑流水线并行。



算子间、算子内并行
Alpa提出的算子间、算子内并行划分方法，通过“是否切分了tensor的维度”来区分不同的并行。

算子内并行：切分了tensor维度的并行方式，包括数据并行和算子并行
算子间并行：不切分tensor，只是把子图进行不同的摆放分布，包括流水线并行。



Alpa设计

Alpa以计算图为输入，输出并行方案。为了决策这样的并行方案，Alpa需要解决两个层面的问题，一个是怎么分子图，子图间是流水线并行的；另一个是分好的子图怎么并行到不同的设备上。这两个层次是相互依赖的，而且每个子图分配多少设备也是需要决策的，这样整个决策问题非常复杂。Alpa的解决方案是在不同层次上用不同的优化算法（整数线性规划和动归）。


---

2. 主要技术

2.1. 算子内并行


2.2 算子间并行
这一步需要解决两个问题：

子图怎么分
每个子图给多少设备
一个完整的计算图，可以看做一个算子列表（拓扑排序后），相邻的一些算子可构成一个stage，那么子图划分可以看成在这个算子列表里划分stage的问题（文章没提，这种划分总是合法的，因为总是保持stage的凸性）。划分子图后，可以给子图分配一个设备阵列，并根据前面介绍的算子内并行优化方法找到最好的并行方案。那么算子间并行的优化问题就是看做枚举不同的子图划分和设备划分，找到整体代价最小的组合划分方案：



Alpa用动归方式解决这一问题



这个动归的复杂度很高，为了能在实际中应用，需要一些技巧：

提前剪枝
提前算子融合减少算子数目

2.3 并行处理

Alpa用XLA和GSPMD进行算子内并行方案的编译。对于算子间并行的编译，Alpa有一些自己的特殊处理。

cross-mesh resharding：相邻的stage可以用不同的设备阵列配置。
generating pipeline execution instruction：这一步需要MIMD的runtime



3. 实验


Alpa的代码是开源的（ https://github.com/alpa-projects/alpa ），用JAX做前端，用XLA做后端。
用Ray actor实现设备阵列管理，用NCCL做通信。










