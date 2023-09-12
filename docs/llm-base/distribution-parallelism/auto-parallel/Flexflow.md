
- 原paper

## Beyond Data and Model Parallelism for Deep Neural Networks

### 概要

训练深度神经网络 (DNN) 的计算要求已经增长到现在并行训练已成为标准做法。 现有的深度学习系统通常使用数据或模型并行性，但不幸的是，这些策略通常会导致并行化性能不佳。

在本文中，我们定义了一个更全面的 DNN 并行化策略搜索空间（称为 SOAP），其中包括在样本、操作、属性和参数维度中并行化 DNN 的策略。 我们还提出了 FlexFlow，这是一种深度学习框架，它使用 SOAP 空间的引导随机搜索来寻找特定并行机的快速并行化策略。 为了加速这种搜索，FlexFlow 引入了一种新颖的执行模拟器，它可以准确预测并行化策略的性能，并且比必须执行每个策略的先前方法快三个数量级。

我们在两个 GPU 集群上使用六个真实世界的 DNN 基准测试来评估 FlexFlow，结果表明，即使包括搜索时间，FlexFlow 也可以将训练吞吐量提高至最先进方法的 3.8 倍，并且还提高了可扩展性。



### 6 Execution Optimizer

本节介绍执行优化器，它以运算符图和设备拓扑作为输入，并自动找到有效的并行化策略。

FlexFlow 使用模拟器作为预言机，将并行优化问题转化为成本最小化问题，即最小化预测执行时间。 这种方法的主要优点是，它避免了显式地编码相互依赖的优化之间的权衡（例如，减少数据传输与平衡工作负载分布），而只是专注于最小化应用程序的整体执行时间。

通过从最小完工时间轻松减少来找到最佳并行化策略是 NP 困难的[29]。 此外，如第 4 节所述，可能的策略数量与运算符图中的操作数量成指数关系，这使得穷举搜索空间变得困难。

为了找到低成本策略，FlexFlow 使用成本最小化搜索程序来启发式探索空间并返回发现的最佳策略。




- 更多维度的深度神经网络并行策略: https://diandiangu.github.io/2020/07/20/FlexFlow/



这篇文章主要的点有如下几个：

相比于data-parallel和model-parallel，提出了更多维度的split方案。SOAP（sample，operator，atrribute，param）这四个维度的split方案。
在四个维度之上，提出了一种在候选空间搜索的方案
提出了一个更加轻量的simulator，可以更快速的对proposed split strategy做evaluate。相比直接执行的方案提升了3个数量级。
实现了总体的框架FlexFlow





FlexFlow的总体框架如下：

operator graph是计算图的描述。包括op作为node，tensor作为edge。
device topology描述实际设备的topo关系，device作为node，connection作为edge
Execution Optimizer是FlexFlow的核心部件，用于搜索最优的split方案，下面是一个runtime，用于执行split方案。



SOAP

描述了与mindspore基本一致的模型切分config description。FlexFlow描述了SOAP维度的切分，是针对op的output tensor来切分的。选择了output tensor的多个维度：

Sample表示input的batch维
Attribute表示tensor的属性维，例如height/width
Parameter表示tensor的param维，例如in-channel/out-channel
Operator表示op之间的切分维度。
从这里看，虽然把tensor分成了多个维度，实际上都是属于tensor本身的维度。这里跟mindspore是一样的。




Execution Simulator

simulator是FlexFLow比较核心的部分，负责对proposed strategy做evaluate。得到candidate的性能数据。

这里为了提供evaluate的速度，没有使用直接执行的方式，而是用模拟执行。

还是正常去构建执行timelines，但是需要在device上执行时，直接从上一次执行相同input-size的数据中取得执行时间。这样降低了总体的执行时间。这里是假设op针对相同input-size的执行时间基本不变，而且跟input-data无关。在大多数模型中，这个假设都是成立的。






search algorithm



对所有解空间的遍历是一个NP-hard问题。FlexFlow采用了一个最小化cost的search算法来启发式探索解空间，并返回找到的最优策略。

采用了MCMC的随机方法search strategy space。MCMC是马尔可夫链蒙特卡洛算法。


蒙特卡洛方法是通过随机生成变量Xi的值，然后用Xi的值做模拟计算。从而得到目标问题的解。

如果随机变量Xi的概率分布比较复杂，不能用简单的均匀分布转换得到，就需要使用马尔可夫链蒙特卡洛方法来生成Xi的序列。

回到search space的问题，采用MCMC随机采样的方法生成随机的Xi，即可作为候选strategy。






- 读论文《FlexFlow-Beyond Data and Model Parallelism for Deep Neural Networks》: https://zhuanlan.zhihu.com/p/464355830


这篇文章提出了一个复杂的深度神经网络的并行策略：SOAP。Sample，Operation，Attribute，Parameter。提出了FlexFlow：一个在SOAP维度为特定的并行机器随机搜索快速并行策略的深度学习框架。为了加速这个搜索的过程，FlexFlow用了一个创新性的、可以准确预测一个并行策略的表现、比原有方法更快速的执行模拟器。实验结果表明FlexFlow可以很大程度上增大训练的吞吐率。本文由斯坦福大学团队发表于MLSys 2019。



这篇文章提出FlexFlow，一个可以自动在更大的范围内找出快速并行策略的深度学习框架。为了形式化这一问题，我们首先定义了SOAP。Operation维度描述了一个DNN中不同的operation是如何并行的。另外，对于一个单独的DNN operation来说，sample和parameter维度描述训练样例和模型参数如何在不同设备之间分布。最终，attribute维度定义一个sample中不同的attribute是如何划分的。已有的系统都是在SOAP的子集中划分的。
在SOAP这个更大的范围内搜索的一个主要的挑战是快速评估候选的并行方案已找到一个高效的方案。已有的工作依赖于在硬件上执行一轮训练来评估不同方案的执行时间。在SOAP的范围内，这样的方法代价太高。



为了解决这样的问题，FlexFlow提出了一个创新性的执行模拟器，可以准确预测并行策略的表现，比profile真实的运行快了三个数量级。设计模拟器的挑战在于如何准确估计不同DNN op的执行时间（非线性，取决于硬件）。模拟器依赖于两个事实：（1）很多DNN模型只用少数几个不同的op（2）op的执行时间通常差异不大，很大程度上取决于输入数据。FlexFlow的模拟器对于每种数据大小，用一个op的计算时间来衡量同种类op的计算时间。然后，这些估算被用于预测各种各样的并行策略。另外，模拟器使用了一种delta simulate算法，这种算法基于对之前的模拟的更新对新的策略作出模拟。何以有的方法比，这样的方法有两个优势：更快、所需资源更少。
模拟器的预测准确率很高。


FlexFlow的execution optimizer使用一种马尔可夫蒙特卡洛搜索算法探索SOAP的搜索空间，并给予对之前的候选策略的模拟表现选出候选策略。搜索过程结束后，optimizer返回最佳的策略。

---

程序接口

与多数深度学习框架不同，FlexFlow用设备拓扑结构描述所有可用的应尽设备和他们之间的关联。拓扑结构中的“边”有带宽和延迟的标签。
FlexFlow可以自动为一个计算图和一个设备拓扑结构找到合适的并行策略。主要有两大优势：提供了易于编程的接口；可移植性（为不同的硬件自动选择搞笑的策略）。


FlexFlow架构


Execution optimizer为计算图和设备拓扑图选择高效的并行策略。


