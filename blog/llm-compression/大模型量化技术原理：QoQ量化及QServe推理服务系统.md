近年来，随着Transformer、MOE架构的提出，使得深度学习模型轻松突破上万亿规模参数，从而导致模型变得越来越大，因此，我们需要一些大模型压缩技术来降低模型部署的成本，并提升模型的推理性能。
模型压缩主要分为如下几类：

-   剪枝（Pruning）
-   知识蒸馏（Knowledge Distillation）
-   量化（Quantization）

本系列将针对一些常见大模型量化方案（GPTQ、LLM.int8()、SmoothQuant、AWQ等）进行讲述。

- [大模型量化概述](https://www.zhihu.com/question/627484732/answer/3261671478)
- 量化感知训练：
    - [大模型量化感知训练技术原理：LLM-QAT](https://zhuanlan.zhihu.com/p/647589650)
    - [大模型量化感知微调技术原理：QLoRA]()
- 训练后量化：
    - [大模型量化技术原理：GPTQ、LLM.int8()](https://zhuanlan.zhihu.com/p/680212402)
    - [大模型量化技术原理：SmoothQuant](https://www.zhihu.com/question/576376372/answer/3388402085)
    - [大模型量化技术原理：AWQ、AutoAWQ](https://zhuanlan.zhihu.com/p/681578090)
    - [大模型量化技术原理：SpQR](https://zhuanlan.zhihu.com/p/682871823)
    - [大模型量化技术原理：ZeroQuant系列](https://juejin.cn/post/7338284106797432873)
    - [大模型量化技术原理：FP8](https://www.zhihu.com/question/658712811/answer/3596678896)
    - [大模型量化技术原理：FP6](https://juejin.cn/post/7412893752090853386)
    - [大模型量化技术原理：KIVI、IntactKV、KVQuant](https://zhuanlan.zhihu.com/p/5932153295)
    - [大模型量化技术原理：Atom、QuaRot](https://zhuanlan.zhihu.com/p/6281447174)
    - [大模型量化技术原理：QoQ量化及QServe推理服务系统](https://zhuanlan.zhihu.com/p/8047106486)
    - [大模型量化技术原理：FP4]()
- [大模型量化技术原理：总结]()


之前讲述了W4A4KV4量化方案Atom和QuaRot，本文将讲述来自 MIT HAN Lab 的W4A8KV4量化方案 QoQ 及 QServe 推理服务系统。

> 文章较长，建议先点赞收藏，后续再慢慢观看。另外，我撰写的**大模型相关的博客及配套代码**均整理放置在Github：[llm-action](https://github.com/liguodongiot/llm-action/tree/main)，有需要的朋友自取。



## 背景

目前，业界主要的整数量化算法可以分为三类：8位权重和8位激活（W8A8）、4位权重和16位激活（W4A16）、4位权重和4位激活（W4A4）量化。前两种方法在准确性方面几乎无损。相比之下，W4A4量化导致显著的准确性降低，尽管通过将其计算映射到高吞吐量的 INT4 Tensor Cores 上有望提供更高的吞吐量。

不幸的是，这种预期的性能提升并没有在当前的GPU上得到一致的观察。例如，最先进的W4A4服务系统 Atom 在 A100 GPU上运行Llama2-7B模型时，比TensorRT-LLM中的 W4A16 和 W8A8 性能还要低20-25%。

也就是说，社区尚未找到一种比W4A16和W8A8更优越的精度组合，用于高效的进行LLM推理服务。例如，W4A16 量化在 FP16 Tensor Cores 上执行计算，由于权重在INT4中，因此需要在 GEMM Kernel 中进行权重反量化。

另一方面，对于W4A4量化，为了保持准确性，**必须对权重和激活应用逐组（per-group）量化，在子通道基础上共享 FP16 缩放因子**。例如，最先进的W4A4量化方法 QuaRot 报告中说从逐组（per-group）量化切换到逐通道（per-channel）量化后，导致困惑度退化0.2。这种逐组（per-group）量化设计需要对局部和进行整数到浮点的反量化（因为INT4 Tensor Core 产生 INT32 局部和），它在W4A4 GEMM的顺序主循环中运行在较慢的 CUDA Core 内。在数据中心GPU上，如A100，**一个CUDA Core 操作与 50 个INT4 Tensor core 操作一样昂贵**。因此，减少CUDA Core 上的开销对于实现LLM服务的最佳吞吐量至关重要。

为了应对这一挑战，作者引入了QoQ，一种W4A8KV4量化算法。QoQ由QServe推理库实现。在QoQ算法中，引入了渐进式量化，在 W4A8 GEMM 中具有较低的反量化开销。此外，还开发了SmoothAttention 来有效减轻4位KV量化引起的准确性下降。在QServe系统中，执行计算感知的权重重排序，并利用寄存器级并行来减少反量化延迟，还使融合注意力算子保持在内存受限区域，利用KV4量化带来性能提升。


## 动机

权重和KV缓存量化（例如：W4、KV4）可以减少LLM服务中的内存占用。同时量化权重和激活（例如：W8A8）也可以提高峰值计算吞吐量。为LLM部署选择合适的精度是一项困难的任务。现有的解决方案如上所述可以大概分为三类：W4A16(per-group）、W8A8（per-channel权重量化+per-token激活量化）、W4A4（per-group）。

**QoQ 为什么选择 W4A8KV4 精度进行量化**？

在LLM中Attention和GEMM运算占大头，所以集中分析这2个部分。在LLM中，计算强度主要被batch_size影响，batch_size越大计算强度越大，如下图所示。

![](https://files.mdnice.com/user/18421/c0d2dfb3-1dec-4cde-9f04-c537ff19c3e0.png)

对于一个 m × n × k GEMM 问题，当n、k比m大得多时，计算强度（定义为 MACs/element）大约是m。这种情况适用于LLM解码阶段，因为m是序列数，而n、k是通道大小。

对图  3中 A100 GPU 的 Roofline 进行性能分析，当 m < 78 时，W4A16的理论吞吐量更高，而当 m > 78 时，W8A8表现更好。当输入批处理大小很小时，LLM中的GEMMs是内存受限的，内存带宽由权重流量主导。因此，W4A16的较小内存占用带来了更好的性能。然而，当m很大时，问题就变成了计算受限的。因此，W8A8由于INT8 Tensor Core具有更高吞吐量而速度更快。而作者期望 W4A8 在所有批处理大小上能结合以上两者的优势。

![](https://files.mdnice.com/user/18421/cf5ec5d5-a9b5-40c2-b276-d4197d801c92.png)

综上所述，在计算强度较小时，主要是memory-bound，memory带宽主要被模型权重占据；因此，占用内存更低的W4A16的吞吐量会比W8A8要高。在计算强度较大时，主要是compute-bound，这时因为能够充分利用INT8 Tensor Core，W8A8的吞吐量会更高。而W4A8可以认为兼具两者的优势，不论计算密度是高是低，它都能保持最优的计算吞吐量。

而 LLM 解码阶段Attention的计算密度很低(每个token逐步回归迭代)，这个阶段还是memory-bound，所以 KV Cache 加载越快计算吞吐就越高，因此，采用KV4可以得到KV8的两倍峰值性能。


**那么为什么不选择更激进的W4A4呢**？

当输入序列数 m 超过78时，由memory-bound变为了compute-bound，W4A4开始获得更好的理论GEMM性能，因为4位 Tensor Core 的性能是8位 Tensor Core 的两倍。然而，除了显著的准确性降低，这种理论性能提升在现有的GPU架构（Ampere和Hopper）上无法实现。如图2b所示，现有的W4A4服务系统Atom和QuaRot甚至比TensorRT-LLM中的W16A16解决方案慢得多。

虽然这种性能差距部分可以用这两个系统中的低效运行时来解释，但以前文献中忽略了将逐组（per-group）量化的 W4A4 GEMM 映射到GPU的固有困难。

最先进的系统实现 Tensor Core GEMM 如图4所示的输出固定数据流。

![](https://files.mdnice.com/user/18421/57314dce-7829-404e-8ab2-3598431262c9.png)

对于一个m × n × k GEMM 问题，每个线程块通过顺序遍历reduction维度k来计算一个 $t_m × t_n$ 输出分片（tile）。这个顺序循环被称为主循环。主循环包含100多个迭代，并且占据了GEMM Kernel 的大部分运行时间。

在 FP16 和 W8A8 GEMM（图5a）中，**主循环完全在 Tensor cores 上执行**。

TensorRT-LLM-W4A16（图5b）和 Atom-W4A4（图5c）都**需要在主循环中进行反量化操作，这些操作在 CUDA Core 上运行**。W4A16 需要 INT4到FP16的权重转换，而 Atom-W4A4 需要INT32到FP32的局部和转换和累加。

![](https://files.mdnice.com/user/18421/3b516e57-7376-4864-88ae-206c2e8fc080.png)

Atom主循环中的反量化过程导致了两个显著的效率瓶颈。
- 首先，在像A100和H100这样的GPU上，FP32 CUDA Core 的峰值性能仅为 INT4 Tensor Core 的2%。也就是说，在Atom中反量化一个单独的局部和相当于 50 个 Tensor Core MACs。而主循环由慢速 CUDA Core 操作而不是快速Tensor Core 操作主导。
- 其次，Atom 创建了两组寄存器（一组用于FP32，一组用于INT32）来保存局部和。由于输出固定数据流的特性，较大的GEMM问题（例如：预填充阶段）通常在GPU上而受到寄存器限制，这导致存储局部和的寄存器消耗很高。每个warp消耗大量寄存器限制了可以在流式多处理器（SM）上同时执行的warp数量。值得注意的是，GPU 依赖于大量 in-flight warp 之间进行低成本上下文切换来隐藏延迟。因此，同时执行的warp数量减少限制了延迟隐藏的机会，进一步加剧了主循环开销。

QServe 中的 W4A8 逐组（per-group）量化 GEMM Kernel 设计如图5d。通过采用了两级渐进式分组量化方法，以确保所有计算都在INT8 Tensor Cores上执行。**选择权重反量化而不是局部和反量化，因为它的寄存器压力较低**。此外，应用4路寄存器级并行来同时解码四个INT4权重，进一步减少了主循环开销。

## QoQ

为了实现 W4A8KV4 量化精度的理论吞吐量优势，同时不牺牲大语言模型的有效性。QoQ算法采用渐进式分组量化、SmoothAttention和各种通用量化优化功能。

### 渐进式分组量化

为了提高低比特量化的准确性，通常使用分组量化。然而，它在系统实现中的反量化开销可能会抵消这些准确性的提高。为了解决这个问题，引入了渐进式分组量化，如图6所示。

![](https://files.mdnice.com/user/18421/b8114690-40be-4d14-b663-b6133e3431a3.png)

给定权重张量 $\mathbf{W} \in \mathbb{R}^{k \times n}$，首先，应用逐通道（per-channel）对称INT8量化：

$$\small \hat{\mathbf{W}} = {\mathbf{Q}_{\mathbf{W}}}^{(0)}_{\mathrm{s8}} \cdot \mathbf{s}^{(0)}_{\mathrm{fp16}}$$


其中，${\mathbf{Q}_{\mathbf{W}}}_{\mathrm{s8}}^{(0)} \in \mathbb{N}^{n \times k}$ 是中间8位量化的权重张量，$\mathbf{s}^{(0)}_{\mathrm{fp16}} \in \mathbb{R}^{n \times 1}$ 是逐通道（channel-wise）量化缩放因子。

然后，进一步在中间权重张量上应用逐组（per-group）非对称INT4量化：

$$ \small {{\mathbf{Q}}_{\mathbf{W}}}_\mathrm{s8}^{(0)} = \left({\mathbf{Q}_{\mathbf{W}}}_{\mathrm{u4}} - \mathbf{z}_{\mathrm{u4}} \right)\cdot \mathbf{s}^{(1)}_{\mathrm{u8}}$$

其中, ${\mathbf{Q}_{\mathbf{W}}}_{\mathrm{u4}} \in \mathbb{N}^{n \times k}$ 是无符号4位量化权重张量，$\mathbf{z}_{\mathrm{u4}} \in \mathbb{N}^{n \times k / g}$ 是无符号4位逐组（group-wise）量化零点， $\mathbf{s}^{(1)}_{\mathrm{u8}} \in \mathbb{N}^{n \times k/g}$  是无符号8位逐组（group-wise）量化缩放因子。


对于W4A8 GEMM计算，4位量化权重张量 
${\mathbf{Q}_{\mathbf{W}}}_{\mathrm{u4}}$ 将首先根据上述方程反量化为中间8位量化权重张量 ${\mathbf{Q}_{\mathbf{W}}}_{\mathrm{s8}}^{(0)}$，然后执行INT8矩阵乘法，就好像是 W8A8 逐通道（per-channel）量化一样。



a) 保护量化范围

简单地应用上述方程并不能保证中间反量化权重完全位于8位整数表示范围内。例如，经过INT8量化后，一组8位权重位于[-113, 120]。4位非对称量化将得到缩放因子 ⌈(120−(−113))/(15−0)⌉=16 和零点 ⌈0−(−113)/16⌉=7 。因此，值120被量化为 ⌈120/16+7⌉=15。它将被反量化为 (15−7)×16=128，这超出了最大8位整数127。

一个直接的解决方案是在反量化过程中的算术指令启用饱和选项。然而，简单地应用饱和将严重损害计算吞吐量，速度降低高达67%。

作者重新考虑反量化过程。

将 $\mathbf{Q}_{\mathbf{X}} = \left\lceil \frac{\mathbf{X}}{s}+z\right\rfloor, s=\frac{\mathbf{X}_{\max}-\mathbf{X}_{\min}}{q_{\max}-q_{\min}}, z= \left\lceil q_{\min}-\frac{\mathbf{X}_{\min}}{s}\right\rfloor$ 代入 $\small {{\mathbf{Q}}_{\mathbf{W}}}_\mathrm{s8}^{(0)} = \left({\mathbf{Q}_{\mathbf{W}}}_{\mathrm{u4}} - \mathbf{z}_{\mathrm{u4}} \right)\cdot \mathbf{s}^{(1)}_{\mathrm{u8}}$ 得到：


$$
\small \hat{q}_{s8} = \lfloor \frac{{q}_{s8}}{{s}_{u8}} \rceil \cdot {{s}_{u8}} \le  {q}_{s8} + \frac{1}{2} {{s}_{u8}}.
$$

由于${s}_{u8} = \frac{{{q}_{s8}}_{\max} - {{q}_{s8}}_{\min}}{{{q}_{u4}}_{\max} - {{q}_{u4}}_{\min}} \le \frac{127-(-128)}{15-0} = 17$ ，得到：$\small \hat{q}_{s8} \le 127  \rightarrow {q}_{s8} \le 127 - \frac{1}{2} {{s}_{u8}}  \rightarrow {q}_{s8} \le 119.5$

因此，**将INT8对称量化范围从[-127, 127]缩小到保护范围[-119, 119]，以避免去量化溢出**，如图6顶部所示。


b) 与以前的两级量化方法比较，渐进式分组量化引入了两个层级的缩放因子： $\mathbf{s}^{(0)}_{\mathrm{fp16}}$ 和 $\mathbf{s}^{(1)}_{\mathrm{u8}}$。

以前的研究，如QLoRA中的VSQuant和DoubleQuant，也引入了两级缩放因子来减少组内缩放因子的内存占用。它与这里的量化流程不同，以前的方法直接使用目标精度进行组量化，然后使用组内浮点缩放因子执行逐通道（per-channel）量化，如图6底部所示：

$$\small \hat{\mathbf{W}} = {\mathbf{Q}_{\mathbf{W}}}_{\mathrm{s4}} \cdot \mathbf{s}_{\mathrm{fp16}}, \;\;\;\hat{\mathbf{s}}_{\mathrm{fp16}} = {\mathbf{s}}^{(1)}_{\mathrm{u8}} \cdot \mathbf{s}^{(0)}_{\mathrm{fp16}}$$

因此，使用组内缩放因子 ${\mathbf{s}}^{(1)}_{\mathrm{u8}}$  反量化 $\mathbf{Q}_{\mathbf{W}_{\mathrm{s4}}}$ 不能产生8位权重张量。在GPU上进行计算时，这些方法首先反量化缩放因子（scales），然后反量化权重为浮点值，这最终限制了峰值吞吐量。

DGQ也遵循VSQuant和DoubleQuant的量化方案，但对缩放因子施加限制，以确保所有计算都可以映射到INT8 Tensor Core上。然而，DGQ服务系统**将反量化Kernel与GEMM Kernel 分开**。因此，DGQ中W4A8 GEMM的端到端延迟甚至比cuBLAS中的W8A8 GEMM还要慢，未能展示4位权重量化的内存带宽优势。

相反，QoQ引入了一个保护范围，允许**将反量化操作融合到 W4A8 GEMM Kernel 中，实现全寄存器级并行**，最小化CUDA Core开销。因此，QServe的 W4A8 逐组（per-group）GEMM 比 cuBLAS GEMM 快 1.5 倍。


###  SmoothAttention

如图16所示，直接将KV缓存减少到4位会显著降低LLM的准确性。图7中可视化了采样的Key和Value缓存激活的幅度分布。可以观察到：Value矩阵没有明显的异常值，**而Key矩阵在每个Attention头中都有固定的异常值通道**。

![](https://files.mdnice.com/user/18421/ed4fef05-ec68-47df-8ff2-6a2defc5337f.png)

![](https://files.mdnice.com/user/18421/260df91c-42b9-46ee-bcaf-e1308908d533.png)

这些异常值比大多数激活值大10倍左右。虽然在以前的工作中很容易处理KV8量化，但对KV4量化来说是一个挑战，因为较低的量化层级。

受SmoothQuant的启发，作者提出了SmoothAttention，通过逐通道（per-channel）因子 $\lambda$ 缩小 Key 缓存中的异常通道：

$$ \small \mathbf{Z} = \left(\mathbf{Q}\mathbf{\Lambda}\right)\cdot \left(\mathbf{K}\mathbf{\Lambda}^{-1}\right)^T,\;\;\;\mathbf{\Lambda}=\mathrm{diag}\left(\mathbf{\lambda}\right)$$

SmoothQuant将**量化难度从激活迁移到权重**，因此需要**通过搜索迁移强度来平衡激活和权重量化**。相反，由于**QoQ不量化Query，只需要专注于Key**，并且简单地选择 SmoothAttention 缩放因子为：

$$    \small \mathbf{\lambda}_{i} = \max\left(|\mathbf{K}_i|\right)^{\alpha}.$$

在实践中，$\alpha = 0.5$ 就足够了。

如图7所示，经过SmoothAttention处理后，Key缓存中的异常值已经大大平滑。**为了消除SmoothAttention缩放的额外Kernel调用开销，最好将缩放因子融合到前一层的权重中**。然而，现代LLM使用旋转位置嵌入（RoPE）处理Key和Query，这需要额外处理。在实践中，旋转位置嵌入将通道i与每个Attention头中的通道i + D/2配对。

因此，为了使SmoothAttention缩放在RoPE方面可交换，增加了一个硬约束，即$\lambda_{i} = \lambda_{i + \frac{D}{2}}$，

$$\small \mathbf{\lambda}_{i} = \lambda_{i + \frac{D}{2}} = \max\left(\max\left(|\mathbf{K}_i|\right), \max\left(|\mathbf{K}_{i+\frac{D}{2}}|\right)\right)^{\alpha}$$

之后，可以轻松地将SmoothAttention缩放 $\mathbf{\Lambda}$  融合到前一层的权重中，按照 $\mathbf{W}_{Q} = \mathbf{\Lambda}\mathbf{W}_{Q}$和  $\mathbf{W}_{K} = \mathbf{\Lambda}^{-1}\mathbf{W}_{K}$。


### LLM量化通用优化

低比特LLM量化的一个关键挑战是每个线性层的激活异常值。作者对不同类型的线性层应用不同的优化，如下所述。

1. **块输入模块旋转**

在 Transformer 块中，定义**接收块输入的组件**作为输入模块，例如：QKV投影层和第一个FFN层。如图8所示，受Quarot、Quip的启发，通过**乘以旋转矩阵来旋转块输入激活**。

为了保持线性层的数学等价性，相应地**以相反的方向旋转对应权重**。**旋转后，每个通道的激活是所有其他通道的线性组合，因此有效地抑制了异常值通道**。

此外，由于旋转是酉变换（酉变换，即酉空间V的等度量变换，复数向量空间中保持向量内积不变的线性变换。这种变换保留了向量的长度和向量之间的夹角，使其在许多数学和物理问题中变得非常有用，如在量子力学中，酉变换用来描述系统的演化，保持概率守恒；在信号处理中，酉变换用于保持信号的能量不变），可以将旋转矩阵与前一层的权重融合。这里简单地选择**缩放后的哈达玛矩阵**（哈达玛矩阵是一种方块矩阵。它的矩阵元素为1或-1。其矩阵中不同的行具备正交性质）作为旋转矩阵。

![](https://files.mdnice.com/user/18421/156c1f61-ff27-4a28-af81-8765d2b7ab35.png)

2. **块输出模块平滑**

输出模块指的是**生成块输出的层**，例如：输出投影层和第二个FFN层。如图9所示，受SmoothQuant的启发，**通过除以每个通道的平滑因子来平滑块中间激活**，原始的SmoothQuant没有平滑块中间激活；

此外，如果这里直接用与输入模块相同的迁移强度平滑这些模块（例如：q_proj、up_proj），在Wikitext-2上Llama2-7B模型的困惑度将退化高达0.05。

**在实践中，发现迁移强度$\alpha$应该接近0。也就是说，平滑因子$\lambda$主要由权重而不是激活决定**，这与SmoothQuant中的观察结果非常不同。

![](https://files.mdnice.com/user/18421/f62a6b2e-3a6b-4877-9596-9839d0852de0.png)


3. **激活感知的通道重排序**

AWQ和Atom都观察到，保持显著权重为FP16可以显著提高模型准确性。这些显著的权重可以通过激活分布来识别。与Atom使用的混合精度量化不同，这里提出了激活感知通道重排序，如图10所示。**使用最大(|X|)来确定通道显著性，然后重新排序通道，使得具有相似显著性的通道在同一个量化组中**。

![](https://files.mdnice.com/user/18421/2f670af9-7655-4184-90e1-a6d8ebdbb669.png)

4. **权重裁剪**

权重裁剪是另一种流行的量化优化技术。它通过$\mathbf{W}_{\max}=\alpha \max\left(\mathbf{W}\right)$ 和 $\mathbf{W}_{\min}=\alpha \min\left(\mathbf{W}\right)$ 对方程中 $\mathbf{Q}_{\mathbf{X}} = \left\lceil \frac{\mathbf{X}}{s}+z\right\rfloor, s=\frac{\mathbf{X}_{\max}-\mathbf{X}_{\min}}{q_{\max}-q_{\min}}, z= \left\lceil q_{\min}-\frac{\mathbf{X}_{\min}}{s}\right\rfloor$中的动态范围应用裁剪比率 $\alpha$ 进行裁剪。


以前的方法Quarot、GPTQ、Awq、Atom通过网格搜索裁剪比率 $\alpha$ 来最小化张量本身的量化误差（即：$\|\mathbf{W} - Q\left(\mathbf{W};\alpha\right)\|$ ）或输出均方误差（即：$\|\mathbf{X}\mathbf{W}^T - \mathbf{X}Q\left(\mathbf{W}^T;\alpha\right)\|$ ）。在QServe中，最小化所有线性层（除了 q_proj 和 k_proj）的**层输出误差**，对于 q_proj 和 k_proj，通过优化**块输出均方误差**：
$$\small \arg\min_{\alpha} \|\mathrm{Block}\left(\mathbf{X}; \mathbf{W}\right) - \mathrm{Block}\left(\mathbf{X}; Q\left(\mathbf{W}; \alpha\right)\right)\|$$


## QServe

在介绍了QoQ量化算法之后，实现图3中预测的理论吞吐量优势仍然是一个挑战。因此，下面将深入探讨QServe系统设计，该设计遵循两个重要原则：

1. 减少 GEMM Kernel 中主循环的开销；
2. 使融合注意力 Kernel 保持在内存受限的范围。

###  QServe 系统运行时

首先介绍图11中的QServe运行时。

![](https://files.mdnice.com/user/18421/95f2e32e-f8e2-4230-bcbc-55058fbb1311.png)

QServe中的**所有GEMM层都使用W4A8输入**，在 INT8 Tensor Core 上执行计算，并生成FP16输出。所有注意力层都在CUDA Core上以FP16执行计算。因此，**QServe中的每个LLM块都有FP16输入和FP16输出**。

**激活量化**。为确保每个GEMM输入为INT8激活，对于QKV投影和第一个FFN层，将激活量化融合到前面的 layernorm 中；对于第二个FFN层，则融合到前面的激活 Kernel 中。此外，在注意力块的输出投影之前插入了一个单独的量化节点。

**KV缓存管理**。为了避免内存碎片化，遵循vLLM和TensorRT-LLM的方法，采用分页KV缓存。与这些框架不同，它们对KV缓存执行逐层（per-tensor）静态量化（即，缩放因子离线计算），QServe由于较低的比特精度需要逐头（per-head）动态KV量化以保持准确性。因此，在每个 KV 缓存页面中量化 KV 特征之后，紧跟存储每个头的 FP16 缩放因子和零点，从而允许动态更新这些值。

此外，QServe还支持与vLLM和TensorRT-LLM相似的连续批处理。


###  QServe 中的 W4A8 GEMM

如前文所讨论的，主循环的开销在**量化GEMM以实现roofline模型（图3）预测的理论性能增益**方面构成了重大障碍。因此，QServe W4A8 GEMM的重点在于减少主循环开销。

具体来说，通过计算感知的权重重排序来解决指针算术操作的成本，并采用乘法后减法（subtraction after multiplicatio）计算顺序和寄存器级并行来减少反量化开销。

1. 计算感知的权重重排序：

在反量化和Tensor Core计算之前，运算对象必须从全局内存加载到L1共享内存中，每个主循环迭代期间都是如此。

如图所示，Tensor Core GEMM 本质要求在计算中为每个线程进行跨步（strided）布局。由于Tensor Core GEMM kernel要求每个线程在计算时都采用跨步布局，因此这一加载过程并不简单。

![](https://files.mdnice.com/user/18421/8539a6e0-db54-4e27-bd32-c352468c9016.png)


例如，线程0不是连续加载八个INT8权重，而是首先加载输入通道0-3，然后跳到输入通道16-19。也就是说，一个简单的权重加载实现将需要每个四个通道执行一次地址计算，这将导致两个效率问题。

- 首先，指针算术操作在CUDA Core上执行，其吞吐量比A100上的INT8 Tensor Core 低32倍。因此，地址计算开销变得不可忽视。
- 其次，跨步内存访问阻止了通过**打包128位加载**实现最高的HBM带宽，进一步减慢了内存流水线。

当存储和计算数据类型相同时，ldmatrix指令解决了这个问题。如图12a所示，线程i连续加载输出通道i%8的128位，ldmatrix指令自动以跨步方式分布数据，确保每个线程最终获得INT8 Tensor Core 计算所需的数据。

不幸的是，当用于存储和计算的数据类型不同时（如W4A8），ldmatrix指令将无法工作。

具体来说，在图12b中，ldmatrix确保每个线程在寄存器文件中的数据置换后获得相同数量的字节，而不是相同数量的元素。

因此，线程0获得了T0自身和线程1所需的分片（tiles），而线程1获得了线程2和线程3在随后的INT8 Tensor Core计算中所需的分片（tiles）。这在每个线程获得的数据和计算中使用的数据之间造成了不匹配。

也就是说，ldmatrix无法用于W4A8 GEMM，并且前述的指针算术开销持续存在。更糟糕的是，当我们连续加载4位权重时，内存带宽利用率进一步恶化。

因此，通过计算感知的权重重排序（图12c）解决了这个挑战。关键是以它们在计算中使用的顺序存储权重。通过将整个GEMM问题分成多个32×32分片（tiles） 。在每个分片中，线程0使用输入通道0-3和16-19用于输出通道0、8、16和24（在图12c中省略了输出通道16-31）。因此，将这32个通道连接成一个单独的128位字(word)。

线程1使用的32个通道紧跟在线程0的32个通道之后存储。**由于权重是静态的，这种重排序不会引入任何运行时开销**。

此外，它不仅将指针算术开销降低到与ldmatrix相同的水平，而且还保证了高带宽的**128位/线程**内存事务。将这种重排序应用于零点和缩放因子，以减轻反量化开销。


2. W4A8 GEMM中逐通道（per-channel）快速反量化

如图5d所示，当权重和激活使用的比特精度不同时，在主循环中反量化权重变得必要。在逐通道（per-channel）量化 W4A8 的情况下，省略了第二层级缩放因子 ，第一层级 FP16 缩放因子被有效地融合到GEMM尾部（epilogue）。因此，这里讨论的重点在于，在主循环中，将 ZINT4（即：有零点的无符号4位整数）有效地转换为SINT8。将这种转换进一步分解为两个步骤：UINT4到UINT8（权重解包）和UINT8到SINT8（零点减法）。

如图13所示，重新排列每32个UINT4权重w0、w1、...、w31为w0、w16、w1、w17、...，这使得可以利用寄存器级并行，并以仅三个逻辑运算高效地将它们解包为UINT8数字。

![](https://files.mdnice.com/user/18421/28e6765a-15bd-4c8e-b484-6822e8960de4.png)

对于从UINT8到SINT8的转换，最直接的方法是在主循环中引入整数减法指令，将其称为**乘法前减法（subtraction before multiplication）**。虽然这种方法简单，但不可避免地给主循环带来了额外的成本，这是不可取的。相反，作者**采用乘法后减法（subtraction after multiplicatio）方法，以最小化主循环开销**。

那么，具有逐通道（per-channel）量化操作对象的GEMM层可以表示为：

$$
\small \mathbf{O} = \hat{\mathbf{X}}\hat{\mathbf{W}} = (\mathbf{Q}_\mathbf{X}\odot\mathbf{S}_\mathbf{X})((\mathbf{Q}_\mathbf{W} - \mathbf{Z}_\mathbf{W})\odot\mathbf{S}_\mathbf{W})
$$

其中，$\mathbf{Q}_\mathbf{W}$ ($\mathbf{Q}_\mathbf{X}$) 是量化权重（激活），$\mathbf{Z}_\mathbf{W}$ 将大小为n（输出通道）的零点向量 $\mathbf{z}_\mathbf{W}$ 扩展到k×n（k是输入通道），$\mathbf{S}_\mathbf{W}$, $\mathbf{S}_\mathbf{X}$ 也是从缩放向量 $\mathbf{s}_\mathbf{W},\mathbf{s}_\mathbf{X}$ 获得的。


将  $\mathbf{Z}_\mathbf{W}\odot\mathbf{S}_\mathbf{W}$ 表示为 $\mathbf{ZS}_\mathbf{W}$ ，然后我们重写上面的方程为：

$$
\mathbf{O} = (\mathbf{Q}_\mathbf{X}\odot\mathbf{S}_\mathbf{X})(\mathbf{Q}_\mathbf{W}\odot\mathbf{S}_\mathbf{W}-\mathbf{ZS}_\mathbf{W}) 
= (\mathbf{Q}_\mathbf{X}\mathbf{Q}_\mathbf{W})\odot(\mathbf{s}_\mathbf{W}\times\mathbf{s}_\mathbf{X}) - (\mathbf{Q}_\mathbf{X}\odot\mathbf{S}_\mathbf{X})\mathbf{ZS}_\mathbf{W}
$$


对于第一项，$(\mathbf{Q}_\mathbf{X}\mathbf{Q}_\mathbf{W})\odot(\mathbf{s}_\mathbf{W}\times\mathbf{s}_\mathbf{X})$ ，类似于TensorRT-LLM中的W8A8 GEMM，其中， $\mathbf{s}_\mathbf{W}\times\mathbf{s}_\mathbf{X}$ 外积缩放在尾部（epilogue）执行。

对于第二项，首先用未量化的 $\mathbf{X}$ 替换 $\mathbf{Q}_\mathbf{X}\mathbf{S}_\mathbf{X}$  ($\hat{\mathbf{X}}$)。然后：

$$\mathbf{X}(\mathbf{ZS}_\mathbf{W}) = \mathbf{t}_\mathbf{X}\times(\mathbf{z}_\mathbf{W}\odot\mathbf{s}_\mathbf{W})$$

其中， $\mathbf{t}_\mathbf{X} = \mathbf{X}\mathbf{1}_k$ ，即对每个 token 的所有输入通道求和。

此时注意到方程$\mathbf{X}(\mathbf{ZS}_\mathbf{W}) = \mathbf{t}_\mathbf{X}\times(\mathbf{z}_\mathbf{W}\odot\mathbf{s}_\mathbf{W})$具有类似于缩放因子外积的形式。因此，它也可以融合到W4A8 GEMM的尾部中，类似于第一项$(\mathbf{Q}_\mathbf{X}\mathbf{Q}_\mathbf{W})\odot(\mathbf{s}_\mathbf{W}\times\mathbf{s}_\mathbf{X})$。

为此，将零点减法从主循环移动到尾部，从而在GEMM Kernel 中大大消除了其开销。这种乘法后减法形式的反量化需要预先计算 $\mathbf{t}_\mathbf{X}$。

通常，每个 W4A8 Kernel 之前总是有一个内存受限的 Kernel ，允许将预计算 Kernel 与它融合，因此，几乎没有延迟开销。

3. W4A8 GEMM中逐组（per-group）快速反量化

W4A8 GEMM 中的逐组与它的逐通道的主要区别在于图5d中的第二层级反量化过程。

首先，由于零点现在定义在组的基础上，因此不再可能如前一节所做的将零点减法合并到尾部。

其次，由于存在第二层级缩放，需要对每个权重执行额外的INT8乘法。类似于前一节，必须确定在第二层级反量化过程中是先应用乘法（缩放）还是减法（零点）。在这种情况下，作者认为在乘法后应用减法仍然是有利的方法，因为它启用了寄存器级并行（RLP）。

如图14所示，NVIDIA GPU 提供 vadd4 指令，可通过单个 INT32 ALU 运算执行四个 INT8 加法。然而，没有指令可以实现4路INT8乘法的类似效果。因此，为了实现RLP，必须通过向 8 位缩放因子的最高有效位 (MSB) 填充 24 个零来模拟这一点。

![](https://files.mdnice.com/user/18421/d00fbda4-65fc-4628-afc4-c5db51f15d38.png)

然而，这种模拟只有在每个INT8乘法的结果保持在INT8范围内时才有效。这种条件对于乘法前减法计算顺序并未满足。如图14a所示，**缩放乘法的结果溢出，导致输出不正确**。

在乘法前减法方法中，只能一个接一个地执行乘法，这是极其低效的。而**有了乘法后的减法计算顺序，渐进式分组量化算法确保了初始乘法步骤的结果永远不会超出INT8范围**。这允许在乘法和减法中充分利用RLP的性能优势。

4. 通用优化：在W4A8 Kernel 中，还采用了GEMM优化的通用技术。

在内存方面，应用了多阶段软件流水线和异步内存复制，以更好地重叠内存访问和计算。
此外，交换了L1共享内存的布局，以消除bank冲突。

为了提高L2缓存利用率，跨不同线程块重新排列计算分区，允许相邻块重用相同的权重。

在计算方面，当输入Token（m）的数量很小时，将reduction维度k分割成多个切片并在L1共享内存中跨不同的 warp 来 reduce 局部和是有益的。

### QServe 中的KV4注意力

注意力占据了LLM总运行时间的30-50%，如图2a所示。尽管图5中的roofline模型表明，将KV缓存量化为4比特应该比8比特基线快2倍，但实际情况并非如此。

使用 TensorRT-LLM 的 KV8-attention 解码阶段 Kernel 作为基线，将所有**静态逐层（per-tensor）8位KV缓存量化**的访问和转换替换为它们的**动态逐头（per-head）4位KV缓存量化**。

这种直接替换，在L40S上带来了的1.7倍加速；但在A100上比KV8基线慢1.2倍。作者通过再次分析揭示了问题的症结所在：慢速的 CUDA Core 负责在解码阶段执行注意力 Kernel 。虽然每个单独的批处理GEMV的计算强度为 1 MAC/element，但融合注意力 Kernel 的计算强度显著增加，该 Kernel 结合了所有算术运算和KV缓存更新。例如，简单地从KV缓存中反量化一个INT4需要5个ALU操作。这包括掩码和移位操作以隔离操作数，从整数到浮点表示的类型转换，以及获取最终结果所需的浮点乘法和减法。

更重要的是，A100 FP32 CUDA Core 的 roofline 转折点仅为 9.8 Ops/Byte。也就是说，KV反量化操作本身就已经达到饱和界限，这可能导致融合的KV4注意力Kernel在数据中心GPU（如:A100）上变成计算受限的。

实际上，其他系统（如QuaRot和Atom）也有类似的观察。比如，QuaRot在注意力操作中引入了计算密集的Hadamard变换，使得4位KV缓存量化难以实现比TRT-LLM-KV8更快的速度。

为了缓解计算受限的瓶颈，重要的是要将解码阶段KV4注意力Kernel从计算受限区域转移出去。作者通过2个方法实现了这个目标：首先，延迟roofline转折点的到来，其次，同时减少融合Kernel内的计算强度。

对于第一部分，将原始TensorRT-LLM Kernel 中的所有FP32操作替换为它们的FP16对应物，有效地将计算 roof 翻倍。

对于第二部分，通过应用位技巧，反量化的算术强度可以显著减少到每个元素2个操作。

此外，通过简化控制逻辑和预取缩放因子和零值，从而简化地址计算，也有助于提高性能。在融合了这些增强功能之后，在A100上比TensorRT-LLM的 KV8 Kernel 加速了1.5倍。

## 实验细节

将QoQ与常用的后训练LLM量化技术（如：SmoothQuant、GPTQ、AWQ）以及4位权重-激活量化框架Atom和QuaRot进行了比较，在WikiText2上，对于Llama2-7B模型，与W8A8 SmoothQuant和W4A16 AWQ相比，QoQ的困惑度最多增加了0.16。无论Atom使用W4A4还是W4A8KV4量化精度，QoQ始终优于Atom。与Quarot相比，QoQ在困惑度上也显示出高达0.49的改进。

> 注意：
> 
> 对于SmoothQuant，遵循TensorRT-LLM中的设置，对KV Cache使用静态逐层（per-tensor）对称8位量化。
> 
> 对于GPTQ，使用官方最新的版本，带有“重排序”技巧，记为“GPTQ-R”。
> 
> 对于QuaRot和Atom，主要使用Pile验证数据集作为校准数据集。同时，还使用了WikiText2作为校准数据集（灰色显示）。
> 
> 对于“W4A8KV4 g128”设置，QuaRot和Atom不支持渐进式组量化，因此使用普通分组权重量化（即每个组有一个FP16 缩放因子）来评估他们。不支持的模型和量化为NaN。


![](https://files.mdnice.com/user/18421/375cfb4e-1a18-48df-ba5e-90a0b40fa26a.png)

对于五个常识任务的 Zero-shot 准确性，QoQ显著优于其他4位量化方法。特别是在WinoGrande任务中，与Quarot相比，QoQ的准确性提高了4.82%。与FP16相比，针对7B、13B和70B大小的Llama-2，QoQ量化仅引入了1.03%、0.89%和0.40%的准确性损失。

![](https://files.mdnice.com/user/18421/98ecf269-add8-4de0-97fd-2ff9650de387.png)

通过与TensorRT-LLM（使用FP16、W8A8和W4A16精度）、Atom（W4A4）和QuaRot（W4A4）进行比较，评估了QServe在A100-80G-SXM4和L40S-48G GPU上的效率。与 TensorRT-LLM 的最佳配置相比，QServe在A100上表现出显著的改进，为Llama1-30B实现了2倍的吞吐量提升，为Llama2模型实现了1.2-1.4×的吞吐量提升，为Mistral和Yi模型实现了1.2×的吞吐量提升，为Qwen-1.5模型实现了2.4×的吞吐量提升。在 L40S GPU 上，所有评估模型的吞吐量提高了1.47×至3.47×。

> 注意：
>
> 系统评估指标：在相同的内存限制下可实现的最大吞吐量。
> 
> 使用1024的输入序列长度和512的输出序列长度。
>
> Atom仅支持Llama-2-7B，QuaRot不支持GQA。因此，在测量基线系统的性能时，跳过了这些不支持的模型。
> 
> 在A100上使用逐通道量化，在L40S上使用逐组量化。因为L40S拥有更强大的CUDA Core进行反量化。

![](https://files.mdnice.com/user/18421/fc776f78-62c6-47e5-a82c-597e3d388be6.png)

尽管 L40S 的内存容量明显小于A100，QServe在L40S上以与TensorRT-LLM在A100上，相同的批处理大小下有效地保持了性能。这一成就归因于对权重和KV缓存都应用了激进的4位量化。

在L40S上，以QServe服务的34B以下的七个模型，有五个模型实现了比在A100上使用TensorRT-LLM更高的吞吐量。

在A100上，与Atom和QuaRot相比，性能提升更为显著，因为这些系统性能没有超过TensorRT-LLM。

在L40S上，对于Atom系统支持的模型Llama-2-7B，尽管QServe使用了更高的量化精度，QServe仍然比Atom高出10%的吞吐量；此外，QServe实现的准确性也比Atom更好。



通过探讨QoQ量化方法中不同量化技术对模型性能的影响。以Llama-2-7B模型为例，首先应用了W8A8的量化配置，然后逐步降低量化精度，并逐步应用不同的量化技术。对于每一步，都评估了在WikiText2数据集上的困惑度（perplexity）以及在L40S GPU上使用64个请求、每个请求包含1024个输入Token和512个输出Token时的端到端推理性能。结果显示，**将权重精度降低到4位会显著损害模型性能**，尽管它将端到端处理速度提高了1.12倍，并节省了3.5GB的GPU内存。**旋转块输入模块有助于抑制激活异常值**，使困惑度提高了0.18。此外，**通过权重剪裁最小化块输出均方误差**（MSE）进一步降低了0.16的困惑度。因此，QoQ的W4A8配置实现了与W4A16相当的困惑度。然而，将**KV缓存量化到4位再次使模型性能退化了0.14**，尽管它显著提高了端到端推理吞吐量1.47倍，并减半了GPU内存使用量。为了解决这个问题，**SmoothAttention技术将困惑度降低了0.05，而没有增加系统开销**。渐进式分组量化进一步将困惑度降低0.02，同时仅增加了微不足道的量化开销。最后，激活感知通道重排序将困惑度降低0.03。

![](https://files.mdnice.com/user/18421/e1fca3ba-1e0e-495c-a7dc-1a69696b8001.png)


## 结语

本文介绍了一种W4A8KV4量化算法QoQ，具有4位权重、8位激活和4位KV缓存。QoQ由QServe推理库实现。在QoQ算法中，引入了渐进式量化，在W4A8 GEMM中具有较低的反量化开销。此外，开发了SmoothAttention来有效减轻4位KV量化引起的准确性下降。在QServe系统中，通过执行计算感知的权重重排序，并利用寄存器级并行来减少反量化延迟。此外利用KV4量化提升吞吐性能，并使融合注意力保持在内存受限（memory-bound）区域。

与TensorRT-LLM相比，QServe显著提高了服务吞吐量，比如：Llama-3-8B在A100上1.2×、在L40S上1.4×；以及Qwen1.5-72B在A100上2.4×、在L40S上3.5×，。值得注意的是，QServe在L40S GPU上可以实现比A100上的TensorRT-LLM更高的吞吐量。因此，QServe有效地将LLM服务的成本降低了3倍。


码字不易，如果觉得我的文章能够能够给您带来帮助，期待您的点赞收藏加关注~~


