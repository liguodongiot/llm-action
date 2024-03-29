




##  ZeroQuant

由于内存/计算要求过高，即使对于强大的云服务器来说，如何在实践中有效地服务越来越大的模型也变得异常具有挑战性。在这项工作中，我们提出了一种高效且经济实惠的训练后量化方法来压缩基于 Transformer 的大模型，称为 ZeroQuant。 

ZeroQuant 是一个端到端量化和推理管道，具有三个主要组件：

- 1）针对权重和激活的细粒度硬件友好量化方案； 
- 2）一种新颖的、经济实惠的逐层知识蒸馏算法（LKD），即使无需访问原始训练数据； 
- 3) 提供了一个高度优化的量化系统后端，以消除量化/反量化开销。

因此，我们能够证明：
(1) ZeroQuant 可以将 BERT 和 GPT-3 等模型的权重和激活精度降低到 INT8，对模型准确率的影响最小，同时，与 FP16 推理相比，这些模型的推理速度提高了 5.19 倍/4.16 倍； 
(2) ZeroQuant 加上 LKD 可将全连接模块中的权重量化为 INT4，以及注意力模块中的INT8权重和INT8激活，与FP16模型相比，内存占用减少了3倍；
（3）ZeroQuant可以直接应用于GPT-J和GPT-NeoX等，其中我们的INT8模型达到了与FP16模型相似的精度，但效率提高了5.2倍。



将 INT8 PTQ 应用于 BERT/GPT-3 模型也会导致准确性显著下降。
关键的挑战是 INT8 的表示无法完全捕获权重矩阵中不同行和不同激活Token的不同数值范围。解决这个问题的一种方法是对权重矩阵（激活）使用group-wise（token-wise）量化。





用于权重的分组量化 

分组权重矩阵量化首先在Q-BERT中提出，其中权重矩阵 $W \in R^{n \times m}$被划分为 g 个组，每个组单独量化。然而，在Q-BERT中，作者仅将其应用于量化感知训练。更重要的是，他们没有考虑硬件效率约束，也没有系统后端支持。因此，它们缺乏真正的降低延迟。




用于激活的按token量化 

现有 PTQ 工作的常见做法是对激活使用静态量化，其中最小/最大范围是在离线校准阶段计算的。
对于激活范围方差较小的小模型来说，这种方法可能就足够了。

然而，GPT-3 和 BERT 等大 Transformer 模型的激活范围存在巨大差异。因此，静态量化方案（通常应用于所有tokens/样本）将导致准确度显著下降。克服这个问题的一个自然想法是采用更细粒度的token-wise量化并动态计算每个token的最小/最大范围，以减少激活引起的量化误差。



ZeroQuant 构建了一个高度优化的推理后端，用于Transformer模型 token-wise 量化。
例如，ZeroQuant 的推理后端采用所谓的内核融合技术将量化运算与其先前的运算（如：层归一化）融合，以减轻 token-wise 量化的数据移动成本。类似地，在将最终 FP16 结果写回到下一个 FP16 运算（如：GeLU）的主存储器之前，使用权重和激活量化 scales 缩放 INT32 accumulation，可以减轻不同 GeMM 输出的反量化成本。



Token-wise 量化可以显着减少量化激活的表示误差。它不需要校准激活范围，对于ZeroQuant 的量化方案（INT8 权重和 INT8 激活）不存在与量化相关的成本（例如，激活范围校准）。



我们在第 5 节中的评估还表明，针对激活的 token-wise 量化显著提高了 GPT-3 和 BERT 模型的准确性。




### 权重量化
ZeroQuant （论文：ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers）对权重做group-wise，对激活值做token-wise。用逐层知识蒸馏缓解精度损失（原网络做老师），量化后的网络做学生。和W8A8的普通方法做比较，在BERT和GPT3-style模型上精度更好，还能把权重量化到4bit，但加速效果糟糕。



逐层知识蒸馏

知识蒸馏（KD）是缓解模型压缩后精度下降的最有力方法之一。然而，KD 存在一些局限性，特别是对于大规模语言模型上的隐藏状态 KD：

（1）KD 需要在训练过程中将教师和学生模型放在一起，这大大增加了内存和计算成本； (2)KD通常需要对学生模型进行充分训练。因此，需要在内存中存储权重参数的多个副本（梯度、一阶/二阶动量）来更新模型；
 (3) KD通常需要原始训练数据，有时由于隐私/机密问题而无法访问。

为了解决这些限制，我们提出了逐层蒸馏（LKD）算法。
假设量化的目标模型有 N 个transformer块 L, ..., L，可访问数据集具有输入 (X, Y)，其可以是原始训练数据或来自其他资源的数据集。
我们的 LKD 逐层量化网络，并使用其原始（即未量化）版本作为教师模型。更具体地说，假设第 Li 层将被量化，其量化版本为




---
量化优化的 Transformer kernel


优化推理延迟和模型大小对于在实践中服务大Transformer 模型至关重要。在推理过程中，batch size往往比较小，因此模型的推理延迟主要取决于从主存加载推理所需数据的时间。通过将权重和激活量化到较低的精度，我们减少了加载这些数据所需的数据量，从而可以更有效地使用内存带宽和更高的加载吞吐量。
然而，简单地将权重/激活转换为 INT8 并不能保证延迟的改善，因为存在与量化/反量化操作相关的额外数据移动开销，
如图 2（红色框）所示。这样的开销变得昂贵，并且在某些情况下超过了使用低精度的性能优势。

为了从 token-wise 量化中获得准确性的提高，同时获得更好的延迟，我们现在提出了我们的优化，可以最大限度地提高内存带宽利用率，以加快 ZeroQuant 的推理延迟。


CUTLASS INT8 GeMM 

为了支持 INT8 计算，我们使用针对不同批量大小进行调整的 CUTLASS  INT8 GeMM 实现。与标准 GPU 后端库（例如 cuDNN）不同，使用 CUTLASS 允许我们在 GeMM 之前和之后更灵活地融合量化操作，以减少kernel启动和数据移动开销。



融合Token-wise激活量化

Token-wise量化/反量化引入了许多额外的操作，从而导致额外的数据移动成本。为了消除这些成本，我们使用核融合将激活的量化操作与其之前的 element-wise 和/或 归约操作（例如: bias-add、GeLU 和 LayerNorm）融合到单个运算中，如绿色框所示如图 2 所示。
对于反量化操作（例如，对 GeMM 运算的整数输出进行反量化），我们同样将其与自定义 GeMM 调度融合，以避免对主存储器进行额外的读/写访问，如图 2 中的蓝色框所示。

通过进行上述优化，我们能够在第 5 节中展示 BERT 和 GPT-3 型模型的延迟显着减少。

---


## ZeroQuant-v2


训练后量化 (PTQ) 已成为一种有前途的技术，可减少大语言模型中的内存消耗和计算成本 (LLMs)。然而，目前缺乏对各种量化方案、模型族和量化位精度的系统检查。

在本文中，我们通过使用舍入到最近（RTN）、GPTQ、 ZeroQuant 及其变体。我们将这些方法应用于参数范围从 125M 到 176B 的两个不同的模型系列。

我们的贡献包括：
（1）敏感性分析表明，激活量化通常更容易受到权重量化的影响，较小的模型在激活量化方面通常优于较大的模型； 
(2) 对现有 PTQ 方法进行评估和比较，使模型尺寸减小，同时最大限度地减少对精度的影响，揭示当前的方法使用 INT4 权重 或 INT4 权重和INT8激活 进行量化都无法达到原始模型质量；
（3）基于这些见解，我们提出了一种称为低秩补偿（LoRC）的优化方法，该方法采用低秩矩阵以最小的模型参数大小的增加来提升模型质量的恢复。



进一步突破训练后量化限制提出的方法

根据前几节的调查和结论，显然仍然需要一种先进的方法来进一步完善现有方法，以完全实现原始 FP16 PPL 质量为目标。

在本节中，我们将介绍一种简单而有效的方法，称为LoRC（低阶补偿），以优化当前存在的量化误差，并进一步缩小原始模型质量与其进行量化后对应的模型的质量之间的差距。



LoRC 的灵感来自于对量化误差矩阵 E := W − ˆW 进行低秩矩阵分解，其中 W 表示原始权重，^ W 是量化权重。 LoRC 通过使用两个低秩矩阵 ^ U 和 ^ V 来近似误差 E ： ^ E = ^ U ^ V 。这样可以通过 ^ W= ^ W + ^ E 更准确地近似原始权重矩阵 W，从而减少量化误差：∥W − ˆW ∥ ≥ ∥W − ˆW∥。 LoRC 包含两个步骤： 步骤 I：在误差矩阵 E = U ΣV 上实现奇异值分解（SVD），其中 U ∈ Rand V ∈ Rare 酉矩阵，Σ ∈ Ri 是一个对角矩阵，其对角元素按降序排列方式。



LoRC 的目标是使用低秩矩阵实现误差矩阵 E 的良好近似，同时对模型大小的增加影响最小。例如，考虑标准变压器模型[32]，其中每一层都由多头注意力（MHA）模块和多线性感知（MLP）模块组成。令 h 表示隐藏维度，l 表示层数。参数总数为 12l，每层包含 4h for MHA（用于键、查询、值和投影矩阵）和 8h for MLP（两个大小为 h × 4h 和 4h × h 的矩阵）。在六个中添加了低阶 LoRC



可以得出几个关键的观察结果。首先，LoRC 始终如一地提高所有位大小和块大小的性能，正如 LoRC 激活时较低的困惑度分数所表明的那样。其次，LoRC带来的增强随着比特大小的减小而变得更加显着，尤其是对于W2A16来说更为明显，在大多数场景下与W4A16和W3A16相比，其影响明显更大。最后，细粒度量化与 LoRC 的结合产生了最令人印象深刻的结果，强调了 LoRC 与 FGQ 集成时的功效。总体而言，结果强调了使用 LoRC 增强权重量化性能的好处及其与 FGQ 的兼容性。值得注意的是，恢复最后 0.05-0.1 的困惑度可能具有挑战性，但通过 LoRC，我们几乎能够恢复 INT4 量化的原始模型质量。





## ZeroQuant-FP（浮点W4A8）


在大语言模型领域，在计算效率和保持模型质量之间取得平衡是一项艰巨的挑战。为了克服统一量化的固有局限性，特别是在处理异常值时，并受到 NVIDIA H100 硬件推出的推动，本研究深入探讨了浮点 (FP) 量化的可行性，特别关注 FP8 和 FP4 作为潜在解决方案。

通过调查显示：
- 对于 LLMs，FP8 激活始终优于其整数 (INT8) ，并且在参数超过 10 亿的模型中，性能优势变得更加明显。
- 对于权重量化，FP4 表现出与 INT4 相当（即使不是更优）的性能，从而简化了在 H100 等支持 FP 的硬件上的部署。

为了减轻权重和激活之间的差异引起的精确对齐的开销，本文提出了两个权重量化的缩放约束，与标准 W4A8 模型相比，它们对性能的影响可以忽略不计。我们还通过集成低秩补偿（LoRC）策略来增强我们的量化方法，特别是在较小的模型中也有提升。



---

LLM由于复杂性和计算强度带来了部署挑战，特别是在资源有限的环境中。一种解决方案是量化，它以较低精度格式（例如：8 位整数或浮点数）表示数据，从而减少内存需求，并通过兼容 GPU 上更好的 GEMM 计算吞吐量潜在地增强推理延迟。训练后量化 (PTQ) 会直接降低完全训练模型参数的精度，由于其简单性和较低的计算开销，通常是 LLMs 的首选。最近的研究表明 PTQ 在 8 位整数上(INT8) 仅权重量化不会损害 LLMs 的质量，并且当应用 GPTQ 等高级算法时，INT4 权重量化仅观察到较小的精度下降。


除了仅权重量化之外，对激活量化的探索也引起了人们的兴趣。这种方法利用统一的精度来加快推理时间，从而在硬件上实现更高效的执行。实现激活量化的主要挑战在于效率和性能之间的权衡。正如 ZeroQuants [34, 35]、SmoothQuant [33] 等研究所证明的那样，将激活精度从 FP16 降低到 INT8 不可避免地会导致模型质量下降。这种退化的部分原因是LLMs激活过程中存在极值或异常值。这部分归因于预训练引起。在存在异常值的情况下，像 INT8 或 INT4 这样的统一量化无法准确表示数据的主体，因为它们会偏向异常值。这个问题源于这些技术中统一数据分布的固有假设，该假设可能与实际数据点分布不对应。


考虑到前面描述的整数量化的缺点，采用 ExMy 表示法的浮点 (FP) 方法（如 FP8 或 FP4）成为更有效的替代方案 [20,2,13,28,37]。与整数类型的固定范围不同，浮点方法允许调整小数点位置，从而实现跨激活图的动态缩放并保留重要特征。虽然关于整数和浮点量化之间的模型质量存在争议 [28]，

但最近在 [37] 中使用 FP8/FP4 对 PTQ LLMs 进行的研究表明，FP8 比 INT8 激活量化要好得多。在硬件支持和性能方面，虽然大多数现代CPU和GPU都广泛支持INT8计算[21, 31]，但较低位浮点运算也越来越受到业界的认可。这方面的一个例子是新发布的 NVIDIA H100 GPU，专为 FP8 计算而设计 [20]。因此，尽管与 INT8 相比，FP8 的计算成本可能更高，并且考虑到硬件支持，但模型质量的提高可能使这种权衡变​​得值得，并值得进一步探索。



本文：

通过 FP8 激活和权重量化导致最小的模型退化：特别是在较大的模型中，FP8 激活和权重量化导致的模型退化可以忽略不计，其性能与原始 FP16 模型相当。


在W4A8浮点模型中，即使对缩放因子施加了约束，也能保持质量。
为了在 W4A8 模型中实现真正的效率，从 FP4 到 FP8 的权重转换至关重要。

为了减轻这种转换开销，我们在这里建议权重量化的两种可能的缩放约束：
（1）将所有缩放因子限制为 2 的幂；
（2）需要将缩放因子在一个计算组中（例如，权重矩阵的几行可以通过简单的位移位来转换）。

我们的分析表明，与传统的 W4A8 配置相比，这两个限制对模型性能的影响可以忽略不计。


---

我们选择使我们的方法与 GPTQ 中概述的原则保持一致.虽然这一战略提供了一个坚实的起点.


根据 ZeroQuant-V2 [35]，我们应用了细粒度量化（FGQ）来进行权重，并对激活进行 token-wise 量化。此外，我们还将研究[35]中提出的附加特征LoRC（低秩补偿），其目的是通过采用低秩矩阵分解来减少权重的量化误差。 
LoRC涉及两个主要步骤：首先，它对误差矩阵进行奇异值分解（SVD），误差矩阵是原始权重与量化权重之间的差值。因此，误差矩阵被分解为两个酉矩阵和一个对角矩阵。其次，该方法使用从第一步中的矩阵导出的两个低秩矩阵来制定新的误差近似。然后将该近似值添加到量化权重中，以产生对原始权重的更准确的估计，从而减少量化误差。

基于GPTQ（不带或带LoRC），我们对使用FP8或INT8对激活进行量化以及调整权重量化到FP8和FP4进行全面比较。


我们特别探索了 FP4 权重和 FP8 激活量化的潜力。

将 FP4 投射到 FP8。

最后，由于对权重 (W) 和激活 (A) 使用不同的精度级别，出现了一个独特的挑战。 W4A8 在 H100 NVIDIA 硬件中的实际软件实现是，需要转换 W 的 FP4 以匹配 A 中使用的 FP8 精度。
直接反量化然后再次量化的方法可能会对推理效率产生不利影响，因此不是一个可行的解决方案。

为了解决这个问题，我们提出了位移方法。这意味着，我们不让等式（`Q(x) = INT(x − Z)/S − Z`）中定义的 S 为任何实值比例因子，而是将 S 限制为 2 的幂，即 $S = 2^n$，n ∈ N（当n为负数时，S仍然可以表示分数；当n不为负数时，S仍然可以表示整数。）。我们将实现两种方法：


(M1) 映射到由 2 的幂表示的最接近的值，即让新的scale为 $\hat{S} = 2^{\lceil \log_2(S)\rceil}$

(M2) 首先收集scales形成向量 $\mathbf{S} = [S_1, S_2, \ldots, S_n]$ 。然后取组（group）中的最大值（通常，该集合由矩阵的（多）行组成），记为$S_{\max}$，将这些元素$S_{\max}/S_i$调整为2的幂表示，然后定义 $\hat{S}_i = S_{\max}/ 2^{\lceil \log_2(S_{\max}/S_i)\rceil}$。与 (M1) 相比，这提供了更好的近似值。


我们重申，这种使用 2 的幂的限制，无论是使用 (M1) 还是 (M2)，都可以简化计算，特别是在基于二进制逻辑操作的数字系统中。
这是我们优化计算效率和保持模型性能的方法的关键要素。

----


应用权重和激活的不同整数 (INT) 和浮点 (FP) 量化方法在 LLaMA（上）和 OPT（下）模型上的评估结果。性能以困惑度来衡量（分数越低越好），使用了三个数据集：WikiText-2 (WIKI)、PTB 和 C4。对于每个模型，结果首先显示整个数据集的平均性能，然后是每个数据集的详细分。


FP8 激活比 INT8 好得多。表 2 中结果的高级摘要表明，对于 LLaMA 和 OPT 模型系列，FP8 激活通常优于 INT8 激活。这一观察结果证实了第 2 节中讨论的动机，强调 FP8 捕获更细致信息的卓越能力，这是大规模 LLMs 生成任务的重要方面。

有趣的是，对于参数大于 67 亿的较大模型，例如 LLaMA-7b/13b 和 OPT-6.7b/13b，FP8 相对于 INT8 的优势变得更加明显。


FP8 权重可与 INT8 相媲美，而 FP4 权重则可能优于 INT4。

从表 2 中，我们观察到当保持 FP8 激活时，各种模型和数据集上 INT8 和 FP8 权重量化之间的性能相当。这可能是由于我们在权重量化上使用了 FGQ。有趣的是，当权重量化降低时，FP4 表现出优于 INT4 的某些优势，在 LLaMA-7b（15.14 至 16.09）和 LLaMA-13b 模型（11.08 至 11.31）中尤其明显。具体来说，在 LLaMA-7b 的 W4A8 配置下，我们看到 FP4 比 INT4 提高了 0.95，这是一个显着的增益。 FP4 优于 INT4 的性能对于 H100 等已支持 FP8 的硬件设计尤其有利。因此，适应 FP4 的简单修改将比实现支持 INT4 权重和 FP8 激活的系统更容易。


LoRC 改进了 W4A8。
表 2 显示低阶补偿 (LoRC) 方法增强了 W4A8 量化方案，减少了量化误差。这种改进在较小的模型中尤其明显，突显了 LoRC 在优化这些计算过程的性能方面的有效性，同时对模型大小的影响很小。



将 FP4 投射到 FP8。如第 3 节所述，为了最大限度地提高 NVIDIA H100 硬件上的实际延迟加速，我们建议将权重量化的比例因子 S 表示为 2 的幂。为了实现这一目标，我们使用 FP4 进行权重和FP8 用于激活量化。
表 3 列出了使用和不使用 LoRC 进行的这些实验的结果。我们的数据显示，限制缩放因子偶尔会导致 LLaMA-7b 和 LLaMA-13b 等模型出现改进，

但我们通常会观察到轻微的退化无论我们使用方法 M1 还是 M2，W4A8 浮点模型中的质量都会受到影响。 M2 通常优于 M1。当我们实施 LoRC 时，可以缓解质量下降的情况，特别是在 OPT-1.3b、LLaMA-7b 和 LLaMA-13b 模型中。因此，我们的结果提倡使用 LoRC，特别是在考虑深度学习模型中权重量化的规模限制时。





### 全量化

ZeroQuant-FP（论文：ZeroQuant-FP: A Leap Forward in LLMs Post-Training W4A8 Quantization Using Floating-Point Formats）探索了浮点（FP）量化的适用性，特别关注FP8和FP4格式。研究揭示，对于LLM，FP8激活在性能上持续优于INT8，而在权重量化方面，FP4在性能上与INT4相比具有可比性，甚至更优越。为了解决由权重和激活之间的差异引起的挑战，ZeroQuant-FP要求所有缩放因子为2的幂，并将缩放因子限制在单个计算组内。值得注意的是，ZeroQuant-FP还集成了Low Rank Compensation (LoRC) 策略，以进一步增强其量化方法的有效性。




## ZeroQuant-HERO(W8A8)

量化技术对于减少深度神经网络推理的内存和计算需求至关重要。 ZeroQuant 等现有解决方案为 BERT 和 GPT 等模型提供动态量化，但忽略了内存限制(memory-bounded)运算和每个token量化的复杂性。为了解决这些差距，我们提出了一种新颖的、完全硬件增强的稳健优化的训练后 W8A8 量化框架 ZeroQuant-HERO。

该框架独特地集成了内存带宽和计算密集型运算，旨在实现最佳硬件性能。此外，它还允许特定的 INT8 模块切换到 FP16/BF16 模式，从而提供灵活性，从而提高准确性。



---


由于机器学习算法和硬件之间的跨学科差距（在这项工作中，我们主要针对 Nvidia GPU，例如 A100），该领域仍然很大程度上缺少硬件感知的 PTQ 方法，特别是对于基于 Transformer 的模型。


例如，ZeroQuant [23] 为 BERT [3] 和 GPT [12] 模型提出了对激活进行每token动态量化和对权重进行每列量化，以实现良好的准确性。

然而，它没有考虑
（1）内存限制（memory bounded）算子，例如： LayerNorm 和注意力，并将这些部分留在 FP16/BF16 中，以及
（2）当没有融合机会时，调用额外kernel的每个token量化的成本，例如：注意输出线性层的 INT8 GeMM 算子。


为了解决这些限制，我们引入了 ZeroQuant-HERO，这是一个完全硬件感知且实用的训练后 W8A8 量化框架。我们的贡献总结如下。

1. ZeroQuant-HERO 在设计时考虑了内存带宽限制和计算密集型运算。因此，该框架可以（有可能）实现最佳硬件性能。
2. 为了进一步提高 ZeroQuant-HERO 的可用性，可以执行 ZeroQuant-HERO 的不同量化级别，即 INT8 运算与 FP16/BF16 对应运算的比率，以实现所需的精度和延迟权衡。


----


量化方案


在整个工作中，除非特别的注释说明，否则我们均使用 INT8 对称量化。然而，我们的方法也适用于其他 8 位精度格式，例如 FP8。特别地，我们使用以下列主权重矩阵格式来执行 GeMM：

$Y = XW$

$W = W_{int8}S_w$

$X\in\R^{n\times d}$

$W\in\R^{d\times m}$


$X = S_xX_{int8}$


$S_x\in\R^{1\times d}$

$X = X_{int8}S_x=S_xX_{int8}$





ZeroQuant-HERO 的三个主要组件

嵌入层量化
注意力模块量化
MLP 模块量化










混合精度推理

结合上一节中的所有技术，我们得到了最终的 ZeroQuant-HERO 设计。然而，不同的模型和/或任务对量化的容忍度不同，并且对准确性和系统效率的权衡也有不同的期望。为了满足各种模型/任务的要求，混合精度推理是量化的解决方案之一。

由于 ZeroQuant-HERO 的模块化设计，我们可以为最终模型设置各种量化级别。
为了证明混合精度推理的必要性，我们在下一节中展示了三个量化级别的准确性（表 1）。





为了解决算法与硬件协调的挑战，本文推出了 ZeroQuant-HERO ，这是一种新的硬件增强型训练后 W8A8 量化框架。



