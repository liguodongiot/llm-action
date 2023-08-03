
## 摘要

多种训练后量化方法已应用于大语言模型 (LLM)，并且已被证明在低至 8 比特的情况下也能表现良好。

我们发现这些方法在较低比特精度下会崩溃，并研究了 LLM 的量化感知训练(LLM-QAT) ，以进一步提高量化水平。

我们提出了一种 data-free 蒸馏方法，该方法利用预训练模型产生的生成，可以更好地保留原始输出分布，并允许独立于其训练数据来量化任何生成模型，类似于训练后量化方法。

除了量化权重和激活之外，我们还量化 KV 缓存，这对于提高吞吐量和支持当前模型大小的长序列依赖关系至关重要。

我们在低至 4 比特的量化级别上对大小为 7B、13B 和 30B 的 LLaMA 模型进行了实验。 我们观察到比 training-free 方法有很大的改进，特别是在低比特设置中。

## 1 Introduction







常识推理任务



我们考虑三种训练后量化（PTQ）方法：round-to-nearest（RTN）、GPT-Q 和 SmoothQuant 作为基线。 
我们在几种不同的设置中与它们进行比较，其中：权重、激活和 KV 缓存值被量化到不同的级别（表示为 W-A-KV）。 

不同的 PTQ 方法在不同的设置中表现良好，我们将我们的方法与每个设置中的最佳 PTQ 结果进行比较。



## 2 方法


在这项工作中，我们研究线性量化，即均匀量化。 根据实际值是否被截断（clipped），线性量化可以分为两类：保留所有值范围的 MinMax 量化和基于截断（裁剪）的量化。


在 MinMax 量化中，量化过程可以表述为：

```math
\mathbf{X}_\mathbf{Q}^i = \alpha \mathbf{\hat{X}_Q}^i = \alpha \lfloor {\frac{\mathbf{X}_\mathbf{R}^i - \beta}{\alpha}} \rceil + \beta
```

$$
\mathbf{X}_\mathbf{Q}^i = \alpha \mathbf{\hat{X}_Q}^i = \alpha \lfloor {\frac{\mathbf{X}_\mathbf{R}^i - \beta}{\alpha}} \rceil + \beta
$$

```math
\mathbf{X}_\mathbf{Q}^i = \alpha \mathbf{\hat{X}_Q}^i = \alpha \nint {\frac{\mathbf{X}_\mathbf{R}^i - \beta}{\alpha}}  + \beta
```



```math
\mathbf{X}_\mathbf{Q}^i = \alpha \mathbf{\hat{X}_Q}^i = \alpha \lfloor{{\rm Clip}(\frac{\mathbf{X}_\mathbf{R}^i - \beta}{\alpha}, 0, 1)}\rceil + \beta
```


$$
\mathbf{X}_\mathbf{Q}^i = \alpha \mathbf{\hat{X}_Q}^i = \alpha \lfloor{{\rm Clip}(\frac{\mathbf{X}_\mathbf{R}^i - \beta}{\alpha}, 0, 1)}\rceil + \beta
$$


$$
\mathbf{X}_\mathbf{Q}^i = \alpha \lfloor{\frac{\mathbf{X}_\mathbf{R}^i}{\alpha}}\rceil, \ \ \ \alpha = \frac{\max(|\mathbf{X}_\mathbf{R}|)}{2^{N-1} -1}
$$


$$
\mathcal{L}_{CE} = -\frac{1}{n}\sum_c\sum^n_{i=1} p_c^{\mathcal{T}}(X_i)\log(p_c^{\mathcal{S}}(X_i)),
$$


```math
\mathbf{X}_\mathbf{Q}^i = \alpha \mathbf{\hat{X}_Q}^i = \alpha \lfloor{{\rm Clip}(\frac{\mathbf{X}_\mathbf{R}^i - \beta}{\alpha}, 0, 1)}\rceil + \beta
```


```math

\mathbf{X}_\mathbf{Q}^i = \alpha \lfloor{\frac{\mathbf{X}_\mathbf{R}^i}{\alpha}}\rceil, \ \ \ \alpha = \frac{\max(|\mathbf{X}_\mathbf{R}|)}{2^{N-1} -1}
```

```math
\mathcal{L}_{CE} = -\frac{1}{n}\sum_c\sum^n_{i=1} p_c^{\mathcal{T}}(X_i)\log(p_c^{\mathcal{S}}(X_i)),
```

## 3.2 Main Results

对于从业者来说，一个重要的问题是是否使用完全精确的小模型，或者具有类似推理成本的较大量化模型。

虽然确切的权衡可能会因多种因素而异，但我们可以根据我们的结果提出一些建议。

首先，8 位量化应该优于较小的全精度模型，并且 PTQ 方法足以满足这种情况。 8-8-8 30B 量化模型优于类似大小的 13B 模型，并且在实践中应该具有更低的延迟和更高的吞吐量。 这也适用于 8 位 13B 模型与 16 位 7B 模型相比。

此外，使用 LLM-QAT 量化的 4 位模型应该优于类似大小的 8 位模型。 例如，4-8-4 LLM-QAT 30B 优于 8 位 LLaMA-13B，4-8-8 LLM-QAT 13B 优于 8 位 LLaMA-7B。

因此，我们建议使用 4 位 LLM-QAT 模型，以实现最佳效率与精度的权衡。



---

Quantization-aware training for key-value cache
然而，之前只有少数工作解决了 LLM 中的 KV 缓存量化问题，且方法主要局限于训练后量化（Sheng et al., 2023）。 

在我们的研究中，我们证明了可以采用用于激活量化的类似量化感知训练方法来量化 KV 缓存。 

如图 3 所示，我们在等式 3 中采用按token量化。

假设key和value是由token生成的。 

在生成过程中，当前的key和value都会被量化，并存储它们对应的缩放因子。 

在 QAT 的训练过程中，我们对键和值的整个激活张量进行量化，如图 2 所示。
通过将量化函数集成到梯度计算中，我们确保使用量化的键值对进行有效的训练。



---

知识蒸馏

我们使用基于交叉熵的logits蒸馏从全精度预训练的教师网络中训练量化的学生网络：




正如第 2.1 节中所讨论的，在数据生成过程中，重要的是从分布中采样下一个标记，而不是总是选择 top-1 候选标记。 

通过这样做，下一个标记不一定代表训练学生模型的最佳标签，因为采样会引入固有的噪声。 
因此，我们建议利用预训练模型的预测作为软标签，这为指导学生模型的训练提供了更多信息的目标。 
我们在第 3.3.3 节中提出了一项全面的消融研究，以深入研究这种方法的细节。



## 3.3 Ablation 消融实验




### 3.3.2 Quantization Function

我们将非裁剪量化方法与基于裁剪的量化方法进行了比较，如表 4 所示。

遵循之前工作（Liu et al., 2022b, 2023）中的做法，我们使用 StatsQ（Liu et al., 2022a），这是一种统计计算的缩放因子，用于基于裁剪的权重量化

和 LSQ（Esser 等人，2019），用于基于裁剪的激活量化的可学习缩放因子。

然而，我们的研究结果表明，这两种最先进的基于裁剪的量化方法并没有超过非裁剪对称方法MinMax所实现的性能。

这一观察结果强化了这样的论点：保留异常值对于大型语言模型的性能至关重要。

此外，我们观察到，对于 LLaMA 模型，激活和权重主要表现出对称分布，这使得使用对称量化成为最佳选择。 然而，值得注意的是，这个结论可能不适用于其他大型语言模型，尤其是那些包含 GeLU 层的语言模型。


## 3.3.3  知识蒸馏

表5显示不同的知识蒸馏方法对微调模型的最终精度有显着影响。

值得注意的是，由于在生成过程中从候选分布中进行采样所引入的固有随机性和噪声，单独使用下一个标记作为标签并不是最优的。

相比之下，logit 蒸馏利用了教师模型的完整 logit 分布预测，与基于标签的训练方法相比，微调模型的性能更优越。

有趣的是，我们观察到合并注意力蒸馏或隐藏层蒸馏实际上会阻碍性能。

因此，我们在所有实验中专门采用 Logit 蒸馏。


3.4 与 SmoothQuant 的兼容性

我们的方法还与 SmoothQuant（Xiao 等人，2022）中提出的权重激活重新缩放技术兼容。

表 6 显示，将 SmoothQuant 合并到 4 位权重 4 位激活 (W4A4) 量化中可以进一步提高精度。

然而，在激活位大于权重位（即W4A8）的情况下，添加SmoothQuant不会产生任何改进，甚至可能会损害性能。




## 结论

我们提出了针对 LLM 的 data-free 量化感知训练，并表明使用该技术可以实现准确的 4 位量化。

考虑到与训练数据无关的蒸馏方法的普遍性，以及 LLM 部署成本的不断增长，我们期望我们的方法具有广泛的适用性。 
例如，该方法还可以用于在多个阶段训练的模型，例如 通过指令精调或强化学习。 

我们将这项调查留给未来的工作。 由于 4 比特量化没有开箱即用的硬件支持，因此，我们没有将硬件实现作为这项工作的一部分。 
不过，我们正在与合作伙伴合作，以期在不久的将来实现这一目标。 虽然我们的方法适用于 4 比特权重、4 比特 KV 缓存和 8 比特激活，
但对于 4 比特激活量化来说还不充分，还需进一步研究。
