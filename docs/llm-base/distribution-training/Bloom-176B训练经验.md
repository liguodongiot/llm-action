


- https://huggingface.co/blog/zh/bloom-megatron-deepspeed


用 FP16 训练巨型 LLM 模型是一个禁忌。FP16 经常溢出！FP16 的最大数值范围为 64k，您只能进行较小数的乘法。例如你可以做 250*250=62500，但如果你尝试 255*255=65025，你就会溢出，这是导致训练出现问题的主要原因。这意味着你的权重必须保持很小。一种称为损失缩放 (loss scaling) 的技术有助于缓解这个问题，但是当模型变得非常大时，FP16 较小的数值范围仍然是一个问题。




由于 BF16 和 FP16 的大小相同，均为 2 个字节，因此，没有免费的午餐，当使用 BF16 时，代价就是它的精度非常差。然而，你应该还记得我们在训练时采用的随机梯度下降法及其变体，该方法有点像蹒跚而行，如果你这步没有找到完美的方向其实没关系，你会在接下来的步骤中纠正自己。

无论使用 BF16 还是 FP16，都有一个权重副本始终在 FP32 中 —— 这是由优化器更新的内容。因此 16 位格式仅用于计算，优化器以全精度更新 FP32 权重，然后将它们转换为 16 位格式以用于下一次迭代。

所有 PyTorch 组件都已更新，以确保它们在 FP32 中执行任何累加，因此不会发生精度损失。

一个关键问题是梯度累积，它是流水线并行的主要特征之一，因为每个 micro batch 处理的梯度都会累积。在 FP32 中实现梯度累积以保证训练的精确性至关重要，这正是 BF16Optimizer 所做的。

除了其他改进之外，我们认为使用 BF16 混合精度训练将潜在的噩梦变成了一个相对平稳的过程。





---

## slurm


- https://github.com/bigscience-workshop/bigscience/blob/master/train/tr10-13B-ml/tr10-13B.slurm
- https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/tr11-176B-ml.slurm








