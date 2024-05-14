




## 基于Megatron-LM实现的项目

- [CodeGeeX](https://github.com/THUDM/CodeGeeX)

- [如何使用 Megatron-LM 训练语言模型](https://huggingface.co/blog/zh/megatron-training)：数据预处理，训练，模型转换，推理等







### 数据加载

Megatron-LM 带有一个高效的 DataLoader，其中数据在训练前被 tokenize 和 shuffle。它还将数据拆分为带有索引的编号序列，并将索引存储，因此 tokenize 只需要计算一次。为了构建索引，首先根据训练参数计算每个 epoch 的数量，并创建一个排序，然后对数据进行 shuffle 操作。这与大多数情况不同，我们通常迭代整个数据集直到其用尽，然后重复第二个 epoch 。这平滑了学习曲线并节省了训练时间。


### 融合 CUDA 内核
当一个计算在 GPU 上运行时，必要的数据会从内存中取出并加载到 GPU 上，然后计算结果被保存回内存。简单来说，融合内核的思想是: 将通常由 PyTorch 单独执行的类似操作组合成一个单独的硬件操作。因此可以将多个离散计算合并为一个，从而减少在多个离散计算中的内存移动次数。


当 f、g 和 h 融合在一个内核中时，f 和 g 的中间结果 x' 和 y' 存储在 GPU 寄存器中并立即被 h 使用。但是如果不融合，x' 和 y' 就需要复制到内存中，然后由 h 加载。因此，融合 CUDA 内核显着加快了计算速度。此外，Megatron-LM 还使用 Apex 的 AdamW 融合实现，它比 PyTorch 实现更快。

虽然我们可以在 transformers 中自定义 Megatron-LM 中的 DataLoader 和 Apex 的融合优化器，但自定义融合 CUDA 内核对新手来说太不友好了。


