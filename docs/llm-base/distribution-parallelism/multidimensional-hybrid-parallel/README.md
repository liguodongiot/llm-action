- https://huggingface.co/docs/transformers/perf_train_gpu_many
- https://huggingface.co/transformers/v4.12.5/parallelism.html





## 大模型多维混合并行汇总



|    模型          | DP  | TP  | PP  | ZeRO Stage | FSDP（ZeRO Stage 3） | GPUs         |   FP16/BF16        |
| ------------ | --- | --- | --- | ---------- | ------------------ | ----------------------- | ------ |
| Bloom-176B   | 8   | 4   | 12  | ZeRO-1     | -                  | 384 张 A100 80GB         | BF16    |
| CodeGeeX-13B | 192 | 8   | -   | ZeRO-2     | -                  | 1,536 张 Ascend 910 32GB | FP16  |
| GLM-130B     | 24  | 4   | 8   | ZeRO-1     | -                  | 768 张 A100 40G          | FP16 |
| OPT-175B     | 124   | 8   | -   | -          | ✅             | 992 张 80GB A100         | FP16 |
| Megatron-Turing NLG-530B | 16 | 8   | 35  |  -    | -                  | 4480 张 A100 80G | BF16 |
| GPT-NeoX-20B | 12 | 2   | 4  |ZeRO-1    | -                  | 96 张 A100 40G |   FP16  |







### Bloom-176B

- https://huggingface.co/bigscience/bloom

- https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/tr11-176B-ml.slurm

- https://github.com/bigscience-workshop/bigscience/



```
Hardware:

- 384 A100 80GB GPUs (48 nodes)
  
- Additional 32 A100 80GB GPUs (4 nodes) in reserve

- 8 GPUs per node Using NVLink 4 inter-gpu connects, 4 OmniPath links
  
- CPU: AMD
  
- CPU memory: 512GB per node
  
- GPU memory: 640GB per node
  
- Inter-node connect: Omni-Path Architecture (OPA)
  
- NCCL-communications network: a fully dedicated subnet
  
- Disc IO network: shared network with other types of nodes
  

model:

vocabulary size: 250,680
Total seen tokens: **366B**

Bf16 weights: 329GB
Full checkpoint with optimizer states: 2.3TB



MICRO_BATCH_SIZE=2  # was MBS=1 till GBS=784
GLOBAL_BATCH_SIZE=2048  # 4.2M tokens. It is larger than the initial plan of 3.2M tokens to get higher throughput


NHIDDEN=14336
NLAYERS=70
NHEADS=112
SEQ_LEN=2048



ZERO_STAGE=0 # important: bf16 must use z0! it implements its own zero stage 1 equivalent

```









## CodeGeeX-13B

为了提高训练效率，我们采用8路模型并行训练和192路数据并行训练，启用 ZeRO-2 进一步减少优化器状态的内存消耗。 最后，微批量大小为每个节点 16 个，全局批量大小达到 3,072。


```
Model parameters： 13B
Vocabulary size： 52224
Position embedding： Learnable
Maximum sequence length： 2048
Hidden size h： 5120
Feed-forward size 4h： 20480
Feed-forward activation： FastGELU
Layernorm epsilon： 1e-5
Layernorm precision： FP32
Number of attention heads hn： 40
Attention softmax precision ：FP32
Dropout rate： 0.1

Global batch size: 3072
```


we use Adam optimizer (Kingma and Ba, 2014) to optimize the loss in Equation 2.
The model weights are under FP16 format, except that we use FP32 for layer-norm and softmax for
higher precision and stability. The model takes about 27GB of GPU memory. We start from an initial
learning rate 1e-4, and apply a cosine learning rate decay 




## GLM-130B

```
adopt 4-way tensor parallelism and 8-way pipeline parallelism
96 台 A100（40G*8）


fp16 True

glu_activation geglu
hidden_size 12288
ffn_hidden_size 32768
num_layers 70
num_attention_heads 96
seq_length 2048
global_batch_size 4224
learning_rate 8e-05

```





## OPT-175B

- https://github.com/facebookresearch/metaseq/tree/main/projects/OPT

- [GitHub - facebookresearch/metaseq: Repo for external large-scale work](https://github.com/facebookresearch/metaseq/)

- [Fully Sharded Data Parallel | FairScale documentation](https://fairscale.readthedocs.io/en/stable/api/nn/fsdp.html)

- [Getting Started with Fully Sharded Data Parallel(FSDP) — PyTorch Tutorials 2.0.1+cu117 documentation](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)



```
FP16

trained OPT-175B on 992 80GB A100 GPUs, 

by utilizing Fully Sharded Data Parallel with Megatron-LM Tensor Parallelism

通过利用完全分片数据并行与 Megatron-LM 张量并行

roughly ~33 days of continuous training

300B tokens


```



### BloombergGPT



We use the Amazon SageMaker service provided by AWS to train and evaluate BloombergGPT. 

We use the latest version available at the time of training and
train on a total of 64 p4d.24xlarge instances. 

Each p4d.24xlarge instance has 8 NVIDIA 40GB A100 GPUs with NVIDIA NVSwitch intra-node connections (600 GB/s) and NVIDIA GPUDirect using AWS Elastic Fabric Adapter (EFA) inter-node connections (400 Gb/s).

This yields a total of 512 40GB A100 GPUs.




we rely on stage 3 of ZeRO optimization. We utilize the proprietary SageMaker Model Parallelism (SMP) library from AWS, which enables the automatic distribution of large models across multiple GPU devices and instances


ZeRO shards the training state (model parameters, gradients, and optimizer state) across a group of GPUs. We shard a model across 128 GPUs, and we have 4 copies of the model during training






## Megatron-Turing NLG（530B）

我们使用了 Transformer 解码器的架构 [52]，它是一个从左到右、自回归、基于生成 Transformer 的语言模型，并将其扩展到 5300 亿个参数。 层数、隐藏维度、注意力头分别为 105、20480 和 128。 序列长度为2048，全局批量大小为1920。

我们使用 8 路张量和 35 路管道并行。 学习率为5:0e−5。

我们使用 10 亿个代币进行线性学习率预热。

我们使用余弦衰减来使学习率目标达到超过 3400 亿代币价值的 10%。

在前 120 亿个代币中，我们从 32 的批量大小开始，并以 32 为增量逐渐增加批量大小，直到达到最终的批量大小 1920。

我们使用 Adam 优化器，β1 = 0:9、β2 = 0:95 和 = 10−8。 我们将梯度范数限制为 1.0，并使用 0.1 的权重衰减。 对于权重初始化，我们使用均值为零、标准差为 4:0e−3 的正态分布。 我们的训练数据集由 3390 亿个令牌组成，我们通过混合上述 15 个训练数据集在 2700 亿个令牌上训练 MT-NLG。

我们还留出 2% 的数据进行验证。

对于 MT-NLG 等模型的规模，训练稳定性是一个根本性的挑战。 在训练模型时，我们观察到学习率、权重初始化和 Adam 优化器参数直接影响模型的稳定性。

我们通过在 [9] 中绘制学习率与模型大小来预测 MT-NLG 的学习率。 较高的学习率会增加模型的稳定性。 我们使用大约 p1=(3 ∗ H) 作为权重初始化的标准差，其中 H 表示隐藏维度的大小。 与[45]类似，我们还观察到使用更高的方差进行权重初始化无法收敛。 我们还降低了 β2 的标准值 0:99，以减少训练损失的峰值。







训练过程一共使用了4480块英伟达A100 GPU


5300亿个参数的模型，每个模型副本跨越280个NVIDIA A100 GPU，节点内采用Megatron-LM的8路张量切片（tensor-slicing），节点间采用35路管道并行。


基于NVIDIA DGX SuperPOD的Selene超级计算机上完成混合精度训练。（该超级计算机由560个DGX A100服务器提供支持，每个DGX A100有8个 NVIDIA A100 80GB Tensor Core GPU，通过NVLink 和 NVSwitch相互完全连接）。



Model training is done with mixed precision using 16-bit bfloat on NVIDIA’s Selene supercomputer with 560 DGX A100 nodes. 

Each cluster node has 8 NVIDIA 80-GB A100 GPUs, connected to each other by NVLink and NVSwitch. 

Each node has eight NVIDIA Mellanox 200Gbps HDR Infiniband
HCAs for application communication, with an additional two HCAs per node for dedicated storage. 

The nodes are connected in a three-level (leaf, spine, core) fat-tree topology with 850 switches. 
This topology allows efficient all-reduce communication (which is the dominant communication pattern in deep learning
training). The cluster uses an all-NVME shared parallel filesystem for high-performance data access and
storage. The peak device throughput of an A100 GPU with 16-bit precision is 312 teraFLOP/s, resulting in
an aggregate of 1.4 exaFLOP/s of peak 16-bit precision performance



mixed precision using 16-bit bfloat 



## GPT-NeoX-20B


We trained GPT-NeoX-20B on twelve Supermicro AS-4124GO-NART servers, each with eight
NVIDIA A100-SXM4-40GB GPUs and configured with two AMD EPYC 7532 CPUs. 

All GPUs can directly access the InfiniBand switched fabric through one of four ConnectX-6 HCAs for
GPUDirect RDMA. 

Two NVIDIA MQM8700-HS2R switches—connected by 16 links—compose the spine of this InfiniBand network, with one link
per node CPU socket connected to each switch.

Figure 2 shows a simplified overview of a node as configured for training。







