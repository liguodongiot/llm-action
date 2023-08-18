- https://huggingface.co/docs/transformers/perf_train_gpu_many







## 大模型多维混合并行汇总



|              | DP  | TP  | PP  | ZeRO Stage | FSDP（ZeRO Stage 3） | GPUs                    |
| ------------ | --- | --- | --- | ---------- | ------------------ | ----------------------- |
| Bloom-176B   | 8   | 4   | 12  | ZeRO-1     | -                  | 384 张 A100 80GB         |
| CodeGeeX-13B | 192 | 8   | -   | ZeRO-2     | -                  | 1,536 张 Ascend 910 32GB |
| GLM-130B     | 24  | 4   | 8   | ZeRO-1     | -                  | 768 张 A100 40G          |
| OPT-175B     | -   | 8   | -   | -          |                    | 992 张 80GB A100         |



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



roughly ~33 days of continuous training
300B tokens


```




