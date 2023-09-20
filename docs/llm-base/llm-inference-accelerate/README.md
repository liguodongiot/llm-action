
- LLM推理优化技术综述：KVCache、PageAttention、FlashAttention、MQA、GQA：https://zhuanlan.zhihu.com/p/655325832

## KVCache


## PageAttention



PageAttention是目前kv cache优化的重要技术手段，目前最炙手可热的大模型推理加速项目VLLM的核心就是PageAttention技术。

在缓存中，这些 KV cache 都很大，并且大小是动态变化的，难以预测。已有的系统中，由于显存碎片和过度预留，浪费了60%-80%的显存。

PageAttention提供了一种技术手段解决显存碎片化的问题，从而可以减少显存占用，提高KV cache可使用的显存空间，提升推理性能。



---


首先，PageAttention命名的灵感来自OS系统中虚拟内存和分页的思想。可以实现在不连续的空间存储连续的kv键值。

另外，因为所有键值都是分布存储的，需要通过分页管理彼此的关系。序列的连续逻辑块通过 block table 映射到非连续物理块。

另外，同一个prompt生成多个输出序列，可以共享计算过程中的attention键值，实现copy-on-write机制，即只有需要修改的时候才会复制，从而大大降低显存占用。





## FlashAttention

Flash attention推理加速技术是利用GPU硬件非均匀的存储器层次结构实现内存节省和推理加速，它的论文标题是“FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness”。意思是通过合理的应用GPU显存实现IO的优化，从而提升资源利用率，提高性能。


首先我们要了解一个硬件机制，计算速度越快的硬件往往越昂贵且体积越小，Flash attention的核心原理是尽可能地合理应用SRAM内存计算资源。

A100 GPU有40-80GB的高带宽内存(HBM)，带宽为1.5-2.0 TB/s，而每108个流处理器有192KB的SRAM，带宽估计在19TB/s左右。

也就是说，存在一种优化方案是利用SRAM远快于HBM的性能优势，将密集计算尽放在SRAM，减少与HBM的反复通信，实现整体的IO效率最大化。比如可以将矩阵计算过程，softmax函数尽可能在SRAM中处理并保留中间结果，全部计算完成后再写回HBM，这样就可以减少HBM的写入写出频次，从而提升整体的计算性能。

如何有效分割矩阵的计算过程，涉及到flash attention的核心计算逻辑Tiling算法。



### v1



### v2







## MQA/GQA


MQA，全称 Multi Query Attention, 而 GQA 则是前段时间 Google 提出的 MQA 变种，全称 Group-Query Attention。MHA（Multi-head Attention）是标准的多头注意力机制，h个Query、Key 和 Value 矩阵。MQA 让所有的头之间共享同一份 Key 和 Value 矩阵，每个头只单独保留了一份 Query 参数，从而大大减少 Key 和 Value 矩阵的参数量。GQA将查询头分成N组，每个组共享一个Key 和 Value 矩阵。


GQA以及MQA都可以实现一定程度的Key value的共享，从而可以使模型体积减小，GQA是MQA和MHA的折中方案。这两种技术的加速原理是（1）减少了数据的读取（2）减少了推理过程中的KV Cache。需要注意的是GQA和MQA需要在模型训练的时候开启，按照相应的模式生成模型。





- 大语言模型推理性能优化综述: https://zhuanlan.zhihu.com/p/656485997

LLM 推理服务重点关注两个指标：吞吐量和时延：

吞吐量：主要从系统的角度来看，即系统在单位时间内能处理的 tokens 数量。计算方法为系统处理完成的 tokens 个数除以对应耗时，其中 tokens 个数一般指输入序列和输出序列长度之和。吞吐量越高，代表 LLM 服务系统的资源利用率越高，对应的系统成本越低。
时延：主要从用户的视角来看，即用户平均收到每个 token 所需位时间。计算方法为用户从发出请求到收到完整响应所需的时间除以生成序列长度。一般来讲，当时延不大于 50 ms/token 时，用户使用体验会比较流畅。
吞吐量关注系统成本，高吞吐量代表系统单位时间处理的请求大，系统利用率高。时延关注用户使用体验，即返回结果要快。这两个指标一般情况下需要会相互影响，因此需要权衡。例如， 提高吞吐量的方法一般是提升 batchsize，即将用户的请求由串行改为并行。但 batchsize 的增大会在一定程度上损害每个用户的时延，因为以前只计算一个请求，现在合并计算多个请求，每个用户等待的时间变长。

LLM 推理性能优化主要以提高吞吐量和降低时延为目的.



## 显存相关优化




### KV Cache


大模型推理性能优化的一个最常用技术就是 KV Cache，该技术可以在不影响任何计算精度的前提下，通过空间换时间思想，提高推理性能。
目前业界主流 LLM 推理框架均默认支持并开启了该功能。



KV Cache 的引入也使得推理过程分为如下两个不同阶段，进而影响到后续的其他优化方法。

预填充阶段：发生在计算第一个输出 token 过程中，计算时需要为每个 Transformer layer 计算并保存 key cache 和 value cache；FLOPs 同 KV Cache 关闭一致，存在大量 GEMM (GEneral Matrix-Matrix multiply) 操作，属于 Compute-bound 类型计算。
解码阶段：发生在计算第二个输出 token 至最后一个 token 过程中，这时 KV Cache 已存有历史键值结果，每轮推理只需读取 Cache，同时将当前轮计算出的新的 Key、Value 追加写入至 Cache；GEMM 变为 GEMV (GEneral Matrix-Vector multiply) 操作，FLOPs 降低，推理速度相对预填充阶段变快，这时属于 Memory-bound 类型计算。


### Paged Attention


LLM 推理服务的吞吐量指标主要受制于显存限制。研究团队发现现有系统由于缺乏精细的显存管理方法而浪费了 60% 至 80% 的显存，浪费的显存主要来自 KV Cache。因此，有效管理 KV Cache 是一个重大挑战。



在 Paged Attention 之前，业界主流 LLM 推理框架在 KV Cache 管理方面均存在一定的低效。

HuggingFace Transformers 库中，KV Cache 是随着执行动态申请显存空间，由于 GPU显存分配耗时一般都高于 CUDA kernel 执行耗时，因此动态申请显存空间会造成极大的时延开销，且会引入显存碎片化。

FasterTransformer 中，预先为 KV Cache 分配了一个充分长的显存空间，用于存储用户的上下文数据。例如 LLaMA-7B 的上下文长度为 2048，则需要为每个用户预先分配一个可支持 2048 个 tokens 缓存的显存空间。如果用户实际使用的上下文长度低于2048，则会存在显存浪费。

aged Attention 将传统操作系统中对内存管理的思想引入 LLM，实现了一个高效的显存管理器，通过精细化管理显存，实现了在物理非连续的显存空间中以极低的成本存储、读取、新增和删除键值向量。

具体来讲，Paged Attention 将每个序列的 KV Cache 分成若干块，每个块包含固定数量token 的键和值。





首先在推理实际任务前，会根据用户设置的 max_num_batched_tokens 和 gpu_memory_util 预跑一次推理计算，记录峰值显存占用量 peak_memory，然后根上面公式获得当前软硬件环境下 KV Cache 可用的最大空间，并预先申请缓存空间。

其中，max_num_batched_tokens 为部署环境的硬件显存一次最多能容纳的 token 总量，
gpu_memory_util 为模型推理的最大显存占用比例，
total_gpu_memory 为物理显存量， 
block_size 为块大小（默认设为 16）。



在实际推理过程中，维护一个逻辑块到物理块的映射表，多个逻辑块可以对应一个物理块，通过引用计数来表示物理块被引用的次数。

当引用计数大于一时，代表该物理块被使用，当引用计数等于零时，代表该物理块被释放。

通过该方式即可实现将地址不连续的物理块串联在一起统一管理。

Paged Attention 技术开创性地将操作系统中的分页内存管理应用到 KV Cache 的管理中，提高了显存利用效率。

另外，通过 token 块粒度的显存管理，系统可以精确计算出剩余显存可容纳的 token 块的个数，配合后文 Dynamic Batching 技术，即可避免系统发生显存溢出的问题。




## 计算相关优化



### 算子融合

算子融合是深度学习模型推理的一种典型优化技术，旨在通过减少计算过程中的访存次数和 Kernel 启动耗时达到提升模型推理性能的目的，该方法同样适用于 LLM 推理。



目前业界基本都针对 Transformer layer 结构特点，手工实现了算子融合。以 DeepSpeed Inference 为例，算子融合主要分为如下四类：

归一化层和 QKV 横向融合：将三次计算 Query/Key/Value 的操作合并为一个算子，并与前面的归一化算子融合。
自注意力计算融合：将自注意力计算涉及到的多个算子融合为一个，业界熟知的 FlashAttention 即是一个成熟的自注意力融合方案。
残差连接、归一化层、全连接层和激活层融合：将 MLP 中第一个全连接层上下相关的算子合并为一个。
偏置加法和残差连接融合。



由于算子融合一般需要定制化实现算子 CUDA kernel，因此对 GPU 编程能力要求较高。随着编译器技术的引入，涌现出 OpenAI Triton 、TVM 等优秀的框架来实现算子融合的自动化或半自动化，并取得了一定的效果。




### 高性能算子


针对 LLM 推理运行热点函数编写高性能算子，也可以降低推理时延。

GEMM 操作相关优化：在 LLM 推理的预填充阶段，Self-Attention 和 MLP 层均存在多个 GEMM 操作，耗时占据了推理时延的 80% 以上。GEMM 的 GPU 优化是一个相对古老的问题，在此不详细展开描述算法细节。英伟达就该问题已推出 cuBLAS、CUDA、CUTLASS 等不同层级的优化方案。例如，FasterTransformer 框架中存在大量基于 CUTLASS 编写的 GEMM 内核函数。另外，Self-Attention 中存在 GEMM+Softmax+GEMM 结构，因此会结合算子融合联合优化。


GEMV 操作相关优化：在 LLM 推理的解码阶段，运行热点函数由 GEMM 变为 GEMV。相比 GEMM，GEMV 的计算强度更低，因此优化点主要围绕降低访存开销开展。


## 服务相关优化

服务相关优化主要包括 Continuous Batching、Dynamic Batching 和 异步 Tokenize / Detokenize。其中 Continuous Batching 和 Dynamic Batching 主要围绕提高可并发的 batchsize 来提高吞吐量，异步 Tokenize / Detokenize 则通过多线程方式将 Tokenize / Detokenize 执行与模型推理过程时间交叠，实现降低时延目的。














