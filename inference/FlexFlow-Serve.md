

## FlexFlow Serve

- https://github.com/flexflow/FlexFlow/tree/inference
- https://github.com/flexflow/FlexFlow.git


```
pip install flexflow


docker run --gpus all -it --rm --shm-size=8g ghcr.io/flexflow/flexflow-cuda-12.0:latest


docker run -it --gpus all --shm-size=8g \
-v ~/.cache/flexflow:/usr/FlexFlow/inference \
ghcr.io/flexflow/flexflow-cuda-12.0:latest

```



- https://huggingface.co/JackFram/llama-68m/tree/main
- https://huggingface.co/JackFram/llama-160m/tree/main

- https://huggingface.co/JackFram/llama-160m/resolve/main/pytorch_model.bin
- https://huggingface.co/JackFram/llama-160m/resolve/main/config.json
- https://huggingface.co/JackFram/llama-160m/resolve/main/generation_config.json
- https://huggingface.co/JackFram/llama-160m/resolve/main/tokenizer.json
- https://huggingface.co/JackFram/llama-160m/resolve/main/tokenizer.model
- https://huggingface.co/JackFram/llama-160m/resolve/main/tokenizer_config.json




### Speculative 推理

使 FlexFlow Serve 能够加速 LLM 服务的一项关键技术是Speculative推理，它结合了各种集体boost-tuned的小型推测模型 (SSM) 来共同预测 LLM 的输出；

预测被组织为token树，每个节点代表一个候选 token 序列。 使用一种新颖的基于树的并行解码机制，根据 LLM 的输出并行验证由 token 树表示的所有候选 token 序列的正确性。

FlexFlow Serve 使用 LLM 作为 token 树验证器而不是增量解码器，这大大减少了服务生成 LLM 的端到端推理延迟和计算要求，同时，可证明保持模型质量。


### CPU Offloading

FlexFlow Serve 还提供基于Offloading的推理，用于在单个 GPU 上运行大型模型（例如 llama-7B）。

CPU Offloading是将张量保存在CPU内存中，并且在计算时仅将张量复制到GPU。 

请注意，现在我们有选择地offload最大的权重张量（线性、注意力中的权重张量）。 此外，由于小模型占用的空间要少得多，如果不构成GPU内存瓶颈，offload会带来更多的运行空间和计算成本，因此我们只对大模型进行offload。 

[TODO：更新说明] 您可以通过启用 -offload 和 -offload-reserve-space-size 标志来运行offloading示例。



### 量化

FlexFlow Serve 支持 int4 和 int8 量化。 压缩后的张量存储在CPU端。 一旦复制到 GPU，这些张量就会进行解压缩并转换回其原始精度。




### 提示数据集

FlexFlow 提供了五个用于评估 FlexFlow Serve 的提示数据集：
- Chatbot 指令提示：https://specinfer.s3.us-east-2.amazonaws.com/prompts/chatbot.json
- ChatGPT 提示：https://specinfer.s3.us-east-2.amazonaws.com/prompts/chatgpt.json
- WebQA：https://specinfer.s3.us-east-2.amazonaws.com/prompts/webqa.json
- Alpaca：https://specinfer.s3.us-east-2.amazonaws.com/prompts/alpaca.json
- PIQA：https://specinfer.s3.us-east-2.amazonaws.com/prompts/piqa.json




```
import flexflow.serve as ff

ff.init(
        num_gpus=4,
        memory_per_gpu=14000,
        zero_copy_memory_per_node=30000,
        tensor_parallelism_degree=4,
        pipeline_parallelism_degree=1
    )





# Specify the LLM
llm = ff.LLM("decapoda-research/llama-7b-hf")

# Specify a list of SSMs (just one in this case)
ssms=[]
ssm = ff.SSM("JackFram/llama-68m")
ssms.append(ssm)


# Create the sampling configs
generation_config = ff.GenerationConfig(
    do_sample=False, temperature=0.9, topp=0.8, topk=1
)

# Compile the SSMs for inference and load the weights into memory
for ssm in ssms:
    ssm.compile(generation_config)

# Compile the LLM for inference and load the weights into memory
llm.compile(generation_config, ssms=ssms)






# Create the sampling configs
generation_config = ff.GenerationConfig(
    do_sample=False, temperature=0.9, topp=0.8, topk=1
)

# Compile the SSMs for inference and load the weights into memory
for ssm in ssms:
    ssm.compile(generation_config)

# Compile the LLM for inference and load the weights into memory
llm.compile(generation_config, ssms=ssms)    



result = llm.generate("Here are some travel tips for Tokyo:\n")

```



## docker

### 下载预构建的包


运行 FlexFlow 的最快方法是使用预构建的容器，我们会在每次提交到推理分支时更新该容器（推理分支当前领先于主分支）。 可用的容器如下，可以在此链接中找到：

flexflow：FlexFlow 的预构建版本。 我们目前发布了四个针对 AMD GPU 的版本（ROCm 版本：5.3、5.4、5.5 和 5.6），以及多个针对 CUDA GPU 的版本（CUDA 版本：11.1、11.2、11.3、11.4、11.5、11.6、11.7、11.8 和 12.0）。 CUDA 镜像被命名为 flexflow-<GPU 后端>-<GPU 软件版本>，例如 flexflow-hip_rocm-5.6 或 flexflow-cuda-12.0 或

Flexflow-environment：这是 Flexflow 的基础层。 这些包用于 CI 或内部使用，并包含构建/运行 Flexflow 所需的所有依赖项。 如果您想自己构建 FlexFlow，您可能会发现它们很有用。 我们还为 AMD GPU 发布了四个版本的 Flexflow 环境，对于 NVIDIA GPU，为上面列表中的每个 CUDA 版本发布了一个版本。 命名约定也类似。 例如，CUDA 12.0 的 flexflow-environment 映像标记为 flexflow-environment-cuda-12.0。










