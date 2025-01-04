


- https://github.com/NVIDIA/TensorRT-LLM
- https://nvidia.github.io/TensorRT-LLM/index.html

- triton trt-llm后端：https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/docs/model_config.md
- 性能基准：https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/performance/perf-overview.md
- 性能优化最佳实践：https://nvidia.github.io/TensorRT-LLM/performance/perf-best-practices.html


## Batch Manager




## FP8


TensorRT-LLM 为用户提供了易于使用的 Python API 来定义大型语言模型 (LLM) 并构建包含 TensorRT 引擎，以便在 NVIDIA GPU 上高效地执行推理。 

TensorRT-LLM 还包含用于创建执行这些 TensorRT 引擎的 Python 和 C++ 运行时的组件。 它还包括一个用于与 NVIDIA Triton 推理服务集成的后端； 为LLM服务的生产提供保障。 使用 TensorRT-LLM 构建的模型可以在从单个 GPU 到具有多个 GPU 的多个节点（使用张量并行/流水线并行）的各种配置上执行。

TensorRT-LLM 的 Python API 的架构与 PyTorch API 类似。 它为用户提供了包含 einsum、softmax、matmul 或 view 等函数的功能模块。 层模块捆绑了有用的构建块来组装LLM； 比如 Attention 块、MLP 或整个 Transformer 层。 特定于模型的组件，例如 GPTAttention 或 BertAttention，可以在 models 模块中找到。

TensorRT-LLM 附带了几种预定义的流行模型（LLaMA、Bloom等）。它们可以轻松修改和扩展以满足定制需求。 

为了最大限度地提高性能并减少内存占用，TensorRT-LLM 允许使用不同的量化模式执行模型。 TensorRT-LLM 支持 仅 INT4/INT8 权重量化以及 SmoothQuant 技术的完整实现。



精度



|                              | FP32  | FP16  | BF16  | FP8  | INT8 | INT4 |
| :--------------------------- | :---- | :---- | :---- | :--- | :--- | :--- |
| Volta (SM70)                 | Y     | Y     | N     | N    | Y    | Y    |
| Turing (SM75)                | Y     | Y     | N     | N    | Y    | Y    |
| Ampere (SM80, SM86)          | Y     | Y     | Y     | N    | Y    | Y    |
| Ada-Lovelace (SM89)          | Y     | Y     | Y     | Y    | Y    | Y    |
| Hopper (SM90)                | Y     | Y     | Y     | Y    | Y    | Y    |


----


| Model                       | FP32 | FP16 | BF16 | FP8  | W8A8 SQ | W8A16 | W4A16 | W4A16 AWQ | W4A16 GPTQ |
| :-------------------------- | :--: | :--: | :--: | :--: | :-----: | :---: | :---: | :-------: | :--------: |
| Baichuan                    | Y    | Y    | Y    | .    | .       | Y     | Y     | .         | .          |
| BERT                        | Y    | Y    | Y    | .    | .       | .     | .     | .         | .          |
| BLOOM                       | Y    | Y    | Y    | .    | Y       | Y     | Y     | .         | .          |
| ChatGLM                     | Y    | Y    | Y    | .    | .       | .     | .     | .         | .          |
| ChatGLM-v2                  | Y    | Y    | Y    | .    | .       | .     | .     | .         | .          |
| Falcon                      | Y    | Y    | Y    | .    | .       | .     | .     | .         | .          |
| GPT                         | Y    | Y    | Y    | Y    | Y       | Y     | Y     | .         | .          |
| GPT-J                       | Y    | Y    | Y    | Y    | Y       | Y     | Y     | Y         | .          |
| GPT-NeMo                    | Y    | Y    | Y    | .    | .       | .     | .     | .         | .          |
| GPT-NeoX                    | Y    | Y    | Y    | .    | .       | .     | .     | .         | Y          |
| LLaMA                       | Y    | Y    | Y    | .    | Y       | Y     | Y     | Y         | Y          |
| LLaMA-v2                    | Y    | Y    | Y    | Y    | Y       | Y     | Y     | Y         | Y          |
| OPT                         | Y    | Y    | Y    | .    | .       | .     | .     | .         | .          |
| SantaCoder                  | Y    | Y    | Y    | .    | .       | .     | .     | .         | .          |
| StarCoder                   | Y    | Y    | Y    | .    | .       | .     | .     | .         | .          |













