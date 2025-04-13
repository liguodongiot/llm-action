

## qwen

```
git clone https://www.modelscope.cn/qwen/Qwen-7B-Chat.git
```


## Qwen1.5

Qwen1.5版本本次开源了包括0.5B、1.8B、4B、7B、14B和72B在内的六种大小的基础和聊天模型，同时，也开源了量化模型。不仅提供了Int4和Int8的GPTQ模型，还有AWQ模型，以及GGUF量化模型。

为了提升开发者体验，Qwen1.5的代码合并到Hugging Face Transformers中，开发者现在可以直接使用transformers>=4.37.0而无需trust_remote_code。

此外，Qwen1.5支持了例如vLLM、SGLang、AutoGPTQ等框架对Qwen1.5的支持。

Qwen1.5显著提升了聊天模型与人类偏好的一致性，并且改善了它们的多语言能力。所有模型提供了统一的上下文长度支持，支持32K上下文, 基础语言模型的质量也有所改进。



## 模型

```
git clone https://www.modelscope.cn/qwen/Qwen1.5-0.5B.git
git clone https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
```



## 代码

- https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py



## 模型结构


https://blog.csdn.net/fan_fan_feng/article/details/138978901
