


- VLLM推理流程梳理（一）: https://zhuanlan.zhihu.com/p/649974825
- VLLM推理流程梳理（二）: https://zhuanlan.zhihu.com/p/649977422
- 大模型推理服务框架vLLM要点简析 (上): https://zhuanlan.zhihu.com/p/654259045
- PagedAttention--大模型推理服务框架vLLM要点简析 (中): https://zhuanlan.zhihu.com/p/655561941




vLLM是一个大模型推理服务框架，声称

最牛的serving 吞吐量PagedAttention
对kv cache的有效管理
传入请求的continus batching，而不是static batching
高性能CUDA kernel
流行的HuggingFace模型无缝集成
有各种decoder算法的高吞吐量服务，包括parallel sampling和beam search等
tensor parallel
兼容OpenAI的API服务


continus batching和PagedAttention


### PagedAttention



### continus batching



