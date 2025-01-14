



- https://nvidia.github.io/TensorRT-LLM/commands/trtllm-build.html



## KV 缓存重用

--use_paged_context_fmha enable

```
triton:
enable_kv_cache_reuse
```


## Chunked Context


--paged_kv_cache enable(过时)

--kv_cache_type paged


## INT8/FP8 KV 缓存


目前，仅支持 per-tensor 量化



## --remove_input_padding

将不同的 token 打包在一起，这样可以减少计算量和内存消耗。

默认值： 'enable'



## --max_num_tokens

每批次中删除填充后批量输入token的最大数量。 

目前，输入填充默认被删除；可以指定--remove_input_padding disable 禁用它。

默认值： 8192


## --context_fmha


在上下文阶段启用融合多头注意力，将触发使用单个内核执行 MHA/MQA/GQA 块的内核。

默认值： 'enable'



## --use_paged_context_fmha


允许 KV 缓存重用和分块上下文等高级功能。


## --use_fp8_context_fmha

当 FP8 量化被激活时，可以通过启用 FP8 Context FMHA 进一步加速注意力




## streamingllm

--streamingllm enable


## --max_input_len

一次请求的最大输入长度。

默认值： 1024

## --max_seq_len, --max_decoder_seq_len

一个请求的最大总长度，包括提示和输出。如果未指定，则从模型配置中推导出该值。



## --weight_streaming

启用将权重卸载到 CPU 并在运行时进行流式加载。

默认值： False


## --reduce_fusion


在 AllReduce 之后将 ResidualAdd 和 LayerNorm 内核融合到单个内核中，从而提高端到端性能。

默认值： 'disable'



## --tokens_per_block

定义每个分页 kv 缓存块中包含多少个token。

默认值： 64


