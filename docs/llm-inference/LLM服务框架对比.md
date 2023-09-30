




## FlexFlow Server

- https://github.com/flexflow/FlexFlow/tree/inference


指标：

每秒生成token的延迟


模型：

LLaMA-30B

LLaMA-65B

OPT-30B


对比框架：vLLM、TGI、FT



## vLLM

- https://github.com/vllm-project/vllm/tree/main
- https://vllm.ai/
- https://github.com/vllm-project/vllm/tree/main/benchmarks


指标：

吞吐 

延迟 



对比框架：


TGI、HF、vLLM





## FT

- https://github.com/NVIDIA/FasterTransformer/blob/main/docs/gpt_guide.md

Latency on input length 60, output length 20, TP means tensor parallelism and PP means pipeline parallelism.


每秒处理的句子

Throughput per GPU on input length 60, output length 20. TP means tensor parallelism and PP means pipeline parallelism.

Megatron 530B


GPT 175B




## Huggingface TGI


- https://github.com/huggingface/text-generation-inference/tree/main/benchmark


预填充延迟 

预填充吞吐量（每秒处理的Token数）


解码总延迟、解码单Token延迟

解码吞吐量（每秒处理的Token数）





















