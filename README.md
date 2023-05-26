# llm-action

LLM 实战

## LLM训练

| LLM                           | 训练/微调/RLHF...                            | 参数  | 教程                                                                                                                                                                                      | 代码                                                                                |
| ----------------------------- | ----------------------------- | --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| Alpaca                        | full fine-turning             | 7B  | [从0到1复现斯坦福羊驼（Stanford Alpaca 7B）](https://zhuanlan.zhihu.com/p/618321077)                                                                                                                 |            N/A                                                                          |
| Alpaca                        | lora                          | 7B~65B  | [足够惊艳，使用Alpaca-Lora基于LLaMA(7B)二十分钟完成微调，效果比肩斯坦福羊驼](https://zhuanlan.zhihu.com/p/619426866)<br/>[使用 LoRA 技术对 LLaMA 65B 大模型进行微调及推理](https://zhuanlan.zhihu.com/p/632492604)                | [配套代码](https://github.com/liguodongiot/llm-action/tree/main/alpaca-lora)          |
| BELLE(LLaMA-7B/Bloomz-7B1-mt) | full fine-turning             | 7B  | [基于LLaMA-7B/Bloomz-7B1-mt复现开源中文对话大模型BELLE及GPTQ量化](https://zhuanlan.zhihu.com/p/618876472)<br/>[BELLE(LLaMA-7B/Bloomz-7B1-mt)大模型使用GPTQ量化后推理性能测试](https://zhuanlan.zhihu.com/p/621128368) | N/A                                                                               |
| ChatGLM                       | lora                          | 6B  | [从0到1基于ChatGLM-6B使用LoRA进行参数高效微调](https://zhuanlan.zhihu.com/p/621793987)                                                                                                                | N/A                                                                               |
| ChatGLM                       | full fine-turning/P-Tuning v2 | 6B  | [使用DeepSpeed/P-Tuning v2对ChatGLM-6B进行微调](https://zhuanlan.zhihu.com/p/622351059)                                                                                                        | N/A                                                                               |
| Vicuna                        | full fine-turning             | 7B  | [大模型也内卷，Vicuna训练及推理指南，效果碾压斯坦福羊驼](https://zhuanlan.zhihu.com/p/624012908)                                                                                                                | N/A                                                                               |
| OPT                           | RLHF                          | N/A | [一键式 RLHF 训练 DeepSpeed Chat（一）：理论篇](https://zhuanlan.zhihu.com/p/626159553) <br/>[一键式 RLHF 训练 DeepSpeed Chat（二）：实践篇](https://zhuanlan.zhihu.com/p/626214655)                            | N/A                                                                               |
| MiniGPT-4                     | full fine-turning             | 7B  | [大杀器，多模态大模型MiniGPT-4入坑指南](https://zhuanlan.zhihu.com/p/627671257)                                                                                                                       | N/A                                                                               |
| Chinese-LLaMA-Alpaca          | lora（预训练+微调）                  | 7B  | [使用 LoRA 技术对 LLaMA 65B 大模型进行微调及推理](https://zhuanlan.zhihu.com/p/631360711)                                                                                                              | [配套代码](https://github.com/liguodongiot/llm-action/tree/main/chinese-llama-alpaca) |

## LLM推理

- [大模型的好伙伴，浅析推理加速引擎FasterTransformer](https://zhuanlan.zhihu.com/p/626008090)
- [模型推理服务化框架Triton保姆式教程（一）：快速入门](https://zhuanlan.zhihu.com/p/629336492)
