# llm-action

LLM 实战


## 目录

- [LLM训练](#LLM训练)
- [LLM推理](#LLM推理)
- [LLM微调技术原理](#LLM微调技术原理)
- [LLM学习交流群](#LLM学习交流群)
- [微信公众号](#微信公众号)


## LLM训练

| LLM                           | 预训练/SFT/RLHF...                            | 参数  | 教程                                                                                                                                                                                      | 代码                                                                                |
| ----------------------------- | ----------------------------- | --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| Alpaca                        | full fine-turning             | 7B  | [从0到1复现斯坦福羊驼（Stanford Alpaca 7B）](https://zhuanlan.zhihu.com/p/618321077)                                                                                                                 |            N/A                                                                          |
| Alpaca                        | lora                          | 7B~65B  | 1. [足够惊艳，使用Alpaca-Lora基于LLaMA(7B)二十分钟完成微调，效果比肩斯坦福羊驼](https://zhuanlan.zhihu.com/p/619426866)<br/>2. [使用 LoRA 技术对 LLaMA 65B 大模型进行微调及推理](https://zhuanlan.zhihu.com/p/632492604)                | [配套代码](https://github.com/liguodongiot/llm-action/tree/main/alpaca-lora)          |
| BELLE(LLaMA-7B/Bloomz-7B1-mt) | full fine-turning             | 7B  | 1. [基于LLaMA-7B/Bloomz-7B1-mt复现开源中文对话大模型BELLE及GPTQ量化](https://zhuanlan.zhihu.com/p/618876472)<br/>2. [BELLE(LLaMA-7B/Bloomz-7B1-mt)大模型使用GPTQ量化后推理性能测试](https://zhuanlan.zhihu.com/p/621128368) | N/A                                                                               |
| ChatGLM                       | lora                          | 6B  | [从0到1基于ChatGLM-6B使用LoRA进行参数高效微调](https://zhuanlan.zhihu.com/p/621793987)                                                                                                                | [配套代码](https://github.com/liguodongiot/llm-action/tree/main/chatglm-lora)                                                                               |
| ChatGLM                       | full fine-turning/P-Tuning v2 | 6B  | [使用DeepSpeed/P-Tuning v2对ChatGLM-6B进行微调](https://zhuanlan.zhihu.com/p/622351059)                                                                                                        | [配套代码](https://github.com/liguodongiot/llm-action/tree/main/chatglm)                                                                            |
| Vicuna                        | full fine-turning             | 7B  | [大模型也内卷，Vicuna训练及推理指南，效果碾压斯坦福羊驼](https://zhuanlan.zhihu.com/p/624012908)                                                                                                                | N/A                                                                               |
| OPT                           | RLHF                          | N/A | 1. [一键式 RLHF 训练 DeepSpeed Chat（一）：理论篇](https://zhuanlan.zhihu.com/p/626159553) <br/>2. [一键式 RLHF 训练 DeepSpeed Chat（二）：实践篇](https://zhuanlan.zhihu.com/p/626214655)                            | N/A                                                                               |
| MiniGPT-4                     | full fine-turning             | 7B  | [大杀器，多模态大模型MiniGPT-4入坑指南](https://zhuanlan.zhihu.com/p/627671257)                                                                                                                       | N/A                                                                               |
| Chinese-LLaMA-Alpaca          | lora（预训练+微调）                  | 7B  | [使用 LoRA 技术对 LLaMA 65B 大模型进行微调及推理](https://zhuanlan.zhihu.com/p/631360711)                                                                                                              | [配套代码](https://github.com/liguodongiot/llm-action/tree/main/chinese-llama-alpaca) |
| QLoRA          | qlora                  | 7B/65B  | [高效微调技术QLoRA实战，基于LLaMA-65B微调仅需48G显存，真香](https://zhuanlan.zhihu.com/p/636644164)                                                                                                              | [配套代码](https://github.com/liguodongiot/llm-action/tree/main/qlora) |

## LLM推理

- [大模型的好伙伴，浅析推理加速引擎FasterTransformer](https://zhuanlan.zhihu.com/p/626008090)
- [模型推理服务化框架Triton保姆式教程（一）：快速入门](https://zhuanlan.zhihu.com/p/629336492)
- [模型推理服务化框架Triton保姆式教程（二）：架构解析](https://zhuanlan.zhihu.com/p/634143650)
- [模型推理服务化框架Triton保姆式教程（三）：开发实践](https://zhuanlan.zhihu.com/p/634444666)


## LLM微调技术原理

对于普通大众来说，进行大模型的预训练或者全量微调遥不可及。由此，催生了各种参数高效微调技术，让科研人员或者普通开发者有机会尝试微调大模型。

因此，该技术值得我们进行深入分析其背后的机理，本系列大体分七篇文章进行讲解。

-    [大模型参数高效微调技术原理综述（一）-背景、参数高效微调简介](https://zhuanlan.zhihu.com/p/635152813)
-    [大模型参数高效微调技术原理综述（二）-BitFit、Prefix Tuning、Prompt Tuning](https://zhuanlan.zhihu.com/p/635686756)
-    [大模型参数高效微调技术原理综述（三）-P-Tuning、P-Tuning v2](https://zhuanlan.zhihu.com/p/635848732)
-    [大模型参数高效微调技术原理综述（四）-Adapter Tuning及其变体](https://zhuanlan.zhihu.com/p/636038478)
-    [大模型参数高效微调技术原理综述（五）-LoRA、AdaLoRA、QLoRA](https://zhuanlan.zhihu.com/p/636215898)
-    [大模型参数高效微调技术原理综述（六）-MAM Adapter、UniPELT](https://zhuanlan.zhihu.com/p/636362246)
-    大模型参数高效微调技术原理综述（七）-最佳实践、总结



## LLM学习交流群

我创建了一个大模型学习交流群，供大家一起学习交流大模型相关的最新技术，可加我微信进群（**加微信请备注来意**，如：进大模型学习交流群+GitHub）。
<p align="center">
  <img src="./pic/微信.jpeg" width="25%" height="25%">
</p>


## 微信公众号

微信公众号：**吃果冻不吐果冻皮**，该公众号主要分享AI工程化（大模型、MLOps等）相关实践经验，免费电子书籍、论文等。

<p align="center">
  <img src="./pic/公众号.jpeg" width="30%" height="30%" div align=center>
</p>









