# llm-action

本项目旨在分享大模型相关技术原理以及实战经验。


## 目录

- [:fire: LLM训练](#llm训练)
  - [:camel: LLM训练实战](#llm训练实战)
  - [:panda_face: LLM参数高效微调技术原理综述](#llm微调技术原理)
  - [:rabbit: LLM参数高效微调技术实战](#llm微调实战)
  - [:elephant: LLM分布式训练并行技术](#llm分布式训练并行技术)
  - [:volcano: 分布式AI框架](#分布式ai框架)
- [:racehorse: LLM推理](#llm推理)
  - [:rocket: 模型推理加速](#模型推理加速)
  - [:airplane: 模型推理服务化](#模型推理服务化)
  - [:triangular_ruler: LLM量化](#llm量化)
- [:jigsaw: LLM应用开发](#llm应用开发)
- [:mahjong: LLM国产化适配](#llm国产化适配)
- [:mushroom: LLM生态相关技术](#llm生态相关技术)
- [:hammer: 服务器基础环境软件安装](#服务器基础环境软件安装)
- [:speech_balloon: LLM学习交流群](#llm学习交流群)
- [:busts_in_silhouette: 微信公众号](#微信公众号)
- [:star: Star History](#star-history)


## LLM训练

### LLM训练实战

下面汇总了我在大模型实践中训练相关的所有教程。从6B到65B，从全量微调到高效微调（LoRA，QLoRA，P-Tuning v2），再到RLHF（基于人工反馈的强化学习）。

| LLM                           | 预训练/SFT/RLHF...                            | 参数  | 教程                                                                                                                                                                                      | 代码                                                                                |
| ----------------------------- | ----------------------------- | --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| Alpaca                        | full fine-turning             | 7B  | [从0到1复现斯坦福羊驼（Stanford Alpaca 7B）](https://zhuanlan.zhihu.com/p/618321077)                                                                                                                 |            [配套代码](https://github.com/liguodongiot/llm-action/tree/main/train/alpaca)                                                                          |
| Alpaca(LLaMA)                         | LoRA                          | 7B~65B  | 1. [足够惊艳，使用Alpaca-Lora基于LLaMA(7B)二十分钟完成微调，效果比肩斯坦福羊驼](https://zhuanlan.zhihu.com/p/619426866)<br/>2. [使用 LoRA 技术对 LLaMA 65B 大模型进行微调及推理](https://zhuanlan.zhihu.com/p/632492604)                | [配套代码](https://github.com/liguodongiot/llm-action/tree/main/train/alpaca-lora)          |
| BELLE(LLaMA/Bloom) | full fine-turning             | 7B  | 1. [基于LLaMA-7B/Bloomz-7B1-mt复现开源中文对话大模型BELLE及GPTQ量化](https://zhuanlan.zhihu.com/p/618876472)<br/>2. [BELLE(LLaMA-7B/Bloomz-7B1-mt)大模型使用GPTQ量化后推理性能测试](https://zhuanlan.zhihu.com/p/621128368) | N/A                                                                               |
| ChatGLM                       | LoRA                          | 6B  | [从0到1基于ChatGLM-6B使用LoRA进行参数高效微调](https://zhuanlan.zhihu.com/p/621793987)                                                                                                                | [配套代码](https://github.com/liguodongiot/llm-action/tree/main/train/chatglm-lora)                                                                               |
| ChatGLM                       | full fine-turning/P-Tuning v2 | 6B  | [使用DeepSpeed/P-Tuning v2对ChatGLM-6B进行微调](https://zhuanlan.zhihu.com/p/622351059)                                                                                                        | [配套代码](https://github.com/liguodongiot/llm-action/tree/main/train/chatglm)                                                                            |
| Vicuna(LLaMA)                         | full fine-turning             | 7B  | [大模型也内卷，Vicuna训练及推理指南，效果碾压斯坦福羊驼](https://zhuanlan.zhihu.com/p/624012908)                                                                                                                | N/A                                                                               |
| OPT                           | RLHF                          | 0.1B~66B | 1. [一键式 RLHF 训练 DeepSpeed Chat（一）：理论篇](https://zhuanlan.zhihu.com/p/626159553) <br/>2. [一键式 RLHF 训练 DeepSpeed Chat（二）：实践篇](https://zhuanlan.zhihu.com/p/626214655)                            | [配套代码](https://github.com/liguodongiot/llm-action/tree/main/train/deepspeedchat)                                                                              |
| MiniGPT-4(LLaMA)                      | full fine-turning             | 7B  | [大杀器，多模态大模型MiniGPT-4入坑指南](https://zhuanlan.zhihu.com/p/627671257)                                                                                                                       | N/A                                                                               |
| Chinese-LLaMA-Alpaca(LLaMA)          | LoRA（预训练+微调）                  | 7B  | [中文LLaMA&Alpaca大语言模型词表扩充+预训练+指令精调](https://zhuanlan.zhihu.com/p/631360711)                                                                                                              | [配套代码](https://github.com/liguodongiot/llm-action/tree/main/train/chinese-llama-alpaca) |
|  LLaMA         | QLoRA                  | 7B/65B  | [高效微调技术QLoRA实战，基于LLaMA-65B微调仅需48G显存，真香](https://zhuanlan.zhihu.com/p/636644164)                                                                                                              | [配套代码](https://github.com/liguodongiot/llm-action/tree/main/train/qlora) |


### LLM微调技术原理

对于普通大众来说，进行大模型的预训练或者全量微调遥不可及。由此，催生了各种参数高效微调技术，让科研人员或者普通开发者有机会尝试微调大模型。

因此，该技术值得我们进行深入分析其背后的机理，本系列大体分七篇文章进行讲解。

-    [大模型参数高效微调技术原理综述（一）-背景、参数高效微调简介](https://zhuanlan.zhihu.com/p/635152813)
-    [大模型参数高效微调技术原理综述（二）-BitFit、Prefix Tuning、Prompt Tuning](https://zhuanlan.zhihu.com/p/635686756)
-    [大模型参数高效微调技术原理综述（三）-P-Tuning、P-Tuning v2](https://zhuanlan.zhihu.com/p/635848732)
-    [大模型参数高效微调技术原理综述（四）-Adapter Tuning及其变体](https://zhuanlan.zhihu.com/p/636038478)
-    [大模型参数高效微调技术原理综述（五）-LoRA、AdaLoRA、QLoRA](https://zhuanlan.zhihu.com/p/636215898)
-    [大模型参数高效微调技术原理综述（六）-MAM Adapter、UniPELT](https://zhuanlan.zhihu.com/p/636362246)
-    [大模型参数高效微调技术原理综述（七）-最佳实践、总结](https://zhuanlan.zhihu.com/p/636999010)

### LLM微调实战

下面给大家分享**大模型参数高效微调技术实战**系列文章，该系列共6篇文章。

| 教程 | 代码 | 框架 |
| ----------------------------- | ----------------------------- |  ----------------------------- | 
| [大模型参数高效微调技术实战（一）-PEFT概述及环境搭建](https://juejin.cn/post/7257895211710627901) | N/A | HuggingFace PEFT | 
| [大模型参数高效微调技术实战（二）-Prompt Tuning](https://zhuanlan.zhihu.com/p/635686756) | [配套代码](https://github.com/liguodongiot/llm-action/blob/main/train/peft/clm/peft_prompt_tuning_clm.ipynb) |  HuggingFace PEFT | 
| [大模型参数高效微调技术实战（三）-P-Tuning]() | [配套代码](https://github.com/liguodongiot/llm-action/blob/main/train/peft/clm/peft_p_tuning_clm.ipynb) | HuggingFace PEFT | 
| [大模型参数高效微调技术实战（四）-Prefix Tuning / P-Tuning v2]() |  [配套代码](https://github.com/liguodongiot/llm-action/blob/main/train/peft/clm/peft_p_tuning_v2_clm.ipynb) |  HuggingFace PEFT | 
| [大模型参数高效微调技术实战（五）-LoRA]() |  [配套代码](https://github.com/liguodongiot/llm-action/blob/main/train/peft/clm/peft_lora_clm.ipynb) |  HuggingFace PEFT | 
| [大模型参数高效微调技术实战（六）-IA3]() |  [配套代码](https://github.com/liguodongiot/llm-action/blob/main/train/peft/clm/peft_ia3_clm.ipynb) |  HuggingFace PEFT | 





### LLM分布式训练并行技术

近年来，随着Transformer、MOE架构的提出，使得深度学习模型轻松突破上万亿规模参数，传统的单机单卡模式已经无法满足超大模型进行训练的要求。因此，我们需要基于单机多卡、甚至是多机多卡进行分布式大模型的训练。

而利用AI集群，使深度学习算法更好地从大量数据中高效地训练出性能优良的大模型是分布式机器学习的首要目标。为了实现该目标，一般需要根据硬件资源与数据/模型规模的匹配情况，考虑对计算任务、训练数据和模型进行划分，从而进行分布式训练。因此，分布式训练相关技术值得我们进行深入分析其背后的机理。

下面主要对大模型进行分布式训练的并行技术进行讲解，本系列大体分八篇文章进行讲解。

- [大模型分布式训练并行技术（一）-概述](https://juejin.cn/post/7195845066887053368)
- [大模型分布式训练并行技术（二）-数据并行](https://juejin.cn/post/7254001262646738981)
- [大模型分布式训练并行技术（三）-流水线并行]()
- 大模型分布式训练并行技术（四）-张量并行
- 大模型分布式训练并行技术（五）-序列并行
- 大模型分布式训练并行技术（六）-多维混合并行
- 大模型分布式训练并行技术（七）-自动并行
- 大模型分布式训练并行技术（八）-MOE并行


### 分布式AI框架

- Pytorch
  - Pytorch 单机多卡训练
  - Pytorch 多机多卡训练
- Megatron-LM
  - Megatron-LM 单机多卡训练
  - Megatron-LM 多机多卡训练
  - [基于Megatron-LM从0到1完成GPT2模型预训练、模型评估及推理](https://juejin.cn/post/7259682893648724029)
- DeepSpeed
  - DeepSpeed 单机多卡训练
  - DeepSpeed 多机多卡训练


## LLM推理

### 模型推理加速
- [大模型的好伙伴，浅析推理加速引擎FasterTransformer](https://zhuanlan.zhihu.com/p/626008090)

### 模型推理服务化
- [模型推理服务化框架Triton保姆式教程（一）：快速入门](https://zhuanlan.zhihu.com/p/629336492)
- [模型推理服务化框架Triton保姆式教程（二）：架构解析](https://zhuanlan.zhihu.com/p/634143650)
- [模型推理服务化框架Triton保姆式教程（三）：开发实践](https://zhuanlan.zhihu.com/p/634444666)


### LLM量化

待更新...


## LLM国产化适配
随着 ChatGPT 的现象级走红，引领了AI大模型时代的变革，从而导致 AI 算力日益紧缺。与此同时，中美贸易战以及美国对华进行AI芯片相关的制裁导致 AI 算力的国产化适配势在必行。本系列将对一些国产化 AI 加速卡进行讲解。

- [大模型国产化适配1-华为昇腾AI全栈软硬件平台总结](https://zhuanlan.zhihu.com/p/637918406)
- [大模型国产化适配2-基于昇腾910使用ChatGLM-6B进行模型推理](https://juejin.cn/post/7254446353367482423)
- [大模型国产化适配3-基于昇腾910使用ChatGLM-6B进行模型训练](https://juejin.cn/post/7256769427600064569)


## LLM应用开发
大模型是基座，要想让其变成一款产品，我们还需要一些其他相关的技术，比如：向量数据库（Pinecone、Milvus、Vespa、Weaviate），LangChain等。

- [云原生向量数据库Milvus（一）-简述、系统架构及应用场景](https://juejin.cn/post/7081440038772293663 )
- [云原生向量数据库Milvus（二）-数据与索引的处理流程、索引类型及Schema](https://juejin.cn/post/7081672823931928606)




## LLM生态相关技术

- [大模型词表扩充必备工具SentencePiece](https://zhuanlan.zhihu.com/p/630696264)
- [大模型实践总结](https://www.zhihu.com/question/601594836/answer/3032763174)
- [百川智能开源大模型baichuan-7B技术剖析](https://www.zhihu.com/question/606757218/answer/3075464500)
- [ChatGLM 和 ChatGPT 的技术区别在哪里？](https://www.zhihu.com/question/604393963/answer/3061358152)
- [现在为什么那么多人以清华大学的ChatGLM-6B为基座进行试验？](https://www.zhihu.com/question/602504880/answer/3041965998)


## 服务器基础环境软件安装

- [英伟达A800加速卡常见软件包安装命令](https://github.com/liguodongiot/llm-action/blob/main/docs/llm-base/a800-env-install.md)
- [英伟达H800加速卡常见软件包安装命令](https://github.com/liguodongiot/llm-action/blob/main/docs/llm-base/h800-env-install.md)
- [昇腾910加速卡常见软件包安装命令](https://github.com/liguodongiot/llm-action/blob/main/docs/llm_localization/ascend910-env-install.md)


## LLM学习交流群

我创建了大模型学习交流群，供大家一起学习交流大模型相关的最新技术，目前已有4个群，每个群都有上百人的规模，**可加我微信进群**（加微信请备注来意，如：进大模型学习交流群+GitHub）。**一定要备注哟，否则不予通过**。

<p align="center">
  <img src="https://github.com/liguodongiot/llm-action/blob/main/pic/%E5%BE%AE%E4%BF%A1.jpeg" width="25%" height="25%">
</p>





## 微信公众号

微信公众号：**吃果冻不吐果冻皮**，该公众号主要分享AI工程化（大模型、MLOps等）相关实践经验，免费电子书籍、论文等。

<p align="center">
  <img src="https://github.com/liguodongiot/llm-action/blob/main/pic/%E5%85%AC%E4%BC%97%E5%8F%B7.jpeg" width="30%" height="30%" div align=center>
</p>



## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=liguodongiot/llm-action&type=Date)](https://star-history.com/#liguodongiot/llm-action&Date)




