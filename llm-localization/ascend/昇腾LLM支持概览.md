


## 基础环境


------

800T A2

https://www.hiascend.com/hardware/firmware-drivers/community

Ascend-hdk-910b-npu-firmware_7.1.0.4.220.run  
Ascend-hdk-910b-npu-driver_23.0.1_linux-aarch64.run


https://www.hiascend.com/zh/software/cann/community-history

Ascend-cann-toolkit_7.0.1_linux-aarch64.run

------

800T (9000)

Ascend-hdk-910-npu-driver_23.0.0_linux-aarch64.run
Ascend-hdk-910-npu-firmware_7.1.0.3.220.run

Ascend-cann-toolkit_7.0.0_linux-aarch64.run






## MindSpore


### 镜像

- http://mirrors.cn-central-221.ovaijisuan.com/mirrors.html


docker pull swr.cn-central-221.ovaijisuan.com/mindformers/mindformers1.0_mindspore2.2.11:aarch_20240125


```
# --device用于控制指定容器的运行NPU卡号和范围
# -v 用于映射容器外的目录
# --name 用于自定义容器名称

docker run -it -u root \
--ipc=host \
--network host \
--device=/dev/davinci0 \
--device=/dev/davinci1 \
--device=/dev/davinci2 \
--device=/dev/davinci3 \
--device=/dev/davinci4 \
--device=/dev/davinci5 \
--device=/dev/davinci6 \
--device=/dev/davinci7 \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-v /etc/localtime:/etc/localtime \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /var/log/npu/:/usr/slog \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
--name {请手动输入容器名称} \
swr.cn-central-221.ovaijisuan.com/mindformers/mindformers1.0_mindspore2.2.11:aarch_20240125 \
/bin/bash



docker run -it  -u root  \
--device=/dev/davinci0   \
--device=/dev/davinci1   \
--device=/dev/davinci2   \
--device=/dev/davinci3   \
--device=/dev/davinci4   \
--device=/dev/davinci5   \
--device=/dev/davinci6   \
--device=/dev/davinci7   \
--device=/dev/davinci_manager   \
--device=/dev/devmm_svm   \
--device=/dev/hisi_hdc   \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver    \
-v /usr/local/dcmi:/usr/local/dcmi   \
-v /usr/local/Ascend/toolbox:/usr/local/Ascend/toolbox    \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi   \
-v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware   \ 
-v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi   \
-v /home/aicc:/home/ma-user/work/aicc    \
--name mindspore_ma   \
--entrypoint=/bin/bash  \
swr.cn-central-221.ovaijisuan.com/dxy/mindspore_kernels:MindSpore2.2.10-cann7.0.0beta1_py_3.9-euler_2.8.3-32GB \
/bin/bash

```



### MindFormers

#### 软件版本

当前支持的硬件为Atlas 800训练服务器与Atlas 800T A2训练服务器。

当前套件建议使用的Python版本为3.9。

| MindFormers | MindPet |                 MindSpore                  |                                                                                                                                               CANN                                                                                                                                               |                               驱动固件                               |                               镜像链接                               | 备注                 |
| :---------: | :-----: | :----------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------: | :------------------------------------------------------------------: | -------------------- |
|     dev     |  1.0.3  | [2.2.11](https://www.mindspore.cn/install) |           7.0.0.beta1:<br> [aarch64](https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%207.0.0/Ascend-cann-toolkit_7.0.0_linux-aarch64.run)<br> [x86_64](https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%207.0.0/Ascend-cann-toolkit_7.0.0_linux-x86_64.run)           | [链接](https://www.hiascend.com/hardware/firmware-drivers/community) |                                  /                                   | 开发分支(非稳定版本) |
|    r1.0     |  1.0.3  | [2.2.11](https://www.mindspore.cn/install) |           7.0.0.beta1:<br> [aarch64](https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%207.0.0/Ascend-cann-toolkit_7.0.0_linux-aarch64.run)<br> [x86_64](https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%207.0.0/Ascend-cann-toolkit_7.0.0_linux-x86_64.run)           | [链接](https://www.hiascend.com/hardware/firmware-drivers/community) | [链接](http://mirrors.cn-central-221.ovaijisuan.com/detail/118.html) | 发布版本             |
|    r0.8     |  1.0.2  | [2.2.1](https://www.mindspore.cn/install)  | 7.0.RC1.3.beta1:<br> [aarch64](https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%207.0.RC1.3/Ascend-cann-toolkit_7.0.RC1.3_linux-aarch64.run)<br> [x86_64](https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%207.0.RC1.3/Ascend-cann-toolkit_7.0.RC1.3_linux-x86_64.run) | [链接](https://www.hiascend.com/hardware/firmware-drivers/community) |                                  /                                   | 历史发布版本                    |

其中CANN，固件驱动的安装需与使用的机器匹配，请注意识别机器型号，选择对应架构的版本



#### baichuan

- https://gitee.com/mindspore/mindformers/blob/dev/research/baichuan/baichuan.md
- Baichuan-7B/13B
- 全参微调、Lora微调
- MindSpore、MindSpore lite

#### baichuan2

- https://gitee.com/mindspore/mindformers/tree/dev/research/baichuan2
- Baichuan-7B/13B
- 全参微调、Lora微调
- MindSpore、MindSpore lite


#### qwen

- https://gitee.com/mindspore/mindformers/blob/dev/research/qwen/qwen.md
- Qwen-7B/14B
- 全参微调、Lora微调
- MindSpore、MindSpore lite

#### bloom

- https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/bloom.md
- Bloom-7B1
- 全参微调
- MindSpore、MindSpore lite

#### chatglm

- https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/glm3.md
- Chatglm-6b
- 全参微调、Lora微调
- MindSpore、MindSpore lite

#### chatglm2

- https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/glm2.md
- Chatglm2-6b
- 全参微调、Lora微调、P-Tuning微调
- MindSpore、MindSpore lite

#### chatglm3


- https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/glm3.md
- Chatglm3-6b
- 全参微调、Lora微调
- MindSpore


#### t5

T5:
- https://gitee.com/mindspore/mindformers/blob/dev/mindformers/models/t5/mt5.py#L200
- MT5ForConditionalGeneration

bert: 
- https://gitee.com/mindspore/mindformers/blob/dev/mindformers/models/bert/bert_tokenizer.py



transformers:

T5:
- MT5ForConditionalGeneration
- https://github.com/huggingface/transformers/blob/main/src/transformers/models/mt5/modeling_mt5.py


T5-Pegasus:

Tokenizer

- https://github.com/renmada/t5-pegasus-pytorch/blob/main/tokenizer.py#L389
```
class JieBaTokenizer(CPTTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_tokenizer = partial(jieba.cut, HMM=False)

    def _tokenize(self, text, *arg, **kwargs):
        split_tokens = []
        for text in self.pre_tokenizer(text):
            if text in self.vocab:
                split_tokens.append(text)
            else:
                split_tokens.extend(super()._tokenize(text))
        return split_tokens
```
- t5-pegasus的huggingface调用: https://zhuanlan.zhihu.com/p/648997663
- 中文生成模型T5-Pegasus详解与实践：https://blog.csdn.net/GJ_0418/article/details/123298099

T5-Pegasus的Tokenizer换为了BERT的Tokenizer，并与jieba分词相结合，实现分词功能。具体地，先用jieba分词，如果当前词在词表vocab.txt中，就用jieba分词的结果；如果当前词不在词表vocab.txt中，再改用BERT的Tokenizer。




## PyTorch

### 镜像

```
docker run -it  -u root  \
--device=/dev/davinci0  \
--device=/dev/davinci1  \
--device=/dev/davinci2  \
--device=/dev/davinci3  \
--device=/dev/davinci4  \
--device=/dev/davinci5  \
--device=/dev/davinci6  \
--device=/dev/davinci7  \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm   \
--device=/dev/hisi_hdc    \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver    \
-v /usr/local/dcmi:/usr/local/dcmi   \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi   \
-v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware  \
-v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi   \
-v /home/aicc:/home/ma-user/work/aicc    \
--name pytorch_ma   \
--entrypoint=/bin/bash   \
swr.cn-central-221.ovaijisuan.com/dxy/pytorch2_1_0_kernels:PyTorch2.1.0-cann7.0.0.alpha003_py_3.9-euler_2.8.3-64GB

```


### ModelLink

目前只支持了类GPT仅Decoder架构的生成模型、像Bert、T5、ChatGLM等模型暂时没有支持。

#### 支持功能

当前ModelLink支撑大模型使用功能:

- 制作预训练数据集/制作指令微调数据集
- 预训练/全参微调/低参微调
- 推理(人机对话)
- 评估基线数据集(Benchmark)
- 使用加速特性（加速算法+融合算子）
- 基于昇腾芯片采集Profiling数据


#### 软件版本

|           软件            | [版本](https://www.hiascend.com/zh/) |
| :-----------------------: |:----------------------------------:|
|          Python           |                3.8                 |
|          driver           |              Ascend HDK 23.0.0              |
|         firmware          |              Ascend HDK 23.0.0              |
|           CANN            |              CANN 7.0.0              |
|           torch           |               2.1.0                |
|         torch_npu         |              release v5.0.0               |


torch_npu:

torch_npu(Ascend Adapter for PyTorch插件)使昇腾NPU可以适配PyTorch框架，为使用PyTorch框架的开发者提供昇腾AI处理器的超强算力。

https://gitee.com/ascend/pytorch




#### baichuan

- https://gitee.com/ascend/ModelLink/blob/master/examples/baichuan
- Baichuan-7B/13B
- 全参微调
- ModelLink

#### baichuan2

- https://gitee.com/ascend/ModelLink/tree/master/examples/baichuan2
- Baichuan2-7B/13B
- 全参微调
- ModelLink

#### bloom

- https://gitee.com/ascend/ModelLink/tree/master/examples/bloom
- Bloom-7B/176B
- 全参微调
- ModelLink

#### qwen

- https://gitee.com/ascend/ModelLink/tree/master/examples/qwen
- Qwen-7B/14B/72B
- 全参微调
- ModelLink


#### llama2

- https://gitee.com/ascend/ModelLink/tree/master/examples/llama2
- LLAMA2-7B/13B/34B/70B
- 全参微调、Lora微调
- ModelLink
- https://gitee.com/ascend/ModelLink/tree/master/modellink/tasks/inference/text_generation
