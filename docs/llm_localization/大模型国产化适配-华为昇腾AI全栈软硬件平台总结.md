随着 ChatGPT 的现象级走红，引领了AI大模型时代的变革，从而导致 AI 算力日益紧缺。与此同时，中美贸易战，导致AI算力国产化适配势在必行。本文主要对最近使用昇腾芯片做一个简单总结。


## 昇腾AI全栈软硬件平台简述

昇腾芯片是华为公司发布的两款 AI 处理器(NPU)，昇腾910（用于训练）和昇腾310（用于推理）处理器，采用自家的达芬奇架构。昇腾在国际上对标的主要是英伟达的GPU，国内对标的包括寒武纪、海光等厂商生产的系列AI芯片产品（如：思元590、深算一号等）。


整个昇腾软硬件全栈包括5层，自底向上为**Atlas系列硬件、异构计算架构、AI框架、应用使能、行业应用**。

![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/d018ad688e6a47a2a8e13a88a8d32bb8~tplv-k3u1fbpfcp-watermark.image?)



![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/2a47859dd75a4c2d87af963c5eae49d5~tplv-k3u1fbpfcp-watermark.image?)

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/189c11c826d042a89ee1dfd88c7e3c23~tplv-k3u1fbpfcp-watermark.image?)

### Atlas系列硬件

Atlas系列产品是基于昇腾910和昇腾310打造出来的、面向不同应用场景（端、边、云）的系列AI硬件产品。比如：
- **Atlas 800（型号：9000）** 是训练服务器，包含8个训练卡（**Atlas 300 T**：采用昇腾910）。
- **Atlas 900** 是训练集群（由128台**Atlas 800（型号：9000）**构成），相当于是由一批训练服务器组合而成。
- **Atlas 800（型号：3000）** 是推理服务器，包含8个推理卡（**Atlas 300 I**：采用昇腾310）。

![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/32888c111cac4c17a9da541d2d75b3ec~tplv-k3u1fbpfcp-watermark.image?)


### 异构计算架构

异构计算架构（CANN）是对标英伟达的CUDA + CuDNN的核心软件层，对上支持多种AI框架，对下服务AI处理器，发挥承上启下的关键作用，是提升昇腾AI处理器计算效率的关键平台，主要包括有各种引擎、编译器、执行器、算子库等。之所以叫异构软件，是因为承载计算的底层硬件包括AI芯片和通用芯片，自然就需要有一层软件来负责算子的调度、加速和执行，最后自动分配到对应的硬件上（CPU或NPU）。

![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/6788095c7819431aba6dea4eb85b6233~tplv-k3u1fbpfcp-watermark.image?)

- **昇腾计算语言**（Ascend Computing Language，AscendCL）接口是昇腾计算开放编程框架，对开发者屏蔽底层多种处理器差异，提供算子开发接口TBE、标准图开发接口AIR、应用开发接口，支持用户快速构建基于Ascend平台的AI应用和业务。
- **昇腾计算服务层**主要提供**昇腾算子库AOL**，通过神经网络（Neural Network，NN）库、线性代数计算库（Basic Linear Algebra Subprograms，BLAS）等高性能算子加速计算；**昇腾调优引擎AOE**，通过算子调优OPAT、子图调优SGAT、梯度调优GDAT、模型压缩AMCT提升模型端到端运行速度。同时提供**AI框架适配器Framework Adaptor**用于兼容Tensorflow、Pytorch等主流AI框架。
- **昇腾计算编译层**通过图编译器（Graph Compiler）将用户输入中间表达（Intermediate Representation，IR）的计算图编译成昇腾硬件可执行模型；同时借助张量加速引擎TBE（Tensor Boost Engine）的自动调度机制，高效编译算子。
- **昇腾计算执行层**负责模型和算子的执行，提供运行时库（Runtime）、图执行器（Graph Executor）、数字视觉预处理（Digital Vision Pre-Processing，DVPP）、人工智能预处理（Artificial Intelligence Pre-Processing，AIPP）、华为集合通信库（Huawei Collective Communication Library，HCCL）等功能单元。
- **昇腾计算基础层**主要为其上各层提供基础服务，如共享虚拟内存（Shared Virtual Memory，SVM）、设备虚拟化（Virtual Machine，VM）、主机-设备通信（Host Device Communication，HDC）等。

### AI框架

AI框架层主要包括**自研框架MindSpore（昇思）** 和**第三方框架（PyTorch、TensorFlow等）** ，其中MindSpore完全由华为自主研发，第三方框架华为只是做了适配和优化，让PyTorch和TensorFlow等框架编写的模型可以高效的跑在昇腾芯片上。

以PyTorch为例，华为的框架研发人员会将其做好适配，然后把适配后的PyTorch源码发布出来，想要在昇腾上用PyTorch的开发者，下载该源码自行编译安装即可。

![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/c6195f81e22940fe9597a0b4d5586240~tplv-k3u1fbpfcp-watermark.image?)


### 应用使能

应用使能层主要包括**ModelZoo、MindX SDK、MindX DL、MindX Edge**等。

- **ModelZoo：** 存放模型的仓库

![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/c50fe98850274524b21a0bbcff1fef9b~tplv-k3u1fbpfcp-watermark.image?)

- **MindX SDK：** 帮助特定领域的用户快速开发并部署人工智能应用，比如工业质检、检索聚类等，致力于简化昇腾 AI 处理器推理业务开发过程，降低使用昇腾AI处理器开发的门槛。


![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/7444199e44b84cfea66ba4e9324f1d33~tplv-k3u1fbpfcp-watermark.image?)


- **MindX DL（昇腾深度学习组件）：** 是支持 Atlas训练卡、推理卡的深度学习组件，提供昇腾 AI 处理器集群调度、昇腾 AI 处理器性能测试、模型保护等基础功能，快速使能合作伙伴进行深度学习平台开发。


![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/245bf318c3b343d58996c378efa82d81~tplv-k3u1fbpfcp-watermark.image?)


![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/988982604f244ff6b748b8a1bbc56b42~tplv-k3u1fbpfcp-watermark.image?)


![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/ea5f8eedab3d409b9df22fddb1ba4b5d~tplv-k3u1fbpfcp-watermark.image?)


![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/236bc27c3f8e4c55818cd0a0616facc0~tplv-k3u1fbpfcp-watermark.image?)


- **MindX Edge（昇腾智能边缘组件）：**  提供边缘 AI 业务容器的全生命周期管理能力，同时提供严格的安全可信保障，为客户提供边云协同的边缘计算解决方案，使能客户快速构建边缘 AI 业务。


![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/0ba533c6c3ed4a62a2eaf3dd38f5614a~tplv-k3u1fbpfcp-watermark.image?)

- **Modelarts：** ModelArts 是面向开发者的一站式 AI 平台，为机器学习与深度学习提供海量数据预处理及交互式智能标注、大规模分布式训练、自动化模型生成，及端-边-云模型按需部署能力，帮助用户快速创建和部署模型，管理全周期 AI 工作流。

![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/09e8210b12244705b076b6967a75d75d~tplv-k3u1fbpfcp-watermark.image?)


- **HiAI Service**：HUAWEI HiAI是面向智能终端的AI能力开放平台，基于 “芯、端、云”三层开放架构，即**芯片能力开放、应用能力开放、服务能力开放**，构筑全面开放的智慧生态，让开发者能够快速地利用华为强大的AI处理能力，为用户提供更好的智慧应用体验。

### 行业应用

主要应用于能源、金融、交通、电信、制造、医疗等行业，这里就不过多介绍了。


## 安装 MindSpore 和 MindFormers 简单流程

> 建议：确定要安装的MindSpore具体版本，再确定需要安装的驱动和固件版本。

主要有物理机、容器和虚拟机安装。其中，容器和虚拟机不支持固件包安装。


**安装流程**：

这里针对昇腾910处理器进行安装。

- 安装驱动和固件
```
chmod +x Ascend-hdk-910-npu-driver_23.0.rc1_linux-aarch64.run
./Ascend-hdk-910-npu-driver_23.0.rc1_linux-aarch64.run --full --install-path=/usr/local/Ascend


chmod +x Ascend-hdk-910-npu-firmware_6.3.0.1.241.run
./Ascend-hdk-910-npu-firmware_6.3.0.1.241.run --check
./Ascend-hdk-910-npu-firmware_6.3.0.1.241.run  --full
```
- 安装开发工具集（CANN）

```
chmod +x Ascend-cann-nnae_6.0.1_linux-aarch64.run
./Ascend-cann-nnae_6.0.1_linux-aarch64.run --install --install-for-all
# 建议配置在~/.bashrc中
source /usr/local/Ascend/nnae/set_env.sh 
```

- 创建虚拟环境

```
cd /tmp
curl -O https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py37_4.10.3-Linux-$(arch).sh
bash Miniconda3-py37_4.10.3-Linux-$(arch).sh -b
cd -
. ~/miniconda3/etc/profile.d/conda.sh
conda init bash

conda create -n mindspore_py37 python=3.7 -y
conda activate mindspore_py37

# python -m pip install -U pip
```

- 升级GCC以及安装Cmake

```
# 安装GCC
sudo yum install gcc -y


# 安装Cmake， aarch64使用
curl -O https://cmake.org/files/v3.19/cmake-3.19.8-Linux-aarch64.sh

sudo mkdir /usr/local/cmake-3.19.8
sudo bash cmake-3.19.8-Linux-*.sh --prefix=/usr/local/cmake-3.19.8 --exclude-subdir


echo -e "export PATH=/usr/local/cmake-3.19.8/bin:\$PATH" >> ~/.bashrc
source ~/.bashrc
```

- 安装通信库

```
# pip uninstall te topi hccl -y

pip install sympy
pip install /usr/local/Ascend/nnae/latest/lib64/topi-*-py3-none-any.whl
pip install /usr/local/Ascend/nnae/latest/lib64/te-*-py3-none-any.whl
pip install /usr/local/Ascend/nnae/latest/lib64/hccl-*-py3-none-any.whl
```

- 安装 MindSpore
```
conda install mindspore-ascend=1.10.1 -c mindspore -c conda-forge
```

- 安装 MindFormers

```
git clone https://gitee.com/mindspore/mindformers.git
cd mindformers/
pip install .
```
- 安装其他库


```
pip install requests
```


## GPT2 模型推理

下面针对 GPT2 进行模型推理测试：
```
from mindformers.pipeline import pipeline
pipeline_task = pipeline("text_generation", model='gpt2', max_length=20)
pipeline_result = pipeline_task("I love Beijing, because", top_k=3)
print(pipeline_result)
```

运行过程：

```
>>> from mindformers.pipeline import pipeline
>>> pipeline_task = pipeline("text_generation", model='gpt2', max_length=20)
[WARNING] ME(55115:281472858920832,MainProcess):2023-06-16-16:26:33.147.258 [mindspore/ops/primitive.py:207] The in_strategy of the operator in your network will not take effect in stand_alone mode. This means the the shard function called in the network is ignored. 
If you want to enable it, please use semi auto or auto parallel mode by context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL or context.set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL)
[WARNING] ME(55115:281472858920832,MainProcess):2023-06-16-16:26:33.311.599 [mindspore/common/_decorator.py:38] 'DropoutGenMask' is deprecated from version 1.5 and will be removed in a future version, use 'ops.Dropout' instead.
[WARNING] ME(55115:281472858920832,MainProcess):2023-06-16-16:26:33.312.329 [mindspore/common/_decorator.py:38] 'DropoutDoMask' is deprecated from version 1.5 and will be removed in a future version, use 'ops.Dropout' instead.
[WARNING] ME(55115:281472858920832,MainProcess):2023-06-16-16:26:33.313.109 [mindspore/common/parameter.py:599] This interface may be deleted in the future.
2023-06-16 16:26:34,473 - mindformers - INFO - Start download ./checkpoint_download/gpt2/gpt2.ckpt
Downloading: 498MB [00:13, 35.7MB/s]                                                                
2023-06-16 16:26:48,825 - mindformers - INFO - Download completed!,times: 14.68s
2023-06-16 16:26:48,829 - mindformers - INFO - start to read the ckpt file: 497772028
[WARNING] ME(55115:281472858920832,MainProcess):2023-06-16-16:26:50.316.896 [mindspore/train/serialization.py:736] For 'load_param_into_net', remove parameter prefix name: backbone., continue to load.
[WARNING] ME(55115:281472858920832,MainProcess):2023-06-16-16:26:50.325.714 [mindspore/train/serialization.py:736] For 'load_param_into_net', remove parameter prefix name: backbone.blocks., continue to load.
2023-06-16 16:26:50,391 - mindformers - INFO - weights in ./checkpoint_download/gpt2/gpt2.ckpt are loaded
2023-06-16 16:26:50,392 - mindformers - INFO - Download the yaml from the url https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/gpt2/vocab.json to ./checkpoint_download/gpt2/vocab.json.
2023-06-16 16:26:50,564 - mindformers - INFO - Start download ./checkpoint_download/gpt2/vocab.json
Downloading: 1.04MB [00:00, 5.61MB/s]                                                               
2023-06-16 16:26:50,751 - mindformers - INFO - Download completed!,times: 0.36s
2023-06-16 16:26:50,752 - mindformers - INFO - Download the yaml from the url https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/gpt2/merges.txt to ./checkpoint_download/gpt2/merges.txt.
2023-06-16 16:26:50,914 - mindformers - INFO - Start download ./checkpoint_download/gpt2/merges.txt
Downloading: 457kB [00:00, 3.59MB/s]                                                                
2023-06-16 16:26:51,042 - mindformers - INFO - Download completed!,times: 0.29s
>>> pipeline_result = pipeline_task("I love Beijing, because", top_k=3)
>>> print(pipeline_result)
[{'text_generation_text': ['I love Beijing, because it\'s so beautiful," he told the crowd, adding, "We\'re']}]
```

模型权重文件：
```
> tree -h gpt2
gpt2
├── [ 475M]  gpt2.ckpt
├── [ 446K]  merges.txt
└── [1018K]  vocab.json
```


NPU显存使用情况：

```
> npu-smi info
+-------------------------------------------------------------------------------------------+
| npu-smi 23.0.rc1                 Version: 23.0.rc1                                        |
+----------------------+---------------+----------------------------------------------------+
| NPU   Name           | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                 | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+======================+===============+====================================================+
| 0     910ProB        | OK            | 78.4        33                0    / 0             |
| 0                    | 0000:C1:00.0  | 0           1198 / 15137      30711/ 32768         |
+======================+===============+====================================================+

+----------------------+---------------+----------------------------------------------------+
| NPU     Chip         | Process id    | Process name             | Process memory(MB)      |
+======================+===============+====================================================+
| 0       0            | 55115         | python                   | 30784                   |
+======================+===============+====================================================+
```

MindFormers 目前适配的一些模型：
```
{
	'swin_base_p4w7',
	'clip_vit_b_32',
	'clip_vit_b_16',
	'clip_vit_l_14',
	'gpt2',
	'llama_13b',
	'bloom_560m',
	't5_small',
	'glm_6b_lora',
	'bert_tiny_uncased',
	'llama_7b',
	'bloom_65b',
	't5_tiny',
	'gpt2_13b',
	'mindspore/txtcls_bert_base_uncased_mnli',
	'glm_6b_chat',
	'common',
	'gpt2_52b',
	'tokcls_bert_base_chinese',
	'mindspore/vit_base_p16',
	'qa_bert_base_uncased_squad',
	'pangualpha_13b',
	'pangualpha_2_6b',
	'mae_vit_base_p16',
	'clip_vit_l_14@336',
	'bloom_7.1b',
	'llama_65b',
	'glm_6b_lora_chat',
	'bloom_176b',
	'glm_6b',
	'mindspore/clip_vit_b_32',
	'qa_bert_base_uncased',
	'tokcls_bert_base_chinese_cluener',
	'vit_base_p16',
	'txtcls_bert_base_uncased_mnli',
	'llama_7b_lora',
	'mindspore/swin_base_p4w7',
	'bert_base_uncased',
	'mindspore/qa_bert_base_uncased',
	'txtcls_bert_base_uncased'
}
```

可以看到，昇腾针对ChatGLM、OpenLLaMA等开源大模型进行了一些适配，不过不太稳定，运行还有一些问题，还在摸索中。而且，像LLaMA之类的大模型，目前多卡模型并行推理是不支持的。


## 注意事项

- 昇腾推理服务器不支持训练，模型训练需要使用训练类型服务器。
- Ascend 310只能用作推理，MindSpore支持在Ascend 910训练，训练出的模型要转化为OM模型用于Ascend 310上进行推理。
- 目前华为昇腾主要还是运行华为自家闭环的大模型产品。
- 任何公开模型都必须经过华为的深度优化才能在华为的平台上运行。而这部分优化工作严重依赖于华为，进度较慢。
- 目前（2023.06.19）昇腾芯片对于Pytorch/TensorFlow等第三方框架的支持很差，如果需要适配最新出来的大模型，使用起来困难重重。
- 目前（2023.06.19）MindFormers 里面适配的大模型（如：ChatGLM、LLaMA、Bloom）仅支持Ascend 910进行模型训练和推理，不支持Ascend 310进行推理。
- 目前（2023.06.19）针对 ChatGLM、LLaMA、Bloom 等大模型不支持int8量化。
- 使用昇腾 NPU 芯片进行大模型推理时是独占的，不支持同一张 NPU 卡运行多个大模型。


## 结语

说实话，刚开始看昇腾的时候，还是有点懵逼的。虽然相关的文档很多，但感觉稍显凌乱。遇到问题的时候，社区使用的人少，很可能得不到相应的解决，所以，经常有一种原地爆炸的心情。目前来看，昇腾的生态相比Nvidia的生态还是差很远。但是，在国内来说，也是做的比较全的了，希望昇腾越来越好吧，保持耐心。


参考文档：
- [npu-smi 命令参考文档](https://support.huawei.com/enterprise/zh/doc/EDOC1100273886/d8e258d6)
- [mindspore 安装（Gitee）](https://gitee.com/mindspore/docs/blob/r1.10/install/mindspore_ascend310_install_pip.md)
- [mindspore 安装（官网）](https://www.mindspore.cn/install)
- [CANN 安装](https://www.hiascend.com/document/detail/zh/canncommercial/601/overview/index.html)
- [mindspore 官方教程](https://www.mindspore.cn/tutorials/zh-CN/r1.10/index.html)






