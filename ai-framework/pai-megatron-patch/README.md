- https://github.com/alibaba/Pai-Megatron-Patch/

```bash
git clone --recurse-submodules https://github.com/alibaba/Pai-Megatron-Patch.git
```

# 环境准备

## 数据准备

```bash
mkdir /mnt/workspace/llama2-datasets/
cd /mnt/workspace/llama2-datasets/
wget https://atp-modelzoo.oss-cn-hangzhou.aliyuncs.com/release/datasets/WuDaoCorpus2.0_base_sample.tgz
tar zxvf WuDaoCorpus2.0_base_sample.tgz 
```

```bash
cd /mnt/workspace/llama2-datasets
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama2-datasets/wudao_train.json
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama2-datasets/wudao_valid.json
mkdir -p /mnt/workspace/llama2-datasets/wudao
cd /mnt/workspace/llama2-datasets/wudao
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama2-datasets/wudao_llamabpe_text_document.bin
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama2-datasets/wudao_llamabpe_text_document.idx
```

## Huggingface&DeepSpeed训练流程

## Megatron训练流程

1. 将Huggingface格式的模型文件转换为Megatron格式。
2. 继续预训练模型。
3. 有监督微调模型。

模型训练完成后，您可以离线推理模型来评估模型效果。根据上面不同的训练流程，PAI也提供了对应的HuggingFace&DeepSpeed和MegatronLM两种格式的推理链路。

---

以Megatron训练流程获得的模型为例

将原始模型目录文件替换成训练获得的模型文件

```bash
cp /mnt/workspace/output_megatron_llama2/checkpoint/dswXXX/iter_XXX/mp_rank_00/model_rng.pt \
/mnt/workspace/llama2-ckpts/llama2-7b-hf-to-megatron-tp1-pp1/release/mp_rank_00/model_rng.pt
```

如果需要部署使用Huggingface&DeepSpeed训练流程获得的模型文件，您需要将 `/mnt/workspace/output_llama2/checkpoint/dswXXX/checkpointXXX/global_stepXXX/mp_rank_00_model_states.pt`文件拷贝到上一层包含config.json文件的目录中，并重命名为pytorch_model.bin。然后参照上述操作步骤将 `/mnt/workspace/output_llama2/checkpoint/dswXXX/checkpointXXX/`目录下的文件上传到OSS Bucket存储空间。

部署模型服务

BladeLLM是阿里云PAI平台提供的大模型部署框架，支持主流LLM模型结构，并内置模型量化压缩、BladeDISC编译等优化技术用于加速模型推理。 使用BladeLLM的预构建镜像，能够便捷地在PAI-EAS平台部署大模型推理服务。本方案以使用Megatron训练流程获得的模型为例，来说明如何部署模型服务，具体操作步骤如下：

```bash

#!/usr/bin/env python

import json

from websockets.sync.client import connect

def hello():
    headers = {"Authorization": "<token>"}
    # URL 也从EAS控制台 - 查看调用信息处获取，把 http:// 换成 ws:// 即可。
    url = "ws://xxxxx.cn-wulanchabu.pai-eas.aliyuncs.com/api/predict/<service_name>"
    with connect(url, additional_headers=headers) as websocket:
        prompts = ["What's the capital of Canada?"]
        for p in prompts:
            print(f"Prompt : {p}")
            websocket.send(json.dumps({"prompt": p}))
            while True:
                msg = websocket.recv()
                msg = json.loads(msg)
                if msg['is_end']:
                    break
                print(msg['text'], end="", flush=True)
            print()
            print("-" * 40)

hello()
```


# 模型训练



## Huggingface&DeepSpeed训练流程


```
ENV=$1                             # 运行环境配置：dsw（单机）,dlc（分布式）
MODEL_SIZE=$2                      # 模型结构参数量级: 7B,13B
BATCH_SIZE=$3                      # 每卡训练一次迭代样本数: 4, 8
GA_STEPS=$4                        # 梯度累积step数
LR=$5                              # 学习率: 1e-5, 5e-5
SEQ_LEN=$6                         # 序列长度: 2048
PR=$7                              # 训练精度: fp16, bf16
ZERO=$8                            # DeepSpeed ZERO降显存: 1,2,3
GC=$9                              # 是否使用gradient-checkpointing: true, false
TRAIN_DATASET_PATH=${10}           # 训练集路径, 支持单一文件或者文件夹形式输入
VALID_DATASET_PATH=${11}           # 验证集路径, 支持单一文件或者文件夹形式输入
PRETRAIN_CHECKPOINT_PATH=${12}     # 预训练模型路径
EPOCH=${13}                        # 训练epoch数
OUTPUT_BASEPATH=${14}              # 训练输出文件路径
```

```
cd /mnt/workspace
mkdir test_llama2_hf
cd test_llama2_hf
export WORK_DIR=/mnt/workspace/
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama2-codes/ds_config_TEMPLATE.json
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama2-codes/ds_train_huggingface_llama.py
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama2-codes/run_ds_train_huggingface_llama.sh
bash run_ds_train_huggingface_llama.sh \
dsw \
7B \
1 \
2 \
1e-5 \
2048 \
bf16 \
2 \
true \
${WORK_DIR}/llama2-datasets/wudao_train.json \
${WORK_DIR}/llama2-datasets/wudao_valid.json \
${WORK_DIR}/llama2-ckpts/Llama-2-7b-hf \
2 \
${WORK_DIR}/output_llama2
```


## Megatron训练流程


```
cd /mnt/workspace/
mkdir llama2-ckpts
cd llama2-ckpts
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama2-ckpts/Llama-2-7b-hf.tgz
tar -zxf Llama-2-7b-hf.tgz
mv Llama-2-7b-hf llama2-7b-hf

cd /mnt/workspace/PAI-Megatron-Patch/toolkits/model_checkpoints_convertor/llama
sh model_convertor.sh \
/root/Megatron-LM-230512        \
/mnt/workspace/llama2-ckpts/llama2-7b-hf         \
/mnt/workspace/llama2-ckpts/llama2-7b-hf-to-megatron-tp1-pp1  \
1  \
1  \
llama-7b \
0 \
false
```


```
cd /mnt/workspace/
mkdir llama2-ckpts
cd llama2-ckpts
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama2-ckpts/llama-2-7b-hf-to-megatron-tp1-pp1.tgz
tar -zxf llama-2-7b-hf-to-megatron-tp1-pp1.tgz
```

---

继续预训练

```
ENV=$1                          # 运行环境: dlc, dsw
MEGATRON_PATH=$2                # 设置开源Megatron的代码路径
MEGATRON_PATCH_PATH=$3          # 设置Megatron Patch的代码路径
MODEL_SIZE=$4                   # 模型结构参数量级：7B, 13B
BATCH_SIZE=$5                   # 每卡训练一次迭代样本数: 4, 8
GLOBAL_BATCH_SIZE=$6            # 全局batch size
LR=$7                           # 学习率: 1e-5, 5e-5
MIN_LR=$8                       # 最小学习率: 1e-6, 5e-6
SEQ_LEN=$9                      # 序列长度
PAD_LEN=${10}                   # Padding长度：100
EXTRA_VOCAB_SIZE=${11}          # 词表扩充大小
PR=${12}                        # 训练精度: fp16, bf16
TP=${13}                        # 模型并行度
PP=${14}                        # 流水并行度
AC=${15}                        # 激活检查点模式: sel, full
DO=${16}                        # 是否使用Megatron版Zero-1降显存优化器: true, false
FL=${17}                        # 是否使用Flash Attention: true, false
SP=${18}                        # 是否使用序列并行: true, false
SAVE_INTERVAL=${19}             # 保存ckpt的间隔
DATASET_PATH=${20}              # 训练数据集路径
PRETRAIN_CHECKPOINT_PATH=${21}  # 预训练模型路径
TRAIN_TOKENS=${22}              # 训练token数
WARMUP_TOKENS=${23}             # 预热token数
OUTPUT_BASEPATH=${24}           # 训练输出文件路径
```

```
export WORK_DIR=/mnt/workspace
cd ${WORK_DIR}/PAI-Megatron-Patch/examples/llama2
sh run_pretrain_megatron_llama.sh  \
dsw  \
/root/Megatron-LM-230512   \
${WORK_DIR}/PAI-Megatron-Patch  \
7B   \
1    \
8 \
1e-5   \
1e-6   \
2048  \
80  \
0   \
fp16  \
1   \
1  \
sel  \
true   \
false  \
false   \
100000  \
${WORK_DIR}/llama2-datasets/wudao/wudao_llamabpe_text_document   \
${WORK_DIR}/llama2-ckpts/llama2-7b-hf-to-megatron-tp1-pp1   \
100000000   \
10000   \
${WORK_DIR}/output_megatron_llama2/
```

有监督微调：

```
ENV=$1                          # 运行环境: dlc, dsw
MEGATRON_PATH=$2                # 设置开源Megatron的代码路径
MEGATRON_PATCH_PATH=$3          # 设置Megatron Patch的代码路径
MODEL_SIZE=$4                   # 模型结构参数量级: 7B, 13B
BATCH_SIZE=$5                   # 每卡训练一次迭代样本数: 4, 8
LR=$6                           # 学习率: 1e-5, 5e-5
MIN_LR=$7                       # 最小学习率: 1e-6, 5e-6
SEQ_LEN=$8                      # 序列长度
PAD_LEN=$9                      # Padding长度：100
EXTRA_VOCAB_SIZE=${10}          # 词表扩充大小
PR=${11}                        # 训练精度: fp16, bf16
TP=${12}                        # 模型并行度
PP=${13}                        # 流水并行度
AC=${14}                        # 激活检查点模式: sel, full
DO=${15}                        # 是否使用Megatron版Zero-1降显存优化器: true, false
FL=${16}                        # 是否使用Flash Attention: true, false
SP=${17}                        # 是否使用序列并行: true, false
TRAIN_DATASET_PATH=${18}        # 训练数据集路径
VALID_DATASET_PATH=${19}        # 验证数据集路径
PRETRAIN_CHECKPOINT_PATH=${20}  # 预训练模型路径
EPOCH=${21}                     # 训练迭代轮次
OUTPUT_BASEPATH=${22}           # 训练输出文件路径
```


```
export WORK_DIR=/mnt/workspace
cd ${WORK_DIR}/PAI-Megatron-Patch/examples/llama2
sh run_finetune_megatron_llama.sh  \
dsw  \
/root/Megatron-LM-230512  \
${WORK_DIR}/PAI-Megatron-Patch  \
7B     \
1      \
1e-5   \
1e-6   \
2048   \
80     \
0      \
bf16   \
1      \
1      \
sel    \
true   \
false  \
false  \
${WORK_DIR}/llama2-datasets/wudao_train.json   \
${WORK_DIR}/llama2-datasets/wudao_valid.json   \
${WORK_DIR}/llama2-ckpts/llama2-7b-hf-to-megatron-tp1-pp1   \
2   \
${WORK_DIR}/output_megatron_llama2/
```





# 模型推理

```
ENV=$1                          # 运行环境: dlc, dsw
MEGATRON_PATH=$2                # 设置开源Megatron的代码路径
MEGATRON_PATCH_PATH=$3          # 设置Megatron Patch的代码路径
CHECKPOINT_PATH=$4              # 模型微调阶段的模型保存路径
MODEL_SIZE=$5                   # 模型结构参数量级: 1.1B, 1.7B, 7.1B
TP=$6                           # 模型并行度
BS=$7                           # 每卡推理一次迭代样本数: 1, 4, 8
SEQ_LEN=$8						# 序列长度: 256, 512, 1024
PAD_LEN=$9                      # PAD长度：需要将文本拼接到的长度
EXTRA_VOCAB_SIZE=${10}          # 模型转换时增加的token数量
PR=${11}                        # 推理采用的精度: fp16, bf16
TOP_K=${12}                     # 采样策略中选择排在前面的候选词数量(0-n): 0, 5, 10, 20
INPUT_SEQ_LEN=${13}             # 输入序列长度: 512
OUTPUT_SEQ_LEN=${14}            # 输出序列长度: 256
INPUT_FILE=${15}                # 需要推理的文本文件: input.txt, 每行为一个样本
OUTPUT_FILE=${16}               # 推理输出的文件: output.txt
# TOP_K和TOP_P必须有一个为0
TOP_P=${17}                     # 采样策略中选择排在前面的候选词百分比(0-1): 0, 0.85, 0.95
TEMPERATURE=${18}               # 采样策略中温度惩罚: 1-n
REPETITION_PENALTY=${19}        # 避免生成是产生大量重复，可以设置为(1-2)默认为1.2
```


```
export WORK_DIR=/mnt/workspace
cd ${WORK_DIR}/PAI-Megatron-Patch/examples/llama2
bash run_text_generation_megatron_llama.sh \
dsw \
/root/Megatron-LM-23.04 \
${WORK_DIR}/PAI-Megatron-Patch \
../../../llama2-train \
7B \
1 \
1 \
1024 \
1024 \
0 \
fp16 \
10 \
512 \
512 \
${WORK_DIR}/pred_input.jsonl \
${WORK_DIR}/llama2_pred.txt \
0 \
1.0 \
1.2
```


**参考文档**：

- 数据预处理: https://github.com/alibaba/Pai-Megatron-Patch/blob/main/toolkits/pretrain_data_preprocessing/README.md
- Huggingface&DeepSpeed训练流程：https://github.com/alibaba/Pai-Megatron-Patch/blob/main/examples/hfds.md
- Megatron训练流程：https://github.com/alibaba/Pai-Megatron-Patch/blob/main/examples/megatron.md
- Megatron推理：https://github.com/alibaba/Pai-Megatron-Patch/blob/main/megatron_patch/generation/megatron.md




