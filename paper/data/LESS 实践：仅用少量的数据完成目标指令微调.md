

之前的文章 [LESS：仅选择5%有影响力的数据优于全量数据集进行目标指令微调](https://juejin.cn/user/3642056016410728/posts) 中详细讲述了LESS，本文对其进行实践。


## LESS 核心思想

LESS 核心思想通过仅给出**少数体现特定能力的示例**，从大量指令数据集中**有效地选择5%有影响力的数据**用于目标指令微调，结果优于全量数据集进行微调，并且所选子集在不同模型参数规模和不同模型系列中仍然普遍有效。

**数据选择流水线**：
1. 使用 LoRA 进行热身训练。
2. 构建了一个**投影低维梯度特征**的**梯度数据存储**，可以重复用于不同的目标任务。
3. 利用**数据选择算法**使用数据存储来构建训练数据集。
4. 使用选择的数据训练模型。

**实验关键结果**：
1. **LESS 在不同的模型中都是有效的**
2. **使用LESS 选择 5% 的数据通常优于完整数据集进行训练**。
3. **使用小模型选择的数据可以提高较大的模型和不同模型的性能**。

**LESS 存在的局限性**：

1. 需要使用候选数据 D 的随机 5% 进行**热身训练**。对于获得有用的梯度特征以进行数据选择至关重要，但增加了 LESS 的复杂性和计算负载
2. 使用补全Token的平均梯度，这增加了较短训练序列的权重，从而导致性能明显变差。为了缓解这个问题，对 **LESS 中的梯度特征进行归一化，并使用余弦相似度而不是点积来估计影响**。
3. 最小化验证损失（即交叉熵损失）不会单调提高准确性。
4. 一阶近似**忽略了将多个数据点添加在一起的影响**。特别是，两个重复的点将获得同样高的分数，并被认为可以双重改进模型，但情况可能并非如此。


## LESS 应用


### 环境安装

```
pip3 install torch==2.1.2 torchvision torchaudio

cd LESS
pip install -r requirement.txt

# 以可编辑模式安装 `less` 包，使其可供您的开发环境访问
pip install -e .
```

### 数据准备

按照 [open-instruct](https://github.com/allenai/open-instruct?tab=readme-ov-file#dataset-preparation)  库来准备指令调优数据集。这里结合使用了四个训练数据集：Flan v2、COT、Dolly 和 Open Assistant。出于评估目的，还使用了三个额外的数据集：MMLU、Tydiqa 和 BBH。[此处](https://huggingface.co/datasets/princeton-nlp/less_data)提供了这些文件的处理版本。

### 数据选择流水线

#### 1. 热身训练

为了提高数据选择的性能，预热训练步骤至关重要。通过选择整个数据集的一小部分来使用 LoRA 方法进行训练。

热身训练执行脚本：

```
DATA_DIR=../data
MODEL_PATH=meta-llama/Llama-2-7b-hf
PERCENTAGE=0.05 # percentage of the full data to train, you can specify the training file you want to use in the script
DATA_SEED=3
JOB_NAME=llama2-7b-p${PERCENTAGE}-lora-seed${DATA_SEED}

./less/scripts/train/warmup_lora_train.sh "$DATA_DIR" "$MODEL_PATH" "$PERCENTAGE" "$DATA_SEED" "$JOB_NAME"
```

#### 2. 构建梯度数据存储

初始预热训练阶段完成后，将收集整个训练数据集的梯度。对于每个检查点，我们的目标是获取我们想要选择的所有训练数据的梯度。


执行脚本：

```bash
CKPT=105

TRAINING_DATA_NAME=dolly
TRAINING_DATA_FILE=../data/train/processed/dolly/dolly_data.jsonl # when changing data name, change the data path accordingly

GRADIENT_TYPE="adam"
MODEL_PATH=../out/llama2-7b-p0.05-lora-seed3/checkpoint-${CKPT}
OUTPUT_PATH=../grads/llama2-7b-p0.05-lora-seed3/${TRAINING_DATA_NAME}-ckpt${CKPT}-${GRADIENT_TYPE}
DIMS="8192"

./less/scripts/get_info/get_train_lora_grads.sh \
"$TRAINING_DATA_FILE" \
"$MODEL_PATH" \
"$OUTPUT_PATH" \
"$DIMS" \
"$GRADIENT_TYPE"
```

创建的一个数据存储，包含了您希望从中选择的所有检查点和训练数据的梯度。

#### 3. 为任务选择数据

要为特定下游任务选择数据，首先使用与训练期间使用的相同的指令调优提示格式准备特定于该任务的数据。

这里为三个评估数据集设置了数据加载模块：BBH、TydiQA 和 MMLU。如果您对其他任务的数据选择感兴趣，可以扩展 `less/data_selection/get_validation_dataset.py` 脚本以适应这些任务。与获取训练数据的梯度类似，运行以下脚本。主要区别在于，此过程将根据影响力估计的公式生成验证数据的 SGD 梯度。

```
CKPT=105
TASK=tydiqa
MODEL_PATH=../out/llama2-7b-p0.05-lora-seed3/checkpoint-${CKPT}
OUTPUT_PATH=../grads/llama2-7b-p0.05-lora-seed3/${TASK}-ckpt${CKPT}-sgd # for validation data, we always use sgd
DATA_DIR=../data
DIMS="4096 8192" # We use 8192 as our default projection dimension 

./less/scripts/get_info/get_eval_lora_grads.sh "$TASK" "$DATA_DIR" "$MODEL_PATH" $OUTPUT_PATH "$DIMS"
```
正常情况下，你获得在上一步中用于构建梯度数据存储的所有检查点的验证数据的梯度。

获得验证数据的梯度后，就可以为任务选择数据。以下脚本将计算每个训练数据点的影响力得分，并选择影响力得分最高的前 k 个数据点。

```bash
# decide which dimension to use
DIM=8192

# checkpoing index
CKPTS="105 211 317 420" 
# average lr of the epoch
CHECKPOINT_WEIGHTS="1.6877e-05 1.2859e-05 7.7030e-06 2.5616e-06" 

GRADIENT_PATH=../grads/llama2-7b-p0.05-lora-seed3/{}-ckpt{}-adam/dim${DIM}
TRAIN_FILE_NAMES="flan_v2 cot dolly oasst1"

VALIDATION_GRADIENT_PATH=../grads/llama2-7b-p0.05-lora-seed3/{}-ckpt{}-sgd/dim${DIM}
TARGET_TASK_NAMES="tydiqa"

SELECTED_DATA_OUTPUT_PATH="../selected_data"

./less/scripts/data_selection/matching.sh \
"$GRADIENT_PATH" \
"$TRAIN_FILE_NAMES" \
"$CKPTS" \
"$CHECKPOINT_WEIGHTS" \
"$VALIDATION_GRADIENT_PATH" \
"$TARGET_TASK_NAMES" \
"$SELECTED_DATA_OUTPUT_PATH"
```

每个训练数据点的影响力得分将保存在 `OUTPUT_PATH` 目录中。使用以下脚本来选择影响力得分最高的前 k 个数据点。

```
python3 -m less.data_selection.write_selected_data \
--target_task_names ${TARGET_TASK_NAMES} \
--train_file_names ${TRAIN_FILE_NAMES} \
--train_files ../data/train/processed/dolly/dolly_data.jsonl ../data/train/processed/oasst1/oasst1_data.jsonl \
--output_path $SELECTED_DATA_OUTPUT_PATH \
--percentage 0.05
```


#### 4. 使用选择的数据进行训练

选择数据后，使用以下脚本使用所选数据训练模型。


```bash
TARGET_TASK_NAME="tydiqa"
PERCENTAGE=0.05
TRAIN_FILES=../selected_data/${TARGET_TASK_NAME}/top_p${PERCENTAGE}.jsonl
MODEL_PATH=meta-llama/Llama-2-7b-hf
JOB_NAME=llama2-7b-less-p${PERCENTAGE}-lora

./less/scripts/train/lora_train.sh "$TRAIN_FILES" "$MODEL_PATH" "$JOB_NAME" 
```

> 注意：这里您也可以通过删除 lora 训练参数来执行全参数微调。


### 评估

这里使用三个评估数据集（MMLU、Tydiqa 和 BBH）来评估数据选择流水线的性能：。使用评估流水线 open-instruct。按照以下步骤操作评估经过训练的模型，请：

#### 1：安装 Open-Instruct

```
git clone https://github.com/allenai/open-instruct.git
cd open-instruct
pip install -e .
```

#### 2：评估

查看 [`evaluation`](https://github.com/princeton-nlp/LESS/tree/main/evaluation) 目录中的 `eval_mmlu.sh` 、 `eval_tydiqa.sh` 和 `eval_bbh.sh` 脚本。这些脚本包含在相应数据集上评估模型所需的命令。`eval_bbh.sh` 脚本如下：

```
source eval.sh

# 主评估函数
eval_bbh() {
    cd $n/space10/open-instruct
    mdir=$1
    type=$2
    set_save_dir $mdir bbh
    mkdir -p $save_dir
    cmd="python -m eval.bbh.run_eval \
    --data_dir $DATA_DIR/bbh \
    --save_dir $save_dir \
    --model $mdir \
    --tokenizer $mdir \
    --eval_batch_size 10 \
    --convert_to_bf16 \
    --max_num_examples_per_task 40"
    eval "$cmd"
}

# 评估校验集，目前还不支持
valid_bbh() {
    cd $n/space10/open-instruct
    mdir=$1
    type=$2
    set_valid_dir $mdir bbh
    echo $save_dir
    mkdir -p $save_dir
    cmd="python -m eval.bbh.run_eval \
    --data_dir $DATA_DIR/bbh-valid \
    --save_dir $save_dir \
    --model $mdir \
    --tokenizer $mdir \
    --eval_batch_size 10 \
    --convert_to_bf16 \
    --eval_valid \
    --max_num_examples_per_task 3"
}

# 提取结果
extract_bbh() {
    mdir=$1
    set_save_dir $mdir bbh-nonchat
    result=$(jq .average_exact_match $save_dir/metrics.json)
    result=$(echo "$result * 100" | bc)
    echo $result
}

# 提取验证集的结果
extract_valid_bbh() {
    mdir=$1
    set_valid_dir $mdir bbh-nonchat
    result=$(jq .average_exact_match $save_dir/metrics.json)
    result=$(echo "$result * 100" | bc)
    echo $result
}
```

## 结语

本文简要介绍了 LESS 的核心思想，同时讲述了 LESS 的应用实践。

码字不易，如果觉得我的文章能够能够给您带来帮助，期待您的点赞收藏加关注~~


参考文档：
- 论文：LESS: Selecting Influential Data for Targeted Instruction Tuning
- 代码：https://github.com/princeton-nlp/LESS







