[TOC]

# BaiChuan2-13B模型-推理指导

# 概述

BaiChuan 2 是百川智能推出的新一代开源大语言模型，采用 2.6 万亿 Tokens 的高质量语料训练，在权威的中文和英文
benchmark上均取得同尺寸最好的效果。本次发布包含有 7B、13B 的 Base 和 Chat 版本，并提供了 Chat 版本的 4bits
量化，所有版本不仅对学术研究完全开放，开发者也仅需邮件申请并获得官方商用许可后，即可以免费商用。

- 参考实现：

  ```
  https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat
  ```

# 快速上手

## 路径变量解释

| 变量名                 | 含义                                                                   |  
|---------------------|----------------------------------------------------------------------|
| model_download_path | 开源权重放置目录                                                             | 
| llm_path            | 加速库及模型库下载后放置目录                                                       |
| model_path          | 工作时模型所在的目录，可以和model_download_path相同，但一般模型是公共的，为了避免影响其他用户，单独建一个模型工作目录 |
| script_path         | 工作脚本所在路径，本文为${llm_path}/pytorch/examples/baichuan2/13b               |
| ceval_work_dir      | ceval数据集、及结果保存所在目录，不必和模型脚本在相同目录                                      |

## 获取源码及依赖

### 1.python requirements

| 包名            | 推荐版本   |  
|---------------|--------|
| transformers  | 4.30.2 | 
| decorator     | 5.1.1  |
| sympy         | 1.11.1 |
| scipy         | 1.11.3 |
| attrs         | 23.1.0 |
| psutil        | 5.9.6  |
| sentencepiece | 0.1.99 |

### 下载模型权重

下载模型权重，放置到自定义`${model_download_path}` (请下载链接中'Files and versions'页签下的所有文件)

```
https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat/tree/main
```

### 拷贝文件

### 准备

#### 1. 将开源模型拷贝到模型工作目录，bin文件使用软链接即可,同时将modeling文件拷贝到模型，并修改开源的config.json,

```shell
cd ${model_path}
cp ${model_download_path}/*.py ${model_path}/
cp ${model_download_path}/*.json ${model_path}/
cp ${model_download_path}/*.model ${model_path}/
cp -s ${model_download_path}/*.bin ${model_path}/
```

#### 2. 安装 atb_speed_sdk

```shell
cd ${llm_path}/pytorch/examples/atb_speed_sdk
pip install .
```

#### 3. 张量并行模型切分（仅在模型需要多卡并行时使用）

```shell
cp ${script_path}/modeling_baichuan_cut.py ${model_path}
```

修改 ${model_path}里的config.json中的kv对，改成`"AutoModelForCausalLM": "modeling_baichuan_cut.BaichuanForCausalLM"`

```text
修改`${script_path}/cut_model_and_run.sh`    
将 `input_dir` 修改为模型所在路径 `${model_path}` 
将 `output_dir` 修改为切分后的模型所存储的路径,比如仍为原目录 `${model_path}`。模型切分成功后，会自动生成新目录part_model(用户无需新建该文件夹)，即：${model_path/part_model}
将 `world_size_` 修改成希望切成的卡的数量
```

目录结构示例建议

```
--model_path
  *.py(模型源文件)
  *.json(模型源文件)
  *.model(模型源文件)
  *.bin(模型源文件,软链接)
  modeling_baichuan_cut.py(权重切分脚本)
  --part_model(权重切分成功后文件夹)
    --0
    --1
  ......(其他)
--script_path
  cut_model_and_run.sh
  cut_model_util.py
  main.py
  config.ini
  ......(其他)
```

执行

```shell
cd ${script_path}
bash cut_model_and_run.sh
```

切分所需时间较长，切分完成后，将会打印 'Tensor parallelism weights have been successfully saved.'。

#### 4.修改config.json配置

- 单卡运行时**必须**修改
- 多卡运行时，会在切分阶段会自动修改，没有定制的情况下，可以不操作

##### 单卡
拷贝修改后的modeling
```shell
cp ${script_path}/modeling_baichuan_ascend.py ${model_path}
```
修改${model_path}/config.json中的kv对，改成

```
AutoModelForCausalLM": "modeling_baichuan_ascend.BaichuanForCausalLM
```

##### 多卡

修改
${model_path}/part_model/{rank_id}/config.json中的kv对，改成

```
AutoModelForCausalLM": "modeling_baichuan_ascend.BaichuanForCausalLM
```

# CPU高性能模式

可开启CPU Performance模式以提高模型推理性能。

```
cpupower frequency-set -g performance
```

### 执行推理

#### 修改 ${script_path}/config.ini

[config文件配置参考](../../atb_speed_sdk/README.md)  
提示：多卡并行推理时，config.json中model_path路径为part_model父文件夹。例如：

```
# 正确示例：
model_path=../model
# 错误示例：
model_path=../model/part_model
```

#### main.py

提供了demo推理，精度测试，性能测试三种下游任务。  
task_name可选inference、precision、performance。  
is_quant代表是否量化（0代表浮点，1代表量化），本节为浮点推理，设置为0即可。

- 单芯
  修改 ${model_path}里的config.json中的kv对，改成`"AutoModelForCausalLM": "modeling_baichuan_ascend.BaichuanForCausalLM"`

```shell
python main.py --task ${task_name}  --is_quant ${is_quant}
```

- 多芯

```shell
bash cut_model_and_run.sh ${task_name}  ${is_quant}
```

#### FAQ

1. **可以使用 MAX_SEQ_LEN 环境变量来设置model支持的最大长度以优化显存占用,
   一般设置为最大输入输出token之和，默认使用config里面的max_model_length**  
   如

```shell
MAX_SEQ_LEN=2048 python main.py --task ${task_name}  --is_quant ${is_quant}
```

或  
修改cut_model_and_run.sh 中的 max_seq_length

```shell
bash cut_model_and_run.sh ${task_name}  ${is_quant}
```

2. ImportError: /root/miniconda3/envs/wqh39/bin/../lib/libgomp.so.1: cannot allocate memory in static TLS block

如果遇到

```text
Traceback (most recent call last):
  File "/root/miniconda3/envs/wqh39/lib/python3.9/site-packages/torch_npu/__init__.py", line 31, in <module>
    import torch_npu.npu
  File "/root/miniconda3/envs/wqh39/lib/python3.9/site-packages/torch_npu/npu/__init__.py", line 46, in <module>
    from .utils import (is_initialized, _lazy_call, _lazy_init, init, set_dump,
  File "/root/miniconda3/envs/wqh39/lib/python3.9/site-packages/torch_npu/npu/utils.py", line 27, in <module>
    import torch_npu._C
ImportError: /root/miniconda3/envs/wqh39/bin/../lib/libgomp.so.1: cannot allocate memory in static TLS block
Segmentation fault (core dumped)
```

则在命令行前加上`LD_PRELOAD=上面的error路径`。如

```shell
LD_PRELOAD=/root/miniconda3/envs/wqh39/bin/../lib/libgomp.so.1 MAX_SEQ_LEN=2048 python main.py --task ${task_name}  --is_quant ${is_quant}
```


## 量化推理

# 量化工具使用

量化权重的获取需要使用大模型量化工具（集成至CANN包中），详细操作手册可见[大模型权重量化工具-ModelSlim](https://www.hiascend.com/document/detail/zh/canncommercial/70RC1/devtools/auxiliarydevtool/modelslim_0001.html)
。针对Baichuan2-13B的权重量化可参考如下步骤，运行时需将下述三个步骤的代码整合为一个python文件

**特别注意1**：本章节依赖**pytorch >= 2.0.0 CANN >= 7.0.0.B060**
环境，大模型量化工具依赖指定pytorch版本（不依赖torch_npu，只依赖原生torch）。该环境的pytorch版本与后续步骤可能不同，后续将优化pytorch版本依赖的限制

**特别注意2**：本章节依赖 hugging face 的标准 transformers 包。若环境中的 transformers 包被改动过，可能引起相关报错，此时建议重新安装
transformers 包

**特别注意3**：本章节执行完毕后，在`QUANT_WEIGHT_PATH`路径下生成如下权重文件，请检查是否缺失：

```
deq_scale.npy  fp_bias.npy
input_offset.npy  input_scale.npy
quant_bias.npy  quant_weight.npy
weight_offset.npy  weight_scale.npy
```

## 校准数据准备

```python
calib_list = ["中国的首都在哪里？",
              "请做一首诗歌：",
              "我想要学习python，该怎么学习？",
              "请帮我写一篇关于大模型推理优化的任职报告：",
              "中国最值得去的几个景点"]


# 获取校准数据函数定义
def get_calib_dataset(tokenizer, calib_list):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer([calib_data], return_tensors='pt').to('cpu')
        print(inputs)
        calib_dataset.append([inputs.data['input_ids'], inputs.data['position_ids'], inputs.data['attention_mask']])
    return calib_dataset


dataset_calib = get_calib_dataset(tokenizer, calib_list)  # 校准数据获取
```

## 量化参数配置与运行

```python
from modelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig

quant_config = QuantConfig(w_bit=8, disable_names=['transformer.output_layer'], dev_type='cpu', act_method=3, pr=0.5,
                           mm_tensor=False, w_hessian=False)
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L1')
calibrator.run()  # 执行PTQ量化校准
calibrator.save('QUANT_WEIGHT_PATH')  # 保存量化参数
```

- 建议直接使用量化权重生成脚本，生成量化权重
  ```
  python quant.py
  ```

> 注：要使用torch2.0.0导出量化权重，否则会有精度偏差 quant.py脚本需要修改calibrator.save('QUANT_WEIGHT_PATH') 最终量化全的指定路径

2. 量化权重切分

- 修改代码
    1. 修改`cut_quant_model_util.py`中`--input_path`为实际存放量化权重的路径
    2. 修改`cut_quant_model_util.py`中`--output_dir`为自定义路径，用于存放切分后的模型量化权重
- 执行切分
  ```
  python cut_quant_model_util.py
  # 切分好的模型权重会存放在自定义的output_dir 
  ```

3. 适配量化推理代码

- 进入modeling_baichuan_quant_parallel.py，适配量化权重路径和回退层
  ```
  # 修改以下全局变量
  self.quant_weight_path = '/code/models/baichuan2/quanted_weight_cut_1123_' 量化切分权重路径 及上一步的output_dir
  self.cut_float_weight = '' 浮点切分权重路径 
  self.roll_back_layer = [0,1,2,3,4,7,9,10,17,18,19,20,22,23,24,26,36,37,38,39]
  ```
  **特别注意**：此处的self.roll_back_layer必须与quant.py里面的disable_idx_lst 保持一致

4. 执行量化模型推理

  ```
  单独推理
  bash cut_model_and_run_baichuan.sh inference 1
  精度
  bash cut_model_and_run_baichuan.sh precision 1
  性能
  bash cut_model_and_run_baichuan.sh performance 1
  
  具体参考atb_speed_sdk 使用README.md
  ```

**特别注意 **

# 竞品对比

# 800T A2

## 精度

| 精度             | NPU         | GPU         | 对比 |
|----------------|-------------|-------------|----|
| STEM           | 0.472093023 | 0.472093023 | 1  |
| Social Science | 0.661818182 | 0.661818182 | 1  |
| Humanities     | 0.630350195 | 0.630350195 | 1  |
| Other          | 0.567708333 | 0.567708333 | 1  |
| Avg acc        | 0.568350669 | 0.568350669 | 1  |

## 性能

| 芯片型号                          | 首token推理速度(token/s) | 比例          | 增量推理速度(token/s)   | 对比          |
|-------------------------------|---------------------|-------------|-------------------|-------------|
| Baichuan-13B NPU              | 14.260809086490132  |             | 31.69616807901823 |             |
| Baichuan-13B A100(80G) NVlink | 15.642417690338782  | 0.911675508 | 36.41638939692089 | 0.870381952 |

# 300I DUO

## 性能

浮点

| 硬件形态  | 批大小 | 输入长度     | 输出长度     | 首次推理（ms/token） | 非首次推理(ms/token) |
|-------|-----|----------|----------|----------------|-----------------|
| Duo双芯 | 1   | 2^5~2^10 | 2^5~2^10 | 327            | 103             |

量化

| 硬件形态  | 批大小 | 输入长度     | 输出长度     | 首次推理（ms/token） | 非首次推理(ms/token) |
|-------|-----|----------|----------|----------------|-----------------|
| Duo双芯 | 1   | 2^5~2^10 | 2^5~2^10 | \              | 75              |

## 精度

| 精度             | NPU         | GPU         | 对比          |
|----------------|-------------|-------------|-------------| 
| STEM           | 0.472093023 | 0.472093023 | 1           |
| Social Science | 0.658181818 | 0.661818182 | 0.994505494 |
| Humanities     | 0.630350195 | 0.630350195 | 1           |
| Other          | 0.572916667 | 0.567708333 | 1.009174313 |
| Avg acc        | 0.569093611 | 0.568350669 | 1.001307189 |

# 附录

# 精度测试指南

## 配置说明

参考 [SDK精度测试指南CEVAL章节](../../atb_speed_sdk/README.md)

## 运行脚本

- 单芯

```shell
cd ${script_path}
python main.py --task precision
```

- 多芯

```shell
cd ${script_path}
bash cut_model_and_run.sh precision
```

结束后在${ceval_work_dir}/test_result目录下查看测试结果。[双芯结果每个两份，只需看其中一份即可]。

| 文件                        | 用途                   | 
|---------------------------|----------------------| 
| device0.log               | 运行过程日志               |
| cache0.csv                | 结果详情，C列为预期答案，D列为测试答案 |
| result_0_classes_acc.json | 测试数据下按不同维度统计准确率      |
| result_0_subject_acc.json | 测试数据下按不同学科统计准确率      |

**注意：后续重新运行， 需要删除当前目录下生成的test_result文件夹，否则只会读取当前的目录下的测试结果**

# 性能测试

在功能运行正常的基础下，执行以下步骤进行性能测试

## 按照推理指导,下载模型及配置路径，并安装atb_speed_sdk

## 1. 准备

参考 [SDK性能测试指南精确打点法章节](../../atb_speed_sdk/README.md) 进行准备

## 2. 修改配置文件

- 配置config.ini中[performance]属性， 如下：
  ```
  model_name=baichuan2_13b
  perf_mode=detail
  ```

## 3. 执行测试脚本

- 单芯

```shell
cd ${script_path}
TIMEIT=1 python main.py --task performance
```

- 多芯

```shell
cd ${script_path}
TIMEIT=1 bash cut_model_and_run.sh performance 0
```

将`TIMEIT`设置成1来返回具体的性能测试的值，默认是0  
上述多芯场景参数

* performance表示性能测试。
* 0 表示浮点，1表示量化

### 性能测试结果

得到性能测试结果csv `performance_test_npu_${model_name}_xxx.csv`

### 结果分析

| 列名                            | 含义         |
|-------------------------------|------------|
| batch_size                    | batch大小    |
| input_seq_len(Encoding)       | 输入长度       |
| output_seq_len(Decoding)           | 输出长度       |
| ResponseTime(s)                     | 总响应时间      |
| forward_first_token_time(ms)  | 首token推理时长 |
| forward_next_token_time(ms)   | 增量推理时长     |
| pre_next_token_time(ms)             | 前处理时长      |
| post_next_token_time_post(ms) | 后处理时长      |