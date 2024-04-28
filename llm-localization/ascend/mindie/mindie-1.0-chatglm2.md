# ChatGLM2-6B 模型推理指导 <!-- omit in toc -->

- [概述](#概述)
- [输入输出数据](#输入输出数据)
- [推理前准备](#推理前准备)
- [量化工具使用](#量化工具使用)
- [快速上手](#快速上手)
  - [获取源码及依赖](#获取源码及依赖)
  - [模型推理](#模型推理)
- [模型参考精度和性能结果](#模型参考精度和性能结果)

# 概述

[ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B/) 是开源中英双语对话模型 [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) 的第二代版本，在保留了初代模型对话流畅、部署门槛较低等众多优秀特性的基础之上，ChatGLM2-6B有更强大的性能、更长的上下文、更高效的推理和更开放的协议。

# 输入输出数据

- 输入数据

  | 输入数据       | 大小                 | 数据类型 | 数据排布格式 | 是否必选 |
  | -------------- | -------------------- | -------- | ------------ | -------- |
  | input_ids      | BATCH_SIZE x SEQ_LEN | INT64    | ND           | 是       |
  | attention_mask | BATCH_SIZE x SEQ_LEN | FLOAT32  | ND           | 否       |

- 输出数据

  | 输出数据   | 大小                        | 数据类型 | 数据排布格式 |
  | ---------- | --------------------------- | -------- | ------------ |
  | output_ids | BATCH_SIZE x OUTPUT_SEQ_LEN | INT64    | ND           |

# 推理前准备

1. 参见 [推理环境准备](../../../../docs/推理环境准备.md) 安装 固件与驱动，CANN，PyTorchAdapter等基础软件。
   ```shell
   # 使能cann环境变量（根据实际安装路径修改）
   source ${path-to-ascend-toolkit}/set_env.sh
   # 使能加速库环境变量（根据实际安装路径修改）
   source ${path-to-ascendTB}/set_env.sh
   # 使能inference库环境变量
   source ${path-to-atb_models}/set_env.sh
   # 稀疏工具在线编译(可选)
   cd ${path-to-ascend-toolkit}/tools/modelslim/pytorch/weight_compression/compress_graph/
   bash build.sh ${path-to-ascend-toolkit}/ascend-toolkit/latest/
   ```

2. 下载模型实现文件和权重文件，并存储到任意路径下 `CHECKPOINT={path-to-weights}`

     - 推荐下载方式

       ```shell
       # 请自行确认已安装 git-lfs
       git lfs install
       git clone https://huggingface.co/THUDM/chatglm2-6b
       cd chatglm2-6b
       git reset --hard 4e38bef4c028beafc8fb1837462f74c02e68fcc2
       ```

     - 其他下载方式

       如果你的网络环境较差，下载模型参数可能会花费较长时间甚至失败。此时可以先将模型下载到本地，然后从本地加载。
       - 分开下载模型实现文件和权重文件
         ```shell
         # 只下载模型实现文件
         GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/THUDM/chatglm2-6b
         cd chatglm2-6b
         git reset --hard 4e38bef4c028beafc8fb1837462f74c02e68fcc2
         ```
         从 [这里](https://cloud.tsinghua.edu.cn/d/674208019e314311ab5c/) 手动下载模型参数文件，并将下载的文件替换到本地的 `chatglm2-6b` 目录下。

       - 手动从 [THUDM/chatglm2-6b](https://huggingface.co/THUDM/chatglm2-6b) 下载所有文件

     - 下载后检查`${CHECKPOINT}`目录如下所示

       ```
       |-- config.json
       |-- configuration_chatglm.py
       |-- modeling_chatglm.py
       |-- pytorch_model-00001-of-00007.bin
       |-- pytorch_model-00002-of-00007.bin
       |-- pytorch_model-00003-of-00007.bin
       |-- pytorch_model-00004-of-00007.bin
       |-- pytorch_model-00005-of-00007.bin
       |-- pytorch_model-00006-of-00007.bin
       |-- pytorch_model-00007-of-00007.bin
       |-- pytorch_model.bin.index.json
       |-- quantization.py
       |-- tokenization_chatglm.py
       |-- tokenizer_config.json
       |-- tokenizer.model
       ```

     - 在config.json中添加如下配置：

       ```
       {
         ......
         "world_size": 1,
         "float_layers_id": [0]
       }
       ```

3. 获取量化权重

     - 直接下载量化权重

       - [A300I DUO 量化权重下载](https://model-weight.obs.cn-north-4.myhuaweicloud.com/chatglm2_6B_310p.tar.gz)
       - [A800I A2 量化权重下载](https://model-weight.obs.cn-north-4.myhuaweicloud.com/chatglm2_6B_910b.tar.gz)

       请使用wget下载，下载完成后请将文件解压到任意路径`QUANT_WEIGHT_PATH=${path-to-quant-weight}`

     - 手动生成量化权重

       详见章节[量化工具使用](#量化工具使用)

4. 下载 `C-Eval` 数据集

   从 [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/e84444333b6d434ea7b0) 下载处理好的 `C-Eval` 数据集，解压到任意目录下 `DATASET={path-to-dataset}` 。

# 量化工具使用

量化权重的获取需要使用大模型量化工具（集成至CANN包中），详细操作手册可见[大模型权重量化工具-ModelSlim](https://www.hiascend.com/document/detail/zh/canncommercial/70RC1/devtools/auxiliarydevtool/modelslim_0001.html)。

导出 ChatGLM2-6B 的量化权重或者是稀疏量化权重：

```shell
# 量化权重导出
python export_quant_weight.py --float_weight ${CHECKPOINT} --data_path ${DATASET}/val/Social_Science/teacher_qualification.jsonl --quant_weight ${QUANT_WEIGHT_PATH}
# 稀疏量化权重导出
python export_quant_weight.py --float_weight ${CHECKPOINT} --data_path ${DATASET}/val/Other/civil_servant.jsonl --quant_weight ${QUANT_WEIGHT_PATH} --sparse
```

参数说明：

- float_weight：浮点权重路径。
- data_path：用于校准的数据文件路径。
- quant_weight：导出的量化权重或者是稀疏量化权重路径。
- sparse：默认为false,指量化，True指稀疏量化。

**特别注意1**：本章节依赖**pytorch 2.0.0**环境，大模型量化工具依赖指定pytorch版本（不依赖torch_npu，只依赖原生torch）。该环境的pytorch版本与后续步骤可能不同，后续将优化pytorch版本依赖的限制

**特别注意2**：本章节依赖 hugging face 的标准 transformers 包。若环境中的 transformers 包被改动过，可能引起相关报错，此时建议重新安装 transformers 包

**特别注意3**：稀疏量化权重的获取详见[大模型稀疏权重工具使用文档](https://codehub-y.huawei.com/mindstudio/MindStudio-Backend/automl/files?ref=master&filePath=modelslim%2Fpytorch%2Fllm_sparsequant%2FREADME.md&isFile=true)

**特别注意4**：本章节执行完毕后，在`QUANT_WEIGHT_PATH`路径下生成如下权重文件，请检查是否缺失：

```
deq_scale.npy  fp_bias.npy
input_offset.npy  input_scale.npy
quant_bias.npy  quant_weight.npy
weight_offset.npy  weight_scale.npy
```

# 快速上手

## 获取源码及依赖

1. 获取源码

   ```shell
   cd ${path-to-atb_models}/pytorch/examples/chatglm2/6b
   ```
2. 安装第三方依赖

    ```shell
    pip install -r requirements.txt
    ```

## 模型推理

- 可开启CPU Performance模式以提高模型推理性能

  ```
  cpupower frequency-set -g performance
  ```

- 推理前开启如下环境变量

  ```shell
  export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE
  export TASK_QUEUE_ENABLE=1
  export ATB_OPERATION_EXECUTE_ASYNC=1
  export ATB_LAYER_INTERNAL_TENSOR_REUSE=1

  # 仅300 Ipro和300 IDuo上开启
  export HCCL_BUFFSIZE=110
  export ATB_USE_TILING_COPY_STREAM=1
  ```

- `C-Eval` 数据集推理

  ```shell
  # 浮点
  # 将TP_SIZE设为对应的并行数，例如单芯场景TP_SIZE=1，双芯场景TP_SIZE=2
  # 多芯场景请先执行权重生成(浮点单芯跳过)
  python process_weights.py --model_path ${CHECKPOINT} --tp_size ${TP_SIZE}
  # 执行浮点推理
  torchrun --nproc_per_node ${TP_SIZE} --master_port 2000 main.py --mode precision_dataset --model_path ${CHECKPOINT} --ceval_dataset ${DATASET} --batch 8 --tp_size ${TP_SIZE}

  # 量化
  # 添加量化环境变量
  export ENABLE_QUANT=1
  export QUANT_WEIGHT_PATH=${QUANT_WEIGHT_PATH}
  # 将TP_SIZE设为对应的并行数，例如单芯场景TP_SIZE=1，双芯场景TP_SIZE=2
  # 执行权重生成（单芯/多芯都要执行）
  python process_weights.py --model_path ${CHECKPOINT} --tp_size ${TP_SIZE}
  # 执行量化推理
  torchrun --nproc_per_node ${TP_SIZE} --master_port 2000 main.py --mode precision_dataset --model_path ${CHECKPOINT} --ceval_dataset ${DATASET} --batch 8 --tp_size ${TP_SIZE}

  # 稀疏量化（当前仅支持300I DUO）
  # 添加稀疏量化环境变量
  export ENABLE_SPARSE=1
  export QUANT_WEIGHT_PATH=${QUANT_WEIGHT_PATH}
  export COMPRESS_WEIGHT_PATH=${COMPRESS_WEIGHT_PATH}
  # 将TP_SIZE设为对应的并行数，例如单芯场景TP_SIZE=1，双芯场景TP_SIZE=2
  # 执行权重生成（单芯/多芯都要执行）
  python process_weights.py --model_path ${CHECKPOINT} --tp_size ${TP_SIZE}
  python3 generate_compress_weight.py --weight_path=${QUANT_WEIGHT_PATH} --save_path=${COMPRESS_WEIGHT_PATH}
  # 执行稀疏量化推理
  torchrun --nproc_per_node ${TP_SIZE} --master_port 2000 main.py --mode precision_dataset --model_path ${CHECKPOINT} --ceval_dataset ${DATASET} --batch 8 --tp_size ${TP_SIZE}
  ```

- <a name="perf">模型性能数据测试</a>

  **性能测试请先配置环境变量`export TIMEIT=1`，测试结束后删除该环境变量`unset TIMEIT`。**

  ```shell
  # 浮点
  # 将TP_SIZE设为对应的并行数，例如单芯场景TP_SIZE=1，双芯场景TP_SIZE=2
  # 多芯场景请先执行权重生成(浮点单芯跳过)
  python process_weights.py --model_path ${CHECKPOINT} --tp_size ${TP_SIZE}
  # 执行浮点推理
  torchrun --nproc_per_node ${TP_SIZE} --master_port 2000 main.py --mode performance --model_path ${CHECKPOINT} --batch ${batch_size} --tp_size ${TP_SIZE}

  # 量化
  # 添加量化环境变量
  export ENABLE_QUANT=1
  export QUANT_WEIGHT_PATH=${QUANT_WEIGHT_PATH}
  # 将TP_SIZE设为对应的并行数，例如单芯场景TP_SIZE=1，双芯场景TP_SIZE=2
  # 执行权重生成（单芯/多芯都要执行）
  python process_weights.py --model_path ${CHECKPOINT} --tp_size ${TP_SIZE}
  # 执行量化推理
  torchrun --nproc_per_node ${TP_SIZE} --master_port 2000 main.py --mode performance --model_path ${CHECKPOINT} --batch ${batch_size} --tp_size ${TP_SIZE}

  # 稀疏量化（当前仅支持300I DUO）
  # 添加稀疏量化环境变量
  export ENABLE_SPARSE=1
  export QUANT_WEIGHT_PATH=${QUANT_WEIGHT_PATH}
  export COMPRESS_WEIGHT_PATH=${COMPRESS_WEIGHT_PATH}
  # 将TP_SIZE设为对应的并行数，例如单芯场景TP_SIZE=1，双芯场景TP_SIZE=2
  # 执行权重生成（单芯/多芯都要执行）
  python process_weights.py --model_path ${CHECKPOINT} --tp_size ${TP_SIZE}
  python3 generate_compress_weight.py --weight_path=${QUANT_WEIGHT_PATH} --save_path=${COMPRESS_WEIGHT_PATH}
  # 执行稀疏量化推理
  torchrun --nproc_per_node ${TP_SIZE} --master_port 2000 main.py --mode performance --model_path ${CHECKPOINT} --batch ${batch_size} --tp_size ${TP_SIZE}
  ```

  备注：

  1. 可通过配置`--seqlen_in_pair`和`--seqlen_out_pair`指定输入输出序列长度，例如以下命令测试的输入输出组合为[256,256]，[512,512]，[1024,1024]

     ```shell
     torchrun --nproc_per_node ${TP_SIZE} --master_port 2000 main.py --mode performance --model_path ${CHECKPOINT} --device 0 --seqlen_in_pair 256,512,1024 --seqlen_out_pair 256,512,1024 --batch 1 --tp_size ${TP_SIZE} --performance_output_file performance_bs1.csv
     ```

  2. 环境变量 `MAX_SEQ_LEN` （默认值2048）必须大于等于 `seqlen_in + seqlen_out`，例如：

     ```shell
     # 若 seqlen_in = 3584 seqlen_out = 512
     export MAX_SEQ_LEN=4096
     ```

- <a name="ui">UI 交互</a>

  - 命令行交互

    ```shell
    # 浮点
    # 将TP_SIZE设为对应的并行数，例如单芯场景TP_SIZE=1，双芯场景TP_SIZE=2
    # 多芯场景请先执行权重生成(浮点单芯跳过)
    python process_weights.py --model_path ${CHECKPOINT} --tp_size ${TP_SIZE}
    # 执行浮点推理
    torchrun --nproc_per_node ${TP_SIZE} --master_port 2000 main.py --mode cli_demo --model_path ${CHECKPOINT} --tp_size ${TP_SIZE}

    # 量化
    # 添加量化环境变量
    export ENABLE_QUANT=1
    export QUANT_WEIGHT_PATH=${QUANT_WEIGHT_PATH}
    # 将TP_SIZE设为对应的并行数，例如单芯场景TP_SIZE=1，双芯场景TP_SIZE=2
    # 执行权重生成（单芯/多芯都要执行）
    python process_weights.py --model_path ${CHECKPOINT} --tp_size ${TP_SIZE}
    # 执行量化推理
    torchrun --nproc_per_node ${TP_SIZE} --master_port 2000 main.py --mode cli_demo --model_path ${CHECKPOINT} --tp_size ${TP_SIZE}
    ```

  - Web 交互

    ```shell
    # 安装依赖
    pip install -r web_requirements.txt
    
    # 下载 GitHub 仓库
    git clone https://github.com/THUDM/ChatGLM2-6B.git
    cd ChatGLM2-6B
    git reset --hard 921d7e9adc69020a19169d1ba4f76c2675a2dd29

    # 应用适配代码
    git apply ../web_demo.patch
    cd ..
    
    # 将 TP_SIZE 设为对应的并行数，例如单芯场景 TP_SIZE=1，双芯场景 TP_SIZE=2

    # Gradio 框架
    torchrun --nproc_per_node ${TP_SIZE} --master_port 2000 ChatGLM2-6B/web_demo.py --model_path ${CHECKPOINT} --tp_size ${TP_SIZE}
    
    # Streamlit 框架
    # ATB OpsRunner 的全局缓存暂不支持多线程，需要降低缓存级别，否则会报错
    # 0 不开启缓存，1 开启本地缓存，2 开启全局缓存，3 同时开启本地和全局缓存，默认为 3
    export ATB_OPSRUNNER_KERNEL_CACHE_TYPE=1
    torchrun --nproc_per_node ${TP_SIZE} --master_port 2000 -m streamlit run ChatGLM2-6B/web_demo2.py -- --model_path ${CHECKPOINT} --tp_size ${TP_SIZE}
    ```

- `main.py` 参数说明：

  ```shell
  --mode: 推理模式，可选单数据推理，数据集推理，性能测试以及命令行交互
  --model_path：模型权重路径
  --model：模型名称，当前仅支持chatglm2和chatglm3，默认为chatglm2
  --tp_size：张量并行数，等于使用的芯片数量
  --device：NPU设备id(可通过npu-smi info查看)，多芯场景则为NPU设备起始id，例：--device=0 --tp_size=4，则使用device：0，1，2，3
  --batch：batch大小
  --model_file：推理使用的modeling文件
  --ceval_dataset：CEval数据集路径
  --seqlen_in_pair：性能测试时需要测试的输入长度，默认为[256, 512, 1024]
  --seqlen_out_pair：性能测试时需要测试的输出长度，默认为[256, 512, 1024]
  --performance_output_file：性能测试数据保存文件，默认为performance.csv
  --print_response：是否打印性能测试的推理回答
  ```

# 模型参考精度和性能结果

- 参考精度

  > 因为 `C-Eval` 数据集test子集需要上传官网得到结果，所以这里使用val子集进行精度对比

  | ChatGLM2   | 类别 | Average Accuracy |
  | ---------- | ---- | ---------------- |
  | GPU (浮点bs8)  | val  | 53.56%           |
  | NPU (浮点bs8)  | val  | 53.12%           |

- 推理性能

  > 这里性能结果仅作为参考，并非版本极致性能优化结果。

  | 硬件形态 | 批大小 | 输入长度 | 输出长度 | 解码速度 |
  | -------- | ------ | -------- | -------- | -------- |
  | 300I Duo | 1      | 8192     | 1024     | 162ms    |