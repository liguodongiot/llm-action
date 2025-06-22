# README

- 千问（Qwen）大语言模型是阿里巴巴集团推出的大型语言模型，具备强大的自然语言处理能力，能够理解和生成文本，能够应用于智能客服、内容生成、问答系统等多个场景，助力企业智能化升级。

# 特性矩阵

- 下表展示Qwen模型各版本支持的特性

| 模型及参数量      | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16 | Flash Attention | Paged Attention | W8A8量化 | W8A16量化 | KV cache量化 | 稀疏量化（仅支持300I DUO） | MOE量化 | MindIE Service | TGI | 长序列 | prefix_cache | FA3量化 | functioncall | Multi LoRA| W4A16量化 |
| ----------------- |----------------------------|--------------------| ---- | ---- | --------------- | --------------- | -------- | --------- | ------------ | -------- | ------- | -------------- | --- | ------ | ---------- | --- | --- | --- |---------|
| Qwen2-57B-A14B    | 支持world size 8             | ×                  | ×    | √    | ×               | √               | ×        | ×         | ×            | ×        | ×       | √              | ×   | ×      | x       | x | x | x |     x    |
| Qwen2-7B          | 支持world size 1,2,4,8       | 支持world size 2,4,8 | √    | √    | ×               | √               | √        | ×         | ×            | √        | x       | √              | ×   | ×      | x       | x | x | x |    x     |
| Qwen2-72B         | 支持world size 1,2,4,8       | 支持world size 2,4,8 | √    | √    | ×               | √               | √        | √         | √            | √        | ×       | √              | ×   | √      | √       | x | √ | √ |   x      |
| gte-Qwen2-7B      | 支持world size 1,2,4         | ×                  | √    | ×    | ×               | √               | ×        | ×         | ×            | ×        | ×       | ×              | ×   | ×      | x       | x | x | x |     x    |
| Qwen2.5-0.5B      | 支持world size 1,2,4,8       | 支持world size 2,4,8 | √    | √    | ×               | √               | ×        | ×         | ×            | ×        | ×       | ×              | ×   | ×      | x       | x | x | x |    x     |
| Qwen2.5-1.5B      | 支持world size 1,2,4,8       | 支持world size 2,4,8 | √    | √    | ×               | √               | ×        | ×         | ×            | ×        | ×       | ×              | ×   | ×      | x       | x | x | x |    x     |
| Qwen2.5-7B        | 支持world size 1,2,4,8       | 支持world size 2,4,8 | √    | √    | ×               | √               | √        | ×         | ×            | √        | ×       | √              | ×   | ×      | √       | x | √ | x |     x    |
| Qwen2.5-14B       | 支持world size 2,4,8         | 支持world size 2,4,8 | √    | √    | ×               | √               | √        | ×         | ×            | √        | ×       | √              | ×   | ×      | x       | x | √ | x |    x     |
| Qwen2.5-32B       | 支持world size 4,8           | ×                  | √    | √    | ×               | √               | √        | ×         | ×            | ×        | ×       | √              | ×   | ×      | x       | x | √ | x |     x    |
| Qwen2.5-72B       | 支持world size 8             | ×                  | √    | √    | ×               | √               | ×        | ×         | ×            | ×        | ×       | √              | ×   | ×      | x       | √ | √ | x |    √     |
| QwenCode2.5-7B          | ×       | 支持world size 2,4,8 | √    | ×    | ×               | √               | ×        | ×         | ×            | √        | x       | √              | ×   | ×      | √       | x | x | x |    x     |
| QwenCode2.5-32B          | 支持world size 4,8       |   x      |  ×   | √    | ×               | √               | ×        | ×         | ×            | x        | x       | x              | ×   | ×      | x       | x | x | x |     x    |

注：表中所示支持的world size为对话测试可跑通的配置，实际运行时还需考虑输入序列长度带来的显存占用。

- 模型支持的张量并行维度（Tensor Parallelism）可以通过查看模型的`config.json`文件中的 **KV头的数量** (`num_key_value_heads` 或者类似字段)来判断模型支持多少维度的张量并行。
> 例如 `Qwen2-0.5B` 的 `"num_key_value_heads"` 为 2，表明其只支持world size 1,2

> 例如 `Qwen2.5-32B` 的 `"num_key_value_heads"` 为 8，表明其理论支持world size 1,2,4,8（不考虑显存占用）

- qwen2/2.5系列模型在800I A2仅支持bfloat16浮点类型，300I DUO仅支持float16浮点类型。
## 开源权重
#### Qwen2
- [Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B/tree/main)
- [Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct/tree/main)
- [gte-Qwen2-7B-Instruct](https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct)
- [Qwen2-57B-A14B-Instruct](https://huggingface.co/Qwen/Qwen2-57B-A14B-Instruct/tree/main)
- [Qwen2-72B](https://huggingface.co/Qwen/Qwen2-72B/tree/main)
- [Qwen2-72B-Instruct](https://huggingface.co/Qwen/Qwen2-72B-Instruct/tree/main)
#### Qwen2.5
- [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/main)
- [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/main)
- [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/tree/main)
- [Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct/tree/main)
- [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/tree/main)
- [Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/tree/main)
#### Qwen2.5-Coder
- [Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct/tree/main)
- [Qwen2.5-Coder-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct/tree/main)

# 版本配套
下表展示运行各个系列Qwen模型所需要的transformers版本
| 模型版本 | transformers版本 |
| -------- | ---------------- |
| Qwen2    | 4.40.1及以上      |
| Qwen2.5  | 4.43.1           |
| Qwen2.5-Coder  | 4.43.1及以上          |

# Paged Attention 推理使用说明

## 推理须知：
- Qwen模型权重所在路径中的config.json文件需添加字段`torch_dtype`，例如`"torch_dtype": "float16"`
- 执行量化推理时，须在量化权重所在路径的config.json文件中添加字段`quantize`，值为当前量化权重的量化方式，例如`"quantize": "w8a8"`、`"quantize": "w8a16"`
- QWen-14B执行[2k,32k]（QWen-7B为[8k,32k]）长序列推理时需增加环境变量`LONG_SEQ_ENABLE=1`。长序列推理过程具有更多计算节点，因此相比于短序列，推理性能将有下降。
- Qwen2-7B建议采用`bf16`格式，即其权重所在路径中的config.json文件字段`torch_dtype`保持为`bfloat16`
- 300I DUO只支持`"torch_dtype": "float16"`
- 稀疏量化w8a8sc仅支持在300I DUO上使用
- 稀疏量化分为两个步骤。步骤一：w8a8s 可在任何机器上生成，注意config中需要将"torch_dtype"改为"float16"。800I A2机器上可以使用多卡进行量化生成w8a8s权重。300I DUO上仅支持单卡或cpu生成w8a8s权重。步骤二：w8a8sc 需要在300I DUO上切分。

## 路径变量解释

| 变量名称    | 含义                                                                                                                                                   |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| working_dir | 加速库及模型库下载后放置的目录                                                                                                                         |
| llm_path    | 模型仓所在路径。若使用编译好的包，则路径为`${working_dir}/MindIE-LLM/`；若使用gitee下载的代码，则路径为`${working_dir}/MindIE-LLM/examples/atb_models` |
| script_path | 脚本所在路径。QWen系列模型的工作脚本所在路径为`${llm_path}/examples/models/qwen`                                                                       |
| weight_path | 模型权重路径                                                                                                                                           |

## 权重格式转换

Paged Attention 场景需要.safetensors格式的权重，如果没有，参考[此README文件](../../README.md)转换
注：huggingface官网给出的QWen模型权重为.safetensors格式

## 量化
量化权重可通过msmodelslim（昇腾压缩加速工具）实现。

### 环境准备
环境配置可参考[此README文件](../../../README.md)

- 设置环境变量

```shell
# 设置CANN包的环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

需要安装 CANN（已包含msmodelslim工具） 以及 pytorch 和 pytorch-npu
以及相关的python库

```shell
pip install transformers  # transformers版本应根据Qwen版本确定，配套关系见‘版本配套’
pip install accelerate==0.27.2
pip install scipy==1.11.4
pip install tiktoken==0.5.2
pip install einops==0.7.0
pip install transformers_stream_generator==0.0.4
```

### 导出量化权重
#### qwen2-7b、qwen2.5-7b、qwen2.5-14b、qwen2.5-32b W8A8量化
- W8A8量化权重请使用以下指令生成
  - 当前支持NPU分布式W8A8量化
  - 执行量化脚本
    ```shell
    - 下载msmodelslim量化工具
    - 下载地址为https://gitee.com/ascend/msit/tree/master/msmodelslim
    - 根据msmodelslim量化工具readme进行相关操作
    注： 安装完cann后 需要执行source set_env.sh 声明ASCEND_HOME_PATH值 后续安装msmodelslim前需保证其不为空
    # 执行"jq --version"查看是否安装jq，若返回"bash：jq：command not found"，则依次执行"apt-get update"和"apt install jq"
    jq --version
    cd ${llm_path}
    # 指定当前机器上可用的逻辑NPU核心 通过修改convert_quant_weight.sh文件中export ASCEND_RT_VISIBLE_DEVICES值 指定使用卡号及数量
    # 7b系列使用单卡 14b 32b使用4卡 eg: ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
    vi examples/models/qwen/convert_quant_weight.sh
    # 生成量化权重
    bash examples/models/qwen/convert_quant_weight.sh -src {浮点权重路径} -dst {W8A8量化权重路径} -type qwen_w8a8
    ```

#### qwen2-7b、qwen2.5-14b、qwen2.5-7b 稀疏量化
  - Step 1
    - 修改模型权重config.json中`torch_dtype`字段为`float16`
    - 下载msmodelslim量化工具
    - 下载地址为https://gitee.com/ascend/msit/tree/master/msmodelslim
    - 根据msmodelslim量化工具readme进行相关操作
    注： 安装完cann后 需要执行source set_env.sh 声明ASCEND_HOME_PATH值 后续安装msmodelslim前需保证其不为空
    ```shell
    # 执行"jq --version"查看是否安装jq，若返回"bash：jq：command not found"，则依次执行"apt-get update"和"apt install jq"
    jq --version
    # 设置CANN包的环境变量
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    cd ${llm_path}
    # 指定当前机器上可用的逻辑NPU核心 通过修改convert_quant_weight.sh文件中export ASCEND_RT_VISIBLE_DEVICES值 指定使用卡号及数量
    # 7b系列使用单卡 14b 32b使用4卡 eg: ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
    vi examples/models/qwen/convert_quant_weight.sh
    bash examples/models/qwen/convert_quant_weight.sh -src {浮点权重路径} -dst {W8A8量化权重路径} -type qwen_w4a8
    ```

  - Step 2：量化权重切分及压缩
    ```shell
    export IGNORE_INFER_ERROR=1
    torchrun --nproc_per_node {TP数} -m examples.convert.model_slim.sparse_compressor --model_path {W8A8S量化权重路径} --save_directory {W8A8SC量化权重路径} --multiprocess_num 4
    ```
    - TP数为tensor parallel并行个数
    - 注意：若权重生成时以TP=4进行切分，则运行时也需以TP=4运行
    - 示例
      ```shell
        torchrun --nproc_per_node 2 -m examples.convert.model_slim.sparse_compressor --model_path /data1/weights/model_slim/Qwen-14b_w8a8s --save_directory /data1/weights/model_slim/Qwen-14b_w8a8sc
      ```

#### Qwen2-72B W8A16量化
- 假设当前位于`${llm_path}`目录下（安装的默认路径为`/usr/local/Ascend/llm_model`）
- 目录`examples/models/qwen/`下的`quant_qwen2_w8a16_fast.py`为Qwen2-72B-W8A16模型已配置好的较优的量化策略。导出量化权重时可直接使用，也可修改为其它策略。
- 通过 `${llm_path}/examples/models/qwen/convert_quant_weight.sh` 脚本导出Qwen2-72B模型W8A16的量化权重（注意量化权重不要和浮点权重放在同一个目录下）。命令如下：
    ```shell
    cd ${llm_path}
    bash examples/models/qwen/convert_quant_weight.sh -src ${浮点权重路径} -dst ${量化权重保存路径} -type qwen_w8a16

    ```
    例：
    ```shell
    bash examples/models/qwen/convert_quant_weight.sh -src /data1/models/Qwen2_72B -dst /data1/models/Qwen2_72B_W8A16 -type qwen_w8a16

    ```
- 导出量化权重后生成`quant_model_weight_w8a16.safetensors`和`quant_model_description_w8a16.json`两个文件。模型浮点权重中的其他文件（除safetensors文件外）需要手工拷贝到目标量化文件夹中。
- 在量化权重保存路径中的config.json文件中添加"quantize"字段。对于W8A16量化，"quantize"字段的值为"w8a16"。
- 在量化权重保存路径中的config.json文件中添加"quantization"字段，其值为'{"group_size": 0}'
  - "group_size"为0时代表W8A16使用的是per channel量化

#### Qwen2-72B KV Cache量化
- 假设当前位于`${llm_path}`目录下（安装的默认路径为`/usr/local/Ascend/llm_model`）
- 使用下列命令进行kv-int8量化权重导出：
```shell
bash examples/models/qwen/convert_quant_weight.sh -src ${浮点权重路径} -dst ${量化权重保存路径} -type qwen2_72b_w8a8c8 -device_type npu -use_devices 0,1,2,3,4,5,6,7
```
- 与Qwen2-72B W8A16量化不同，量化脚本已经替用户按要求修改好了config.json，用户无需再修改

#### Qwen2-72B W8A8量化
- 假设当前位于`${llm_path}`目录下（安装的默认路径为`/usr/local/Ascend/llm_model`）
- 使用下列命令进行W8A8量化权重导出：
  - Step 1
    - 下载msmodelslim量化工具
    - 下载地址为https://gitee.com/ascend/msit/tree/master/msmodelslim
    - 根据msmodelslim量化工具readme进行相关操作
    注： 安装完cann后 需要执行source set_env.sh 声明ASCEND_HOME_PATH值 后续安装msmodelslim前需保证其不为空
```shell
bash examples/models/qwen/convert_quant_weight.sh -src ${浮点权重路径} -dst ${量化权重保存路径} -type qwen2_72b_w8a8 -device_type npu -use_devices 0,1,2,3,4,5,6,7
```
- 与qwen2-72B W8A16量化不同，量化脚本已经替用户按要求修改好了config.json，用户无需再修改

#### Qwen2-72B W8A8稀疏量化

- 假设当前位于`${llm_path}`目录下（安装的默认路径为`/usr/local/Ascend/llm_model`）
- 使用下列命令进行W8A8稀疏量化权重导出：

  - Step 1
    - 修改模型权重config.json中`torch_dtype`字段为`float16`
    - 下载msmodelslim量化工具
    - 下载地址为https://gitee.com/ascend/msit/tree/master/msmodelslim
    - 根据msmodelslim量化工具readme进行相关操作
    ```shell
    bash examples/models/qwen/convert_quant_weight.sh -src ${浮点权重路径} -dst ${W8A8SC量化权重路径} -type qwen2_72b_w8a8s -device_type npu -use_devices 0,1,2,3,4,5,6,7
    ```
  - Step 2：切分及压缩权重
    ```shell
    export IGNORE_INFER_ERROR=1
    torchrun --nproc_per_node {TP数} -m examples.convert.model_slim.sparse_compressor --model_path {W8A8S量化权重路径} --save_directory {W8A8SC量化权重路径} --multiprocess_num 4
    ```
    - TP数为tensor parallel并行个数
    - 注意：若权重生成时以TP=4进行切分，则运行时也需以TP=4运行
    - multiprocess_num必须设置为4以减小机器压力
    - 示例
    ```shell
      torchrun --nproc_per_node 2 -m examples.convert.model_slim.sparse_compressor --model_path /data1/weights/model_slim/Qwen-14b_w8a8s --save_directory /data1/weights/model_slim/Qwen-14b_w8a8sc --multiprocess_num 4
    ```
  - 与qwen2-72B W8A16量化不同，量化脚本已经替用户按要求修改了config.json，用户只需要将config.json中的quant_type字段修改为"w8a8s"即可。

#### Qwen2.5-72B FA3量化
  - 下载msmodelslim量化工具
  - 下载地址为https://gitee.com/ascend/msit/tree/master/msmodelslim
  - 根据msmodelslim量化工具readme进行相关操作
  - 阅读链接中的readme文件生成权重，或者直接问msModelSlim团队索要：
  https://gitee.com/ascend/msit/blob/master/msmodelslim/docs/FA%E9%87%8F%E5%8C%96%E4%BD%BF%E7%94%A8%E8%AF%B4%E6%98%8E.md
  - ModelSlim团队会提供`quant_model_description_w8a8.json`和`quant_model_weight_w8a8.safetensors`两个文件。

  - 通过 `${llm_path}/examples/models/qwen/convert_quant_weight.sh` 脚本导出Qwen2.5-72B模型FA3的量化权重（注意量化权重不要和浮点权重放在同一个目录下）。命令如下：
  ```shell
    bash examples/models/qwen/convert_quant_weight.sh -src ${浮点权重路径} -dst ${fa3量化权重路径} -type qwen2p5_fa3 -msmodelslim_path ${msmodelslim工具路径}
  ```
  - ${msmodelslim工具路径}为下载目录/msit/msmodelslim，例如在/home目录下面下载的msmodelslim工具，则实际路径为：/home/msit/msmodelslim

  - 示例
  ```shell
    bash examples/models/qwen/convert_quant_weight.sh -src /opt/models/Qwen2.5-72B-Instruct/ -dst /opt/models/Qwen2.5-72B-fa3 -type qwen2p5_fa3 -msmodelslim_path /home/msit/msmodelslim
  ```
  - 新版本的msmodelslim工具如果需要添加`anti_calib_file`参数，可以在上述命令中加入`-fa3_use_anti_calib True`
  - 示例
  ```shell
    bash examples/models/qwen/convert_quant_weight.sh -src /opt/models/Qwen2.5-72B-Instruct/ -dst /opt/models/Qwen2.5-72B-fa3 -type qwen2p5_fa3 -msmodelslim_path /home/msit/msmodelslim -fa3_use_anti_calib True
  ```

  - 模型浮点权重中的其他文件（除safetensors文件外）需要手工拷贝到目标量化文件夹中。
  - 拷贝好之后，用户需在`config.json`文件中手动添加以下两个字段：
  ```json
      "quantize": "w8a8",
      "quantization_config": {"fa_quant_type": "FAQuant"}
  ```

#### Qwen2.5-72B W4A16量化
- W8A8量化权重请使用以下指令生成
  - 当前支持NPU分布式W8A8量化
  - 执行量化脚本
    ```shell
    - 下载msmodelslim量化工具
    - 下载地址为https://gitee.com/ascend/msit/tree/master/msmodelslim
    - 根据msmodelslim量化工具readme进行相关操作
    注： 安装完cann后 需要执行source set_env.sh 声明ASCEND_HOME_PATH值 后续安装msmodelslim前需保证其不为空
    # 执行"jq --version"查看是否安装jq，若返回"bash：jq：command not found"，则依次执行"apt-get update"和"apt install jq"
    jq --version
    cd ${llm_path}
    # 指定当前机器上可用的逻辑NPU核心 通过修改convert_quant_weight.sh文件中export ASCEND_RT_VISIBLE_DEVICES值 指定使用卡号及数量
    # 72B使用8卡 eg: ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    vi examples/models/qwen/convert_quant_weight.sh
    # 生成量化权重
    bash examples/models/qwen/convert_quant_weight.sh -src {浮点权重路径} -dst {W8A8量化权重路径} -type qwen_w4a16
    ```

#### Qwen2.5-Coder-7B 稀疏量化
  - Step 1
    - 修改模型权重config.json中`torch_dtype`字段为`float16`
    - 下载msmodelslim量化工具
    - 下载地址为https://gitee.com/ascend/msit/tree/master/msmodelslim
    - 根据msmodelslim量化工具readme进行相关操作
    ```shell
    # 执行"jq --version"查看是否安装jq，若返回"bash：jq：command not found"，则依次执行"apt-get update"和"apt install jq"
    jq --version
    # 设置CANN包的环境变量
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    cd ${llm_path}
    # Qwen2.5-Coder-7B加载到cpu上生成量化权重 注释掉convert_quant_weight.sh里export ASCEND_RT_VISIBLE_DEVICES这一行
    vi examples/models/qwen/convert_quant_weight.sh
    bash examples/models/qwen/convert_quant_weight.sh -src {浮点权重路径} -dst {W8A8量化权重路径} -type qwencode_w8a8s -device_type cpu
    ```
    稀疏量化后的"quantize"类型为w8a8s

  - Step 2：量化权重切分及压缩

    ```shell
    export IGNORE_INFER_ERROR=1
    torchrun --nproc_per_node {TP数} -m examples.convert.model_slim.sparse_compressor --model_path {W8A8S量化权重路径} --save_directory {W8A8SC量化权重路径}
    ```
    - TP数为tensor parallel并行个数
    - 注意：若权重生成时以TP=4进行切分，则运行时也需以TP=4运行
    - 示例
    ```shell
      torchrun --nproc_per_node 4 -m examples.convert.model_slim.sparse_compressor --model_path /data/Qwen2.5-Coder-7B-w8a8s --save_directory /data/Qwen2.5-Coder-7B-w8a8sc
    ```

#### Qwen2.5-14B Qwen2.5-72B pdmix W8A8C8量化

- 通过 `${llm_path}/examples/models/qwen/convert_quant_weight.sh` 脚本导出Qwen2.5-14B和Qwen2.5-72B模型pdmix W8A8C8的量化权重（注意量化权重不要和浮点权重放在同一个目录下）。命令如下：
- 假设当前位于`${llm_path}`目录下（安装的默认路径为`/usr/local/Ascend/atb-models`），`trust_remote_code`为可选参数代表是否信任本地的可执行文件，传入该参数代表信任本地可执行文件
    ```shell
    bash examples/models/qwen/convert_quant_weight.sh -src ${浮点权重路径} -dst ${量化权重保存路径} -type ${type} -device_type npu -use_devices 0,1,2,3,4,5,6,7 -msmodelslim_path ${msmodelslim工具路径} -trust_remote_code
    ```
    - ${msmodelslim工具路径}为下载目录/msit/msmodelslim，例如在/home目录下面下载的msmodelslim工具，则实际路径为：/home/msit/msmodelslim

  Qwen2.5-72B示例：
    ```shell
    bash examples/models/qwen/convert_quant_weight.sh -src /data/Qwen2.5-72B-Instruct/ -dst /data/qwen2.5-72B-pdmix-w8a8c8/ -type qwen2p5_72b_w8a8c8_pdmix -device_type npu -use_devices 0,1,2,3,4,5,6,7 -msmodelslim_path /home/msit/msmodelslim -trust_remote_code

    ```
  Qwen2.5-14B示例：
    ```shell
    bash examples/models/qwen/convert_quant_weight.sh -src /data/Qwen2.5-14B-Instruct/ -dst /data/qwen2.5-14B-pdmix-w8a8c8/ -type qwen2p5_14b_w8a8c8_pdmix -device_type npu -use_devices 0,1,2,3,4,5,6,7 -msmodelslim_path /home/msit/msmodelslim -trust_remote_code

    ```

## 推理

### 对话测试

量化权重生成路径下可能缺少一些必要文件（与转换量化权重时使用的cann版本有关），若启动量化推理失败，请将config.json等相关文件复制到量化权重路径中，可执行以下指令进行复制：
```shell
cp ${浮点权重路径}/*.py ${量化权重路径}
cp ${浮点权重路径}/*.json ${量化权重路径}
cp ${浮点权重路径}/*.tiktoken ${量化权重路径}
```

启动量化推理时，请在权重路径的config.json文件中添加(或修改)`torch_dtype`字段，例如`"torch_dtype": "float16"`。

启动量化推理时，请在权重路径的config.json文件中添加(或修改)`quantize`字段，值为相应量化方式，例如`"quantize": "w8a8"`、`"quantize": "w8a16"`

在`${llm_path}`目录执行以下指令

```shell
bash examples/models/qwen/run_pa.sh -m ${weight_path} --trust_remote_code true
```

注：

1.推理支持浮点和量化，若启动浮点推理则在`${weight_path}`中传入浮点权重路径，若启动量化则传入量化权重路径

2.--trust_remote_code为可选参数代表是否信任本地的可执行文件，默认false。传入true，则代表信任本地可执行文件，-r为其缩写

3.同时支持Qwen和Qwen1.5模型推理，若启动Qwen模型推理时在`${weight_path}`中传入Qwen权重路径，若启动Qwen1.5模型推理时则在`${weight_path}`中传入Qwen1.5权重路径

4.Qwen系列chat模型需要开启chat模式才能正常输出。
执行：

```shell
bash examples/models/qwen/run_pa.sh -m ${weight_path} --trust_remote_code true -c true
```

5.对于embedding类模型，例如gte-Qwen2-7B-Instruct时，运行命令如下：
```shell
bash examples/models/qwen/run_pa.sh -m ${weight_path} -e true
```

6.启动qwen需要安装三方依赖tiktoken，若环境中没有该依赖可使用以下命令安装：

```shell
pip install tiktoken
```
**运行Multi-Lora**
- 下载Lora权重：Lora权重中需包含至少一个safetensors格式的文件，和一个名为`adapter_config.json`的配置文件
- 在基础模型的权重文件夹中，新增`lora_adapter.json`文件，内容为需要预加载的Lora权重，例如：
    ```json
    {
      "qwen_lora1": "/home/data/lora/Qwen1.5-14b-chat/adapter1",
      "qwen_lora2": "/home/data/lora/Qwen1.5-14b-chat/adapter2"
    }
    ```
- 进行推理时需指定每个请求所使用的adapter权重，默认仅使用基础模型权重
- 运行示例
    ```shell
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    torchrun --nproc_per_node 8 --master_port 20030 -m examples.run_pa --model_path /data1/models/qwen2/Qwen1.5-14b-chat --is_chat_model --max_output_length 256 --max_batch_size 2 --input_dict '[{"prompt": "What is deep learning?", "adapter": "qwen_lora1"}, {"prompt": "What is deep learning?"}]'
    ```
- 约束与限制
    - 仅支持在Atlas 800I A2上运行
    - Lora权重不支持热加载，如果未获取到`adapter_id`，将会默认使用`base`
    - 仅支持浮点模型
    - `lora_adapter.json`文件中的键就是`input_dict`参数的键`adapter`的值，也叫`adapter_id`。
    - `adapter_id`唯一 且 不能与字符串`base`重名
    - 在显存充足的情况下至多加载10个Lora权重
    - **用于精度测试的`lora_data.jsonl`文件包含的`adapter_id`数量必须比数据集的数量多，否则多余的数据会默认使用base**

### run_pa.sh 参数说明（需要到脚本中修改）
根据硬件设备不同请参考下表修改run_pa.sh再运行

| 参数名称                  | 含义                                      | 800I A2推荐值    | 300I DUO推荐值   |
| ------------------------- | ----------------------------------------- | ---------------- | ---------------- |
| BIND_CPU                  | 绑定CPU核心开关,默认进行绑核              | 1                | 1                |
| ASCEND_RT_VISIBLE_DEVICES | 使用的硬件卡号，多个卡间使用逗号相连      | 根据实际情况设置 | 根据实际情况设置 |
| RESERVED_MEMORY_GB        | 保留内存，通常未加速库需要的内存+通信内存 | 3                | 3                |
| MASTER_PORT               | 卡间通信端口,通常不用修改，有冲突时再改   |                  |                  |

注：暂不支持奇数卡并行

## 精度测试

- 参考[此README文件](../../../tests/modeltest/README.md)

示例：

```shell
bash run.sh pa_fp16 full_BoolQ 1 qwen /data1/models/qwen2/qwen_quant_test/ 8
bash run.sh pa_fp16 full_BoolQ 1 qwen ${Qwen-7b权重路径} 8
bash run.sh pa_fp16 full_BoolQ 1 qwen ${Qwen-14b权重路径} 8
bash run.sh pa_fp16 full_BoolQ 1 qwen ${Qwen-72b权重路径} 8
bash run.sh pa_fp16 full_BoolQ 1 qwen ${Qwen1.5-14b权重路径} 8
bash run.sh pa_fp16 full_BoolQ 1 qwen ${Qwen-14b-chat权重路径} 8
bash run.sh pa_fp16 full_BoolQ 1 qwen ${Qwen-72b-chat权重路径} 8
bash run.sh pa_fp16 full_BoolQ 1 qwen ${Qwen1.5-0.5b-chat权重路径} 8
bash run.sh pa_fp16 full_BoolQ 1 qwen ${Qwen1.5-4b-chat权重路径} 4
bash run.sh pa_fp16 full_BoolQ 1 qwen ${Qwen1.5-7b权重路径} 8
bash run.sh pa_fp16 full_BoolQ 1 qwen ${Qwen1.5-14b-chat权重路径} 8
bash run.sh pa_fp16 full_BoolQ 1 qwen ${Qwen1.5-32b-chat权重路径} 8
bash run.sh pa_fp16 full_BoolQ 1 qwen ${Qwen1.5-72b权重路径} 8
bash run.sh pa_fp16 full_BoolQ 1 qwen ${Qwen1.5-MoE-A2.7B-Chat权重路径} 8
bash run.sh pa_fp16 full_HumanEval_X 1 qwen ${Qwen2.5-Coder-7B权重路径} 8
```
- gte_qwen测试
    - 依赖安装
     C_MTEB、optimum、tqdm、datasets、faiss-cpu

     C_MTEB 需要安装依赖 pytrec-eval，安装该依赖时需要请求 https://github.com/usnistgov/trec_eval/archive/v9.0.8.tar.gz 时发生SSL证书错误
     解决方案
     手动下载 pytrec_eval并上传至服务器或在服务器上执行
     ```shell
     wget https://files.pythonhosted.org/packages/2e/03/e6e84df6a7c1265579ab26bbe30ff7f8c22745aa77e0799bba471c0a3a19/pytrec_eval-0.5.tar.gz
     tar -xzvf pytrec_eval-0.5.tar.gz
     cd pytrec_eval-0.5
     ```
     修改 pytrec_eval-0.5/setup.py，在开头处增加
     ```shell
     import ssl
     ssl._create_default_https_context = ssl._create_unverified_context
     ```
     安装
     ```shell
     tar -zcvf pytrec_eval-0.5.tar.gz pytrec_eval-0.5
     pip install pytrec_eval-0.5.tar.gz
     ```
     同理安装依赖 pytrec-eval
     ```shell
     wget http://mirrors.aliyun.com/pypi/packages/dc/61/9003ffdb64f74a91208d69235dbcd380ae1a8d267089348eb8f7aab9819a/pytrec_eval_terrier-0.5.7.tar.gz
     tar -xzvf pytrec_eval_terrier-0.5.7.tar.gz
     cd pytrec_eval_terrier-0.5.7
     ```
     修改 pytrec_eval_terrier-0.5.7/setup.py，在开头处增加
     ```shell
     import ssl
     ssl._create_default_https_context = ssl._create_unverified_context
     tar -zcvf pytrec_eval_terrier-0.5.7.tar.gz pytrec_eval_terrier-0.5.7
     pip install pytrec_eval_terrier-0.5.7.tar.gz
     ```
     安装
     ```shell
     tar -zcvf pytrec_eval_terrier-0.5.7.tar.gz pytrec_eval_terrier-0.5.7
     pip install pytrec_eval_terrier-0.5.7.tar.gz
     ```
    - 获取测试数据集
     ```shell
     mkdir dataset
     ```
     下载数据集文件 [corpus、queries](https://huggingface.co/datasets/C-MTEB/T2Retrieval/tree/main/data) 及 [dev](https://huggingface.co/datasets/C-MTEB/T2Retrieval-qrels/tree/main/data) 至 `dataset` 目录中
    - 修改embedding输出存储位置
    ```shell
    vim ../../../atb_llm/models/qwen2/flash_causal_qwen2_gte.py
    ```
    将其中233行修改为 logits_name = f"embedding_tensor_0"
    - 运行指令
     ```shell
     python eval_t2retrieval_gte_npu/gpu.py --model_type_or_path model_type_or_path --batch_size batch_size --device device
     ```
     结果保存在当前路径results/

## 性能测试

- 进入以下路径
  ```shell
  ${llm_path}/tests/modeltest
  ```
- 运行指令
  ```shell
  bash run.sh pa_fp16 [performance|full_CEval|full_BoolQ] ([case_pair]) [batch_size] qwen [weight_dir] [chip_num] ([max_position_embedding/max_sequence_length])
  ```

- 环境变量释义

1. HCCL_DETERMINISTIC=false          LCCL_DETERMINISTIC=0

这两个会影响性能，开启了变慢，但是会变成确定性计算，不开会变快，所以设置为0。

2. HCCL_BUFFSIZE=120

这个会影响hccl显存，需要设置，基本不影响性能。

3. ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1

这个是显存优化，需要开，小batch、短序列场景不开更好。

示例：

  ```shell
  HCCL_DETERMINISTIC=false LCCL_DETERMINISTIC=0 HCCL_BUFFSIZE=120 ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1 bash run.sh pa_fp16 performance　[[2048,2048],[1024,1024],[512,512],[256,256]] 1 qwen ${Qwen-7b权重路径} 8
  HCCL_DETERMINISTIC=false LCCL_DETERMINISTIC=0 HCCL_BUFFSIZE=120 ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1 bash run.sh pa_fp16 performance　[[2048,2048],[1024,1024],[512,512],[256,256]] 1 qwen ${Qwen-14b权重路径} 8
  HCCL_DETERMINISTIC=false LCCL_DETERMINISTIC=0 HCCL_BUFFSIZE=120 ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1 bash run.sh pa_fp16 performance　[[2048,2048],[1024,1024],[512,512],[256,256]] 1 qwen ${Qwen-72b权重路径} 8
  HCCL_DETERMINISTIC=false LCCL_DETERMINISTIC=0 HCCL_BUFFSIZE=120 ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1 bash run.sh pa_fp16 performance　[[2048,2048],[1024,1024],[512,512],[256,256]] 1 qwen ${Qwen1.5-14b权重路径} 8
  HCCL_DETERMINISTIC=false LCCL_DETERMINISTIC=0 HCCL_BUFFSIZE=120 ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1 bash run.sh pa_fp16 performance　[[2048,2048],[1024,1024],[512,512],[256,256]] 1 qwen ${Qwen-14b-chat权重路径} 8
  HCCL_DETERMINISTIC=false LCCL_DETERMINISTIC=0 HCCL_BUFFSIZE=120 ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1 bash run.sh pa_fp16 performance　[[2048,2048],[1024,1024],[512,512],[256,256]] 1 qwen ${Qwen-72b-chat权重路径} 8
  HCCL_DETERMINISTIC=0 LCCL_DETERMINISTIC=0 HCCL_BUFFSIZE=120
  ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1 bash run.sh pa_fp16 performance　[[2048,2048],[1024,1024],[512,512],[256,256]] 1 qwen ${Qwen1.5-0.5b-chat权重路径} 8
  HCCL_DETERMINISTIC=false LCCL_DETERMINISTIC=0 HCCL_BUFFSIZE=120
  ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1 bash run.sh pa_fp16 performance　[[2048,2048],[1024,1024],[512,512],[256,256]] 1 qwen ${Qwen1.5-4b-chat权重路径} 8
  HCCL_DETERMINISTIC=false LCCL_DETERMINISTIC=0 HCCL_BUFFSIZE=120 ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1 bash run.sh pa_fp16 performance　[[2048,2048],[1024,1024],[512,512],[256,256]] 1 qwen ${Qwen1.5-7b权重路径} 8
  HCCL_DETERMINISTIC=false LCCL_DETERMINISTIC=0 HCCL_BUFFSIZE=120 ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1 bash run.sh pa_fp16 performance　[[2048,2048],[1024,1024],[512,512],[256,256]] 1 qwen ${Qwen1.5-14b-chat权重路径} 8
  HCCL_DETERMINISTIC=false LCCL_DETERMINISTIC=0 HCCL_BUFFSIZE=120 ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1 bash run.sh pa_fp16 performance　[[2048,2048],[1024,1024],[512,512],[256,256]] 1 qwen ${Qwen1.5-32b-chat权重路径} 8
  HCCL_DETERMINISTIC=false LCCL_DETERMINISTIC=0 HCCL_BUFFSIZE=120 ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1 bash run.sh pa_fp16 performance　[[2048,2048],[1024,1024],[512,512],[256,256]] 1 qwen ${Qwen1.5-72b权重路径} 8
  HCCL_DETERMINISTIC=false LCCL_DETERMINISTIC=0 HCCL_BUFFSIZE=120 ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1 bash run.sh pa_fp16 performance　[[2048,2048],[1024,1024],[512,512],[256,256]] 1 qwen ${Qwen1.5-MoE-A2.7B-Chat权重路径} 8
  ```

- 参考[此README文件](../../../tests/modeltest/README.md)
## prefix_cache
- 参考[此README文件](../../../../../mindie_llm/text_generator/plugins/prefix_cache)
目前此特性仅支持qwen2-72b fp16使用
# Flash Attention推理使用说明

路径变量和权重转换等均与Paged Attention相同。

## 推理

### 对话测试

在`${llm_path}`目录执行以下指令

```shell
bash examples/models/qwen/run_fa.sh -m ${weight_path}
```

# 使用虚拟机运行Qwen（包含Qwen系列，Qwen1.5系列，Qwen2系列，Qwen2.5系列）模型
如果在虚拟机内运行Qwen模型，且虚拟机所在的物理机支持HCCS通信，需引入下列环境变量：
```shell
export NPU_VM_SUPPORT_HCCS = 1
```

# Qwen长序列推理
qwen长序列推理需要在qwen权重(config.json)中将use_dynamic_ntk参数与use_logn_attn同时设置成True。如Qwen-14B-Chat：
注意：如果不使用长序列推理，请将use_dynamic_ntk与use_logn_attn参数同时设置成False。

```json
{
  "architectures": [
    "QwenLMHeadModel"
  ],
  // ...
  "use_dynamic_ntk": true,
  // ...
  "use_logn_attn": true,
  // ...
}
```
qwen2长序列推理需要在qwen2权重(config.json)中新增rope_scaling参数。如Qwen2-72B-Instruct：
注意：如果不使用长序列推理，请不要添加。
```json
{
  "architectures": [
    "Qwen2ForCausalLM"
  ],
  // ...
  "vocab_size": 152064,

  // adding the following snippets
  "rope_scaling": {
    "factor": 4.0,
    "original_max_position_embeddings": 32768,
    "type": "yarn"
  }
}
注：
- 除启动命令外，其他操作与执行PA相同
- qwen qwen1.5暂不支持bf16格式，请将权重路径下config.json文件的`torch_dtype`字段修改为`float16`
- 暂不支持chat模式。部分chat模型输出可能存在异常，如qwen1.5-32b-chat，若出现上述情况，请优先使用PA
- 长序列推理过程具有更多计算节点，因此相比于短序列，推理性能将有下降。
- qwen1.5部分Chat模型(4B、32B)fa暂不支持chat推理，请优先使用pa。如需使用fa请将输入改造成续写的样式，如：`What's deep learning?`改写成`Deep learning is`
- Qwen2 Qwen2.5系列模型当前800I A2采用bf16， 300I DUO使用fp16
