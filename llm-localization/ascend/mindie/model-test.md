# ModelTest README

ModelTest为大模型的性能和精度提供测试功能。

目前支持：

1. NPU，PA场景，性能/精度测试，float16
2. GPU，FA场景，精度测试，float16

功能：

1. 性能测试：指定batch，指定输入输出长度的e2e性能、吞吐，首Token以及非首Token性能，吞吐。
2. 精度测试：CEval, MMLU, BoolQ, HumanEval下游数据集

PA模型支持：

1. Llama (Llama-7B, Llama-13B, Llama-65B, Llama2-7B, Llama2-13B, Llama2-70B)
2. Starcoder-15.5B
3. Chatglm2-6B
4. CodegeeX2-6B
5. Baichuan2 (Baichuan2-7B, Baichuan2-13B)
6. Qwen (Qwen-14B, Qwen-72B)
7. Aquila (Aquila-7B)
8. Deepseek (Deepseek16B)
9. Mixtral (Mixtral8 * 7B)
10. Bloom-7B
11. Baichuan1 (Baichuan1-7B, Baichuan1-13B)
12. CodeLlama (CodeLlama-13B)
13. Yi (Yi-6B-200K, Yi-34B)
14. Chinese Alpaca (Chinese-Alpaca-13B)

# 使用说明

### 环境变量

```shell
# source cann环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# source 加速库环境变量
source /usr/local/Ascend/atb/set_env.sh
# source 模型仓tar包解压出来后的环境变量
source set_env.sh
# 设置ATB_TESTDATA环境变量
export ATB_TESTDATA="[path]" # 用于存放测试结果的路径
# 设置使用卡号
export ASCEND_RT_VISIBLE_DEVICES="[卡号]" # NPU场景，如"0,1,2,3,4,5,6,7"
或
export CUDA_VISIBLE_DEVICES="[卡号]" # GPU场景，如"0,1,2,3,4,5,6,7"
```

### 安装python依赖

```
pip install -r requirements.txt
```

### 运行指令

```
# NPU
bash run.sh pa_fp16 [performance|full_CEval|full_MMLU|full_BoolQ|full_HumanEval] ([case_pair]) [batch_size] [model_name] ([use_refactor]) [weight_dir] [chip_num] ([max_position_embedding/max_sequence_length])
或
# GPU
bash run.sh fa [full_CEval|full_MMLU|full_BoolQ|full_HumanEval] [batch_size] [model_name] ([use_refactor]) [weight_dir] [chip_num]

说明:
1. case_pair只在performance场景下接受输入，接收一组或多组输入，格式为[[seq_in_1,seq_out_1],...,[seq_in_n,seq_out_n]], 如[[256,256],[512,512]]
2. model_name:
    Llama-65B, Llama2-7B, Llama2-13B, Llama2-70B: llama
    CodeLlama-13B, Chinese-Alpaca-13B, Yi-6B-200K, Yi-34B: llama
    Starcoder-15.5B: starcoder
    Chatglm2-6B: chatglm2_6b
    CodegeeX2-6B: codegeex2_6b
    Baichuan2-7B: baichuan2_7b
    Baichuan2-13B: baichuan2_13b
    Qwen-14b, Qwen-72b: qwen
    Aquila-7B: aquila_7b
    Deepseek16B: deepseek
    Mixtral8 * 7B: mixtral
    Bloom-7B: bloom_7b
    Baichuan1-7B: baichuan2_7b
    Baichuan1-13B: baichuan2_13b
3. 当model_name为llama时，须指定use_refactor为True或者False（统一使用True）
4. weight_dir: 权重路径
5. chip_num: 使用的卡数
6. max_position_embedding: 可选参数，不传入则使用config中的默认配置
7. 运行完成后，会在控制台末尾呈现保存数据的文件夹

举例：
1. 测试Llama-70B在8卡[512, 512]场景下，16 batch的性能，使用归一代码
bash run.sh pa_fp16 performance [[512,512]] 16 llama True /path 8
1. 测试Starcoder-15.5B在8卡1 batch下游数据集BoolQ
bash run.sh pa_fp16 full_BoolQ 1 starcoder /path 8
``` 

## startcoder 特别运行操作说明

- 对于300I DUO设置环境变量，修改core/starcoder.py中prepare_environ函数。

```shell
os.environ['ATB_LAUNCH_KERNEL_WITH_TILING'] = "1"
os.environ['LCCL_ENABLE_FALLBACK'] = "0"
```

## baichuan2-13b 特别运行操作说明

- 对于300I DUO设置环境变量，修改core/baichuan2_13b_test.py中prepare_environ函数。

```shell
os.environ['ATB_OPERATION_EXECUTE_ASYNC'] = "0"
os.environ['TASK_QUEUE_ENABLE'] = "0"
``
