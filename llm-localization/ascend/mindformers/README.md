

```

docker pull --platform=arm64 swr.cn-central-221.ovaijisuan.com/dxy/mindspore_kernels:MindSpore2.2.10-cann7.0.0beta1_py_3.9-euler_2.8.3-32GB

```



```
docker run -it  -u root  \
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
--name mindspore_ma-4-7   \
--entrypoint=/bin/bash  \
swr.cn-central-221.ovaijisuan.com/dxy/mindspore_kernels:MindSpore2.2.10-cann7.0.0beta1_py_3.9-euler_2.8.3-32GB
```



```
docker start pytorch_ma
docker exec -it mindspore_ma-4-7 /bin/bash
```

```
import numpy as np
import mindspore.context as context
from mindspore import Tensor
from mindspore.ops import functional as F

context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")

x = Tensor(np.ones([1,3,3,4]).astype(np.float32))
y = Tensor(np.ones([1,3,3,4]).astype(np.float32))
print(F.tensor_add(x, y))

```


```
# 0-7  仅Ascend模式可用
from mindspore import context
context.set_context(device_target="Ascend", device_id=6)
```

---





```
docker pull swr.cn-central-221.ovaijisuan.com/dxy/mindspore_kernels:MindSpore2.2.10-cann7.0.0beta1_py_3.9-euler_2.8.3-64GB 
```




```

docker run -it  -u root   \
--device=/dev/davinci0   \
--device=/dev/davinci1   \
--device=/dev/davinci2   \
--device=/dev/davinci3   \
--device=/dev/davinci4   \
--device=/dev/davinci5   \
--device=/dev/davinci6   \
--device=/dev/davinci7   \
--device=/dev/davinci_manager  \
--device=/dev/devmm_svm   \
--device=/dev/hisi_hdc   \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver    \
-v /usr/local/dcmi:/usr/local/dcmi   \
-v /usr/local/Ascend/toolbox:/usr/local/Ascend/toolbox    \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi   \
-v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware    \
-v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi   \
-v /home/aicc:/home/ma-user/work/aicc    \
--name mindspore_ma_64   \
--entrypoint=/bin/bash  \
swr.cn-central-221.ovaijisuan.com/dxy/mindspore_kernels:MindSpore2.2.10-cann7.0.0beta1_py_3.9-euler_2.8.3-64GB

```








---



```
docker run -it -u root \
    --ipc=host \
    --network=host \
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
    -v /var/log/npu/:/usr/slog \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    --name mindformers_dev \
    swr.cn-central-221.ovaijisuan.com/mindformers/mindformers1.0_mindspore2.2.11:aarch_20240125 \
    /bin/bash
```




## 模型支持


- https://mindformers.readthedocs.io/zh-cn/latest/docs/model_support_list.html#llm

### translation

|     模型 <br> model     | 模型规格<br/>type | 数据集 <br> dataset | 评估指标 <br> metric | 评估得分 <br> score |                        配置<br>config                        |
| :---------------------: | ----------------- | :-----------------: | :------------------: | :-----------------: | :----------------------------------------------------------: |
| [t5](model_cards/t5.md) | t5_small          |        WMT16        |          -           |          -          | [configs](https://gitee.com/mindspore/mindformers/tree/dev/configs/t5) |



### translation

|     模型 <br> model     | 模型规格<br/>type | 数据集 <br> dataset | 评估指标 <br> metric | 评估得分 <br> score |                        配置<br>config                        |
| :---------------------: | ----------------- | :-----------------: | :------------------: | :-----------------: | :----------------------------------------------------------: |
| [t5](model_cards/t5.md) | t5_small          |        WMT16        |          -           |          -          | [configs](https://gitee.com/mindspore/mindformers/tree/dev/configs/t5) |

### [text_generation](task_cards/text_generation.md)

|                    模型 <br> model                    |                                 模型规格<br/>type                                  | 数据集 <br> dataset |             评估指标 <br> metric             |                          评估得分 <br> score                          |                                            配置<br>config                                             |
| :---------------------------------------------------: | :--------------------------------------------------------------------------------: | :-----------------: | :------------------------------------------: | :-------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------: |
|             [llama](model_cards/llama.md)             |                     llama_7b <br/>llama_13b <br/>llama_7b_lora                     |       alpaca        |                      -                       |                                   -                                   |               [configs](https://gitee.com/mindspore/mindformers/tree/dev/configs/llama)               |
|            [llama2](model_cards/llama2.md)            | llama2_7b <br/>llama2_13b <br/>llama2_7b_lora <br/>llama2_13b_lora <br/>llama2_70b |       alpaca        |                PPL / EM / F1                 | 6.58 / 39.6 / 60.5 <br/> 6.14 / 27.91 / 44.23 <br/> - <br/> - <br/> - |               [configs](https://gitee.com/mindspore/mindformers/tree/dev/configs/llama)               |
|               [glm](model_cards/glm.md)               |                               glm_6b<br/>glm_6b_lora                               |        ADGEN        | BLEU-4 / Rouge-1 / Rouge-2 / Rouge-l <br>  - |                  8.42 / 31.75 / 7.98 / 25.28 <br> -                   |                [configs](https://gitee.com/mindspore/mindformers/tree/dev/configs/glm)                |
|              [glm2](model_cards/glm2.md)              |                              glm2_6b<br/>glm2_6b_lora                              |        ADGEN        | BLEU-4 / Rouge-1 / Rouge-2 / Rouge-l <br>  - |     7.47 / 30.78 / 7.07 / 24.77 <br> 7.23 / 31.06 / 7.18 / 24.23      |               [configs](https://gitee.com/mindspore/mindformers/tree/dev/configs/glm2)                |
|              [glm3](model_cards/glm3.md)              |                                      glm3_6b                                       |        ADGEN        |                      -                       |                                   -                                   |               [configs](https://gitee.com/mindspore/mindformers/tree/dev/configs/glm3)                |
|         [CodeGeex2](model_cards/codegeex2.md)         |                                    codegeex2_6b                                    |     CodeAlpaca      |                      -                       |                                   -                                   |             [configs](https://gitee.com/mindspore/mindformers/tree/dev/configs/codegeex2)             |
|             [bloom](model_cards/bloom.md)             |                          bloom_560m<br/>bloom_7.1b <br/>                           |       alpaca        |                      -                       |                                   -                                   |               [configs](https://gitee.com/mindspore/mindformers/tree/dev/configs/bloom)               |
|              [gpt2](model_cards/gpt2.md)              |                          gpt2_small <br/> gpt2_13b <br/>                           |     wikitext-2      |                      -                       |                                   -                                   |               [configs](https://gitee.com/mindspore/mindformers/tree/dev/configs/gpt2)                |
|        [pangualpha](model_cards/pangualpha.md)        |                        pangualpha_2_6_b<br/>pangualpha_13b                         |     悟道数据集      |           TNEWS / Em / F1 <br/> -            |                     0.646 / 2.10 / 21.12 <br>   -                     |            [configs](https://gitee.com/mindspore/mindformers/tree/dev/configs/pangualpha)             |
|     [baichuan](../research/baichuan/baichuan.md)      |                           baichuan_7b <br/>baichuan_13b                            |       alpaca        |                      -                       |                                   -                                   |  [configs](https://gitee.com/mindspore/mindformers/tree/dev/research/baichuan/run_baichuan_7b.yaml)   |
|    [baichuan2](../research/baichuan2/baichuan2.md)    |  baichuan2_7b <br/>baichuan2_13b  <br/>baichuan2_7b_lora <br/>baichuan2_13b_lora   |        belle        |                      -                       |                                   -                                   |            [configs](https://gitee.com/mindspore/mindformers/tree/dev/research/baichuan2)             |
|       [skywork](../research/skywork/skywork.md)       |                                    skywork_13b                                     |        ADGEN        |            C-Eval / MMLU / CMMLU             |                         60.63 / 62.14 / 61.83                         |             [configs](https://gitee.com/mindspore/mindformers/tree/dev/research/skywork)              |
| [Wizardcoder](../research/wizardcoder/wizardcoder.md) |                                  wizardcoder_15b                                   |     CodeAlpaca      |                 MBPP Pass@1                  |                                 50.8                                  | [configs](https://gitee.com/mindspore/mindformers/tree/dev/research/wizardcoder/run_wizardcoder.yaml) |
|           [Qwen](../research/qwen/qwen.md)            |                               qwen_7b <br/>qwen_14b                                |       alpaca        |                    C-Eval                    |                            63.3 <br/>72.13                            |      [configs](https://gitee.com/mindspore/mindformers/tree/dev/research/qwen/run_qwen_7b.yaml)       |
|     [internlm](../research/internlm/internlm.md)      |                           internlm_7b <br/>internlm_20b                            |       alpaca        |                      -                       |                                   -                                   |  [configs](https://gitee.com/mindspore/mindformers/tree/dev/research/internlm/run_internlm_7b.yaml)   |
|           [ziya](../research/ziya/ziya.md)            |                                      ziya_13b                                      |       alpaca        |                      -                       |                                   -                                   |    [configs](https://gitee.com/mindspore/mindformers/tree/dev/research/baichuan/run_ziya_13b.yaml)    |








