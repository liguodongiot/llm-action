


## PaddleNLP

- https://hub.docker.com/r/paddlecloud/paddlenlp
- https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm


- pytorch转paddle: https://github.com/PaddlePaddle/PaddleNLP/blob/v2.6.1/docs/community/contribute_models/convert_pytorch_to_paddle.rst



```
pip install --upgrade paddlenlp==2.6.1 -i https://pypi.org/simple



sudo pip install --pre --upgrade paddlenlp -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html
```


```
docker run --name dev \
--runtime=nvidia \
-v $PWD:/mnt \
-p 8888:8888 \
-it \
paddlecloud/paddlenlp:develop-gpu-cuda10.2-cudnn7-cdd682 \
/bin/bash
```





---












## 支持的模型

```
bigscience/bloom-560m
bigscience/bloomz-560m/
```


### bloom

```
https://bj.bcebos.com/paddlenlp/models/community/bigscience/bloomz-560m/tokenizer_config.json

```



```
from paddlenlp.transformers.bloom.tokenizer import BloomTokenizer
tokenizer = BloomTokenizer.from_pretrained("bigscience/bloomz-560m")
```





## 推理

```
import paddlenlp
from pprint import pprint
from paddlenlp import Taskflow
schema = ['时间', '选手', '赛事名称'] # Define the schema for entity extraction
ie = Taskflow('information_extraction', schema=schema)
pprint(ie("2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！"))
```



```
https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base_v1.1/model_state.pdparams

/Users/liguodong/.paddlenlp/taskflow/information_extraction/uie-base/model_state.pdparams

```

---


```
from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")

model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m", dtype="float32")
input_features = tokenizer("你好！请自我介绍一下。", return_tensors="pd")
outputs = model.generate(**input_features, max_length=128)
tokenizer.batch_decode(outputs[0])




tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m", from_hf_hub=True)


tokenizer = AutoTokenizer.from_pretrained("ziqingyang/chinese-llama-7b", from_hf_hub=True)





model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m", dtype="float32",from_aistudio = False, from_hf_hub=True, convert_from_torch=True)


tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m", from_hf_hub=True)

```


### 动态图推理

```
# 预训练&SFT动态图模型推理
python predictor.py \
    --model_name_or_path meta-llama/Llama-2-7b-chat \
    --batch_size 1 \
    --data_file ./data/dev.json \
    --dtype "float16" \
    --mode "dynamic"
```


### 静态图推理



```
# 首先需要运行一下命令将动态图导出为静态图
# LoRA需要先合并参数，详见3.7LoRA参数合并
# Prefix Tuning暂不支持
python export_model.py \
    --model_name_or_path meta-llama/Llama-2-7b-chat \
    --output_path ./inference \
    --dtype float16


# 静态图模型推理
python predictor.py \
    --model_name_or_path inference \
    --batch_size 1 \
    --data_file ./data/dev.json \
    --dtype "float16" \
    --mode "static"
```


## Flask & Gradio UI服务化部署

```
python -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" flask_server.py \
    --model_name_or_path meta-llama/Llama-2-7b-chat \
    --port 8010 \
    --flask_port 8011 \
    --src_length 1024 \
    --dtype "float16"
```




## 量化


量化算法可以将模型权重和激活转为更低比特数值类型表示，能够有效减少显存占用和计算开销。

下面我们提供GPTQ和PaddleSlim自研的PTQ策略，分别实现WINT4和W8A8量化。

- https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/tutorials/quant/advanced_quantization.md


```


https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html

python -m pip install paddlepaddle-gpu==0.0.0.post117 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html


安装发布版本：

pip install paddleslim
安装develop版本：

git clone https://github.com/PaddlePaddle/PaddleSlim.git & cd PaddleSlim
python setup.py install

验证安装：安装完成后您可以使用 python 或 python3 进入 python 解释器，输入import paddleslim, 没有报错则说明安装成功。
```




```
# PTQ 量化
python  finetune_generation.py ./llama/ptq_argument.json

# GPTQ 量化
python  finetune_generation.py ./llama/gptq_argument.json

```


