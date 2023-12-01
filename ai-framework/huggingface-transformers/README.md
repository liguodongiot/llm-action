

## 量化

transformers 已经集成并 原生 支持了 bitsandbytes 和 auto-gptq 这两个量化库。


- https://huggingface.co/docs/transformers/v4.35.2/en/main_classes/quantization
- 更多量化方案：https://github.com/huggingface/optimum



### GPTQ量化


```
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

model_id = "facebook/opt-125m"

quantization_config = GPTQConfig(
     bits=4,
     group_size=128,
     dataset="c4",
     desc_act=False,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
quant_model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config, device_map='auto')
```


### LLM.int8()-bitsandbytes

```
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "bigscience/bloomz-7b1-mt"


tokenizer = AutoTokenizer.from_pretrained(model_name)
model_8bit = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True)
```

```
from transformers import BitsAndBytesConfig

double_quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
)

model_double_quant = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=double_quant_config)
```






