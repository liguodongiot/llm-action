

- https://huggingface.co/docs/accelerate/package_reference/cli

```
accelerate env 

# 
accelerate config default [arguments]



accelerate config update --config_file




```



## huggingface 加载大模型

- 使用HuggingFace的Accelerate库加载和运行超大模型: https://zhuanlan.zhihu.com/p/605640431


```
import torch
from transformers import AutoModelForCausalLM
​
checkpoint = "facebook/opt-13b"
model = AutoModelForCausalLM.from_pretrained(
    checkpoint, device_map="auto", offload_folder="offload", offload_state_dict = True, torch_dtype=torch.float16
)

```

