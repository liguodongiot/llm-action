


## llm 框架对比


- https://github.com/ninehills/llm-inference-benchmark
- https://github.com/triton-inference-server/perf_analyzer




## nvidia-ml-py3、pynvml

- Model training anatomy(模型训练解剖 - huggingface )：https://huggingface.co/docs/transformers/model_memory_anatomy
- https://www.cnblogs.com/devilmaycry812839668/p/15563995.html
- BERT优化技术：https://www.jianshu.com/p/c9c34c5f7bcb



- train performance: 550 tokens/s
- predict performance: 17.75 tokens/s




```
pip install nvidia-ml-py psutil
```



```
from pynvml import *

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

```


## 压测工具

Locust



