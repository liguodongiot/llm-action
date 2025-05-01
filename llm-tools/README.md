



- [LLM-Viewer](https://github.com/hahnyuan/LLM-Viewer.git) : 可视化大语言模型 (LLMs) 并分析不同硬件平台上性能
- [llm-analysis](https://github.com/cli99/llm-analysis) : 对 Transformer 模型的训练和推理进行延迟和内存分析
- [llm_profiler](https://github.com/harleyszhang/llm_counts) : 大模型理论性能分析工具


- [vLLM 性能分析](https://vllm.hyper.ai/docs/contributing/profiling_index/)
	- https://docs.vllm.ai/en/stable/contributing/profiling/profiling_index.html
	- PyTorch Profiler
	- NVIDIA Nsight Systems 
		

- [SGLang Benchmark and Profiling](https://docs.sglang.ai/references/benchmark_and_profiling.html)



--json-model-override-args


--load-format dummy





## Pytorch Profiler

- https://pytorch.org/tutorials/beginner/profiler.html
- https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html

教程：

1. https://pytorch.org/tutorials/prototype/numeric_suite_tutorial.html
2. https://pytorch.org/tutorials/intermediate/fx_profiling_tutorial.html








## NVIDIA Nsight Systems 

- https://help.aliyun.com/zh/ack/cloud-native-ai-suite/use-cases/using-nsight-system-to-realize-performance-analysis?spm=a2c4g.11186623.help-menu-85222.d_3_0_1.77d64381Fe4MY6&scm=20140722.H_2710010._.OR_help-T_cn~zh-V_1
- https://zhuanlan.zhihu.com/p/718956195

属于系统级性能分析工具，提供从全局视角对整个系统的性能进行监控和分析，包括 CPU、GPU、内存、IO 等多种硬件资源的使用情况，以及它们之间的交互信息。


使用 --python-backtrace=cuda 查看所有 CUDA 内核的 python 调用堆栈，就像在 PyTorch Profiler 中一样。（注意：这可能会导致基于 CUDA 事件的计时的内核运行时间不准确）



chrome://tracing


python -m sglang.bench_serving --backend sglang --model meta-llama/Llama-3.1-8B-Instruct --num-prompts 2 --sharegpt-output-len 100 --profile



## NVIDIA Nsight Compute

专注于 GPU 内核级的性能分析，主要针对 CUDA 应用程序，深入到 GPU 内部，分析 CUDA 内核的执行情况。





## NVIDIA Tools Extension Library（NVTX）

通过使用NVTX，开发者可以在代码中添加注释，这些注释可以被NVIDIA的开发工具识别，从而在性能分析和调试过程中提供帮助。

```
pip install nvtx
```


```
# example_lib.py

import time
import nvtx


def sleep_for(i):
    time.sleep(i)

@nvtx.annotate()
def my_func():
    time.sleep(1)

with nvtx.annotate("for_loop", color="green"):
    for i in range(5):
        sleep_for(i)
        my_func()

```

```
nsys profile python demo.py
```














