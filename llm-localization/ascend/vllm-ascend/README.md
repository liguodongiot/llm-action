

# vllm 支持 NPU
https://github.com/vllm-project/vllm/issues/7692 



https://vllm-ascend.readthedocs.io/en/latest/quick_start.html



# 镜像

https://quay.io/repository/ascend/vllm-ascend?tab=tags


https://quay.io/repository/ascend/cann?tab=tags





```
TAG=v0.7.3rc2
docker pull m.daocloud.io/quay.io/ascend/vllm-ascend:$TAG


docker pull quay.io/ascend/vllm-ascend:v0.8.5rc1-openeuler


docker pull quay.io/ascend/vllm-ascend:v0.8.5rc1
```



## 支持的特性

https://vllm-ascend.readthedocs.io/en/latest/user_guide/suppoted_features.html


## 推理模型输出

https://github.com/vllm-project/vllm/blob/main/docs/features/reasoning_outputs.md


- v0.9.0 支持 qwen3 推理输出解析器
https://github.com/vllm-project/vllm/tree/v0.9.0/vllm/reasoning