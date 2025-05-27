


```
git clone git@github.com:liguodongiot/llm-compressor.git


git remote add upstream git@github.com:vllm-project/llm-compressor.git


# 拉取原始仓库数据
git fetch upstream --tags

# 如果你的主分支不是叫master，就把前面的master换成你的名字，比如main之类
git rebase upstream/main

# 推送
git push

# 推送tags
git push --tags

```


## llm-compressor

支持的量化类型：
- https://github.com/neuralmagic/compressed-tensors/blob/main/src/compressed_tensors/quantization/quant_scheme.py



int8:
https://github.com/vllm-project/llm-compressor/tree/main/examples/quantization_w8a8_int8

fp8 dynamic:
https://github.com/vllm-project/llm-compressor/tree/main/examples/quantization_w8a8_fp8























