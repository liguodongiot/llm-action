


- https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file
- https://github.com/ggerganov/llama.cpp


GGUF量化格式

- https://lightning.ai/cosmo3769/studios/post-training-quantization-to-gguf-format-and-evaluation
- https://medium.com/@metechsolutions/llm-by-examples-use-gguf-quantization-3e2272b66343
- ctransformers、llama.cpp







```
CMAKE_ARGS="-DGGML_METAL=on" pip install -U llama-cpp-python --no-cache-dir
pip install 'llama-cpp-python[server]'
```


```
export MODEL=/Users/liguodong/model/qwen2/qwen2-0_5b-instruct-q2_k.gguf
python3 -m llama_cpp.server --model $MODEL  --n_gpu_layers 1
```