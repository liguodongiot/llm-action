




- [SGLang 后端代码解析](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/sglang/code-walk-through/readme-CN.md)



## 量化


https://docs.sglang.ai/backend/quantization.html






## 镜像

- https://github.com/sgl-project/sglang/blob/072df753546b77438479f18a05e691fad91d7f9c/.github/workflows/release-docker.yml#L60
- https://github.com/sgl-project/sglang/blob/main/docker/Dockerfile
- https://hub.docker.com/r/lmsysorg/sglang/tags


```
docker build . -f docker/Dockerfile --build-arg CUDA_VERSION=${{ matrix.cuda_version }} --build-arg BUILD_TYPE=${{ matrix.build_type }} -t lmsysorg/sglang:${tag}${tag_suffix} --no-cache
          docker push lmsysorg/sglang:${tag}${tag_suffix}
```



- docker pull lmsysorg/sglang:v0.4.5-cu125          





## 文档


- SGLang 学习材料：https://github.com/sgl-project/sgl-learning-materials
- Fast and Expressive LLM Inference with RadixAttention and SGLang：https://lmsys.org/blog/2024-01-17-sglang/
- Achieving Faster Open-Source Llama3 Serving with SGLang Runtime (vs. TensorRT-LLM, vLLM)：https://lmsys.org/blog/2024-07-25-sglang-llama3/
- SGLang v0.3 Release: 7x Faster DeepSeek MLA, 1.5x Faster torch.compile, Multi-Image/Video LLaVA-OneVision：https://lmsys.org/blog/2024-09-04-sglang-v0-3/
- SGLang v0.4: Zero-Overhead Batch Scheduler, Cache-Aware Load Balancer, Faster Structured Outputs：https://lmsys.org/blog/2024-12-04-sglang-v0-4/







