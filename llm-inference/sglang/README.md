




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
