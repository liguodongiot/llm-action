



- https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver




- https://github.com/triton-inference-server/backend
- https://github.com/triton-inference-server/server






```
docker run --privileged --gpus all -it --net=host -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:23.05-py3-sdk bash
```
