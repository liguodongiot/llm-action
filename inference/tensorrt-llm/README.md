






```
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git submodule update --init --recursive
git lfs install
git lfs pull




docker pull nvcr.io/nvidia/pytorch:23.08-py3

```








```
docker rm -f tensorrt_llm

docker run -dt --name tensorrt_llm \
--restart=always \
--gpus all \
--network=host \
--shm-size=4g \
-m 64G \
-v /home/gdong/workspace:/workspace \
-w /workspace \
nvcr.io/nvidia/pytorch:23.08-py3 \
/bin/bash


docker exec -it tensorrt_llm bash




pip install transformers==4.31.0 -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn


pip install transformers==4.31.0 --progress-bar off -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn



```




```

python3 ./scripts/build_wheel.py --trt_root /usr/local/tensorrt

```