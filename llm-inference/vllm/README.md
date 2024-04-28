

```
docker run -dt --name nvidia_vllm_env --restart=always --gpus all \
--network=host \
--shm-size 4G \
-v /home/guodong.li/workspace:/workspace \
-w /workspace \
nvcr.io/nvidia/pytorch:23.04-py3 \
/bin/bash


docker exec -it nvidia_vllm_env bash
```


```
On the server side, run one of the following commands:

(vLLM backend)
python -m vllm.entrypoints.api_server \
    --model <your_model> --swap-space 16 \
    --disable-log-requests

(TGI backend)
./launch_hf_server.sh <your_model>

On the client side, run:

python benchmarks/benchmark_serving.py \
    --backend <backend> \
    --tokenizer <your_model> --dataset <target_dataset> \
    --request-rate <request_rate>





python -m vllm.entrypoints.api_server \
	--model /workspace/model/bloomz-7b --swap-space 16 \
	--disable-log-requests \
	--host 127.0.0.1
	--port 8001
```


```
docker run -it --gpus all \
--network=host \
--shm-size 4G \
-v /home/guodong.li/workspace:/workspace \
-w /workspace \
vllm:v1 \
python -m vllm.entrypoints.api_server \
--model /workspace/model/bloomz-7b --swap-space 16 \
--disable-log-requests \
--host 10.99.2.xx \
--port 18001
```


```
python benchmarks/benchmark_serving.py \
    --backend vllm \
    --tokenizer /workspace/model/bloomz-7b \
    --dataset /workspace/model/ShareGPT_V3_unfiltered_cleaned_split.json \
    --host 10.99.2.xx \
    --port 18001

```



