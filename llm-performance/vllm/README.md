

```
nohup python vllm-performance-stream-qwen1.5-long.py > latency-qwen1.5-7b-a800-long.log 2>&1  &


locust -f locust-qwen1.5-7b-long.py --users 100 --spawn-rate 100 -H http://10.112.xxx.xxx:9009 --run-time 10m



python -m vllm.entrypoints.openai.api_server \
--port 9009 \
--disable-custom-all-reduce \
--gpu-memory-utilization 0.95 \
--dtype auto \
--model /workspace/models/Qwen1.5-7B-Chat \
--tensor-parallel-size 1 \
--max-model-len 10000 \
--served-model-name qwen1.5 \
--max-num-seqs 256 \
--max-num-batched-tokens 10000





python -m vllm.entrypoints.openai.api_server \
--port 9009 \
--disable-custom-all-reduce \
--gpu-memory-utilization 0.95 \
--dtype auto \
--model /workspace/models/Qwen1.5-14B-Chat \
--tensor-parallel-size 1 \
--max-model-len 10000 \
--served-model-name qwen1.5 \
--max-num-seqs 256 \
--max-num-batched-tokens 10000


fp8:

python -m vllm.entrypoints.openai.api_server \
--port 9009 \
--disable-custom-all-reduce \
--gpu-memory-utilization 0.95 \
--dtype auto \
--model /workspace/models/Qwen1.5-7B-Chat \
--tensor-parallel-size 1 \
--quantization="fp8" \
--max-model-len 10000 \
--served-model-name qwen1.5 \
--max-num-seqs 256 \
--max-num-batched-tokens 10000

```
