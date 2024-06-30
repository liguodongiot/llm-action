

```
nohup python vllm-performance-stream-qwen1.5-long.py > latency-qwen1.5-7b-a800-long.log 2>&1  &


locust -f locust-qwen1.5-7b-long.py --users 100 --spawn-rate 100 -H http://10.112.xxx.xxx:9009 --run-time 10m



--quantization="fp8"
```
