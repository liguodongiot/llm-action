


延迟：


```python
def run_to_completion(profile_dir: Optional[str] = None):

    start_time = time.perf_counter()
    llm.generate(prompt_token_ids=dummy_prompt_token_ids,
                 sampling_params=sampling_params,
                 use_tqdm=False)
    end_time = time.perf_counter()
    latency = end_time - start_time
    return latency

# 第一次预热不计入统计
print("Warming up...")
run_to_completion(profile_dir=None)

# Benchmark.
latencies = []
for _ in tqdm(range(args.num_iters), desc="Profiling iterations"):
    latencies.append(run_to_completion(profile_dir=None))
print(f'Avg latency: {np.mean(latencies)} seconds')

```


- https://github.com/vllm-project/vllm/blob/v0.3.3/benchmarks/benchmark_latency.py#L84



吞吐量：

- 每秒的请求数
- 每秒token处理数：（输入token+输出token）/时延
- https://github.com/vllm-project/vllm/blob/v0.3.3/benchmarks/benchmark_throughput.py#L227



```

total_num_tokens = sum(prompt_len + output_len
                       for _, prompt_len, output_len in requests)
print(f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
      f"{total_num_tokens / elapsed_time:.2f} tokens/s")
```




```
# run python-based benchmarks and upload the result to buildkite
python3 benchmarks/benchmark_latency.py --output-json latency_results.json 2>&1 | tee benchmark_latency.txt
bench_latency_exit_code=$?

python3 benchmarks/benchmark_throughput.py --input-len 256 --output-len 256 --output-json throughput_results.json 2>&1 | tee benchmark_throughput.txt
bench_throughput_exit_code=$?
```




