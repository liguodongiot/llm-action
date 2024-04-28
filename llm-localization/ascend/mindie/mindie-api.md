




## OpenAI


```
curl -H "Accept: application/json" -H "Content-type: application/json" -X POST -d '{
    "model": "qwen-72b",
    "messages": [
      {
        "role": "system",
        "content": "你是一个有用的助手."
      },
      {
        "role": "user",
        "content": "如何养生？"
      }
    ]
  }' http://127.0.0.1:1025/v1/chat/completions




curl "http://127.0.0.1:1025/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "baichuan2-7b",
    "messages": [
      {
        "role": "user",
        "content": "如何养生？"
      }
    ],
    "max_tokens":128
  }'




curl -H "Accept: application/json" -H "Content-type: application/json" -X POST -d '{
    "model": "qwen-72b",
    "messages": [
      {
        "role": "user",
        "content": "请给我5条人生建议？"
      }
    ],
    "max_tokens":128
  }' http://127.0.0.1:1025/v1/chat/completions



# 流式
curl -H "Accept: application/json" -H "Content-type: application/json" -X POST -d '{
"model": "gpt-3.5-turbo-16k",
"messages": [
  {
    "role": "system",
    "content": "You are a helpful assistant."
  },
  {
    "role": "user",
    "content": "Hello!"
  }
],
"stream": true
}' http://127.0.0.1:1025/v1/chat/completions




curl -H "Accept: application/json" -H "Content-type: application/json" -X POST -d '{
"model": "baichuan2-7b",
"messages": [
  {
    "role": "user",
    "content": "保持健康的方法"
  }
],
"top_p": 0.85,
"max_tokens":128
}' http://127.0.0.1:1025/v1/chat/completions







curl -H "Accept: application/json" -H "Content-type: application/json" -X POST -d '{
"model": "qwen-72b",
"messages": [
  {
    "role": "user",
    "content": "保持健康的方法"
  }
],
"stream": true
}' http://127.0.0.1:1025/v1/chat/completions


```



## vLLM

- https://github.com/vllm-project/vllm/blob/main/examples/api_client.py

- https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/api_server.py

此API服务仅用于演示AsyncEngine的使用和简单的性能基准测试。它不打算用于生产使用。
对于生产使用，我们建议使用我们的OpenAI兼容服务。

- 推荐：https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/api_server.py

```
curl -H "Accept: application/json" -H "Content-type: application/json" -X POST -d '{
    "prompt": "保持健康的方法",
    "n": 5,
    "temperature": 0.0,
    "max_tokens": 64
}' http://127.0.0.1:1025/generate
```



## tgi

- https://huggingface.github.io/text-generation-inference/

```
{
  "inputs": "My name is Olivier and I",
  "parameters": {
    "best_of": 1,
    "decoder_input_details": false,
    "details": true,
    "do_sample": true,
    "frequency_penalty": 0.1,
    "grammar": null,
    "max_new_tokens": 20,
    "repetition_penalty": 1.03,
    "return_full_text": false,
    "seed": null,
    "stop": [
      "photographer"
    ],
    "temperature": 0.5,
    "top_k": 10,
    "top_n_tokens": 5,
    "top_p": 0.95,
    "truncate": null,
    "typical_p": 0.95,
    "watermark": true
  }
}
```



```
curl -H "Accept: application/json" -H "Content-type: application/json" -X POST -d '{
  "inputs": "如何才能拥有性感的身材?",
  "parameters": {
  	"do_sample": true,
    "frequency_penalty": 0.1,
    "temperature": 0.5,
    "top_k": 10,
    "top_n_tokens": 5,
    "max_new_tokens": 256
  }
}' http://127.0.0.1:1025/generate


# 流式输出

curl -H "Accept: application/json" -H "Content-type: application/json" -X POST -d '{
  "inputs": "如何才能拥有性感的身材?",
  "parameters": {
    "max_new_tokens": 50
  }
}' http://127.0.0.1:1025/generate_stream
```



## MindIE-service
```
curl -H "Accept: application/json" -H "Content-type: application/json" -X POST -d '{
"inputs": "保持健康的方法",
"parameters": {
"best_of": 1,
"decoder_input_details": true,
"details": true,
"do_sample": true,
"max_new_tokens": 64,
"repetition_penalty": 1.03,
"return_full_text": false,
"seed": null,
"stop": [
"photographer"
],
"temperature": 0.5,
"top_n_tokens": 5,
"top_p": 0.95,
"truncate": null,
"typical_p": 0.95,
"watermark": true
},
"stream": false}' http://127.0.0.1:1025/generate
```




