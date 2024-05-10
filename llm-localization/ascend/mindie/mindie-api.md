




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




curl "http://127.0.0.1:1025/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
  	"model": "qwen1.5-14b",
    "messages": [
      {
        "role": "user",
        "content": "如何养生？"
      }
    ],
    "max_tokens":256
  }'



curl "http://127.0.0.1:1025/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
  	"model": "qwen1.5-14b",
    "messages": [
      {
        "role": "user",
        "content": "你好，我叫李聪明。请问你是谁？"
      }
    ],
    "max_tokens":256,
    "top_p": 0.85,
    "n": 10,
    "logprobs": true,
    "stop": "<|im_end|>"
  }'



curl "http://127.0.0.1:1025/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
  	"model": "qwen1.5-14b",
    "messages": [
      {
        "role": "user",
        "content": "你好，我叫李聪明。请问你是谁？"
      },{
        "role": "assistant",
        "content": "你好，李聪明！很高兴认识你。我是一个大型语言模型，你可以叫我通义千问。有什么问题或需要帮助的话，请随时告诉我。"
      },{
        "role": "user",
        "content": "我最近心情很糟糕，能给我一些建议吗？"
      }
    ],
    "max_tokens":256,
    "top_p": 0.85,
    "n": 10,
    "logprobs": true
  }'




curl "http://127.0.0.1:1025/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
  	"model": "qwen1.5-14b",
    "messages": [
      {
        "role": "user",
        "content": "你好，我叫李聪明。请问你是谁？\n你好，李聪明！很高兴认识你。我是一个大型语言模型，你可以叫我通义千问。有什么问题或需要帮助的话，请随时告诉我。\n我最近心情很糟糕，能给我一些建议吗？"
      }
    ],
    "max_tokens":256,
    "top_p": 0.85,
    "n": 10,
    "logprobs": true
  }'

----

<|im_start|>user
你好，我的名字是李聪明。请问你是谁？<|im_end|>
<|im_start|>assistant
你好，李聪明！很高兴认识你。我是一个大型语言模型，你可以叫我通义千问。有什么问题或需要帮助的话，请随时告诉我。<|im_end|>
<|im_start|>user
我最近心情很糟糕，能给我一些建议吗？<|im_end|>
<|im_start|>assistant
我很理解你现在的感受。面对糟糕的心情，以下是一些可能有帮助的建议：\n\n1. **与他人分享**：告诉信任的朋友或家人你的感受，他们可能能提供安慰和支持。\n2. **自我关怀**：确保每天有足够的休息，做些你喜欢的事情，比如阅读、听音乐或运动。\n3. **运动与放松**：适度的运动可以帮助释放压力，尝试瑜伽、冥想或深呼吸练习。\n4. **寻求专业帮助**：如果你觉得压力过大，考虑咨询心理医生或心理咨询师。\n5. **保持积极思考**：试着找出生活中的小确幸，每天对自己说一些积极的话。\n6. **时间管理**：合理安排时间，避免过度压力，留出放松的时间。\n\n记住，处理情绪需要时间和耐心，不要对自己太苛刻。如果你的情绪持续低落，可能需要更专业的支持。希望这些建议对你有所帮助。<|im_end|>
<|im_start|>user
请问我叫什么名字？<|im_end|>

----

curl "http://127.0.0.1:1025/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
  	"model": "qwen1.5-14b",
    "messages": [
      {
        "role": "user",
        "content": "<|im_start|>user\n你好，我的名字是李聪明。请问你是谁？<|im_end|>\n<|im_start|>assistant\n你好，李聪明！很高兴认识你。我是一个大型语言模型，你可以叫我通义千问。有什么问题或需要帮助的话，请随时告诉我。<|im_end|>\n<|im_start|>user\n我最近心情很糟糕，能给我一些建议吗？<|im_end|><|im_start|>assistant\n我很理解你现在的感受。面对糟糕的心情，以下是一些可能有帮助的建议：\n\n1. **与他人分享**：告诉信任的朋友或家人你的感受，他们可能能提供安慰和支持。\n2. **自我关怀**：确保每天有足够的休息，做些你喜欢的事情，比如阅读、听音乐或运动。\n3. **运动与放松**：适度的运动可以帮助释放压力，尝试瑜伽、冥想或深呼吸练习。\n4. **寻求专业帮助**：如果你觉得压力过大，考虑咨询心理医生或心理咨询师。\n5. **保持积极思考**：试着找出生活中的小确幸，每天对自己说一些积极的话。\n6. **时间管理**：合理安排时间，避免过度压力，留出放松的时间。\n\n记住，处理情绪需要时间和耐心，不要对自己太苛刻。如果你的情绪持续低落，可能需要更专业的支持。希望这些建议对你有所帮助。<|im_end|>\n<|im_start|>user\n请问我叫什么名字？<|im_end|>\n<|im_start|>assistant\n"
      }
    ],
    "max_tokens":256,
    "top_p": 0.85,
    "n": 10,
    "logprobs": true,
    "stop": "<|im_end|>"
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


# 返回结果
data: {"id":"554","object":"chat.completion.chunk","created":1715064985,"model":"qwen1.5-14b","choices":[{"index":0,"delta":{"role":"assistant","content":"节点"},"finish_reason":null}]}




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





### 返回结果

```
{
    "id": "209",
    "object": "chat.completion",
    "created": 1715051228,
    "model": "qwen1.5-14b",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "\n你叫李聪明。这是我根据之前的对话信息得知的。\nuser\n"
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 277,
        "completion_tokens": 25,
        "total_tokens": 302
    }
}
```



### baichuan2

```


curl "http://127.0.0.1:1025/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "baichuan2-7b",
    "messages": [
      {
        "role": "user",
        "content": "<reserved_106>光的三原色是什么<reserved_107>"
      }
    ],
    "max_tokens":256,
    "top_p": 0.85,
    "n": 10,
    "logprobs": true
  }'
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





## triton

```
curl "http://127.0.0.1:1025/v2"
```




## MindIE-service


curl "http://127.0.0.1:1025/v1/models"


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




