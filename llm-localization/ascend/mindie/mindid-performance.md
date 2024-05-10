





```
curl --location --request POST 'http://192.168.16.211:1025/v1/chat/completions' \
--header 'Content-Type: application/json' \
--data-raw '{
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
    "stop": "<|im_end|>",
    "stream": true
  }'
```





