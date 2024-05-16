
# Locust用户脚本就是Python模块  
import time  
from locust import HttpUser, task, between, HttpLocust
import json
import random

# input_path = "/home/aicc/alpaca_data_1k.json"
input_path = "/home/aicc/alpaca_gpt4_data_input_1k.json"
list_str = json.load(open(input_path, "r"))

list_token = []

# 定义用户行为
# 类继承自HttpUser  
class QuickstartUser(HttpUser):  

	# 被@task装饰的才会并发执行  
    @task  
    def hello_world(self):
        # wait_time = between(1, 2)

        # client属性是HttpSession实例，用来发送HTTP请求  
        # inputs = list_str[random.randint(1, 500)]
        # instruction = inputs.get("instruction")

        instruction = list_str[random.randint(1, 500)]
        content = f"<reserved_106>{instruction}<reserved_107>"
        payload = {
          "model": "baichuan2-7b",
          "messages": [
            {
              "role": "user",
              "content": content
            }
          ],
          "max_tokens": 256,
          "top_p": 0.85,
          "n": 10,
          "logprobs": True
        }

        response = self.client.post("/v1/chat/completions", json=payload)
        usage = json.loads(response.text).get("usage")
        list_token.append(usage)
		

    def on_stop(self):
    	print("list_token:", len(list_token))
    	prompt_tokens = 0
    	completion_tokens = 0
    	total_tokens = 0
    	for token in list_token:
    		prompt_tokens += token.get("prompt_tokens")
    		completion_tokens += token.get("completion_tokens")
    		total_tokens += token.get("total_tokens")
    	print("prompt_tokens: ", prompt_tokens,"completion_tokens: ", completion_tokens, "total_tokens: ", total_tokens)




