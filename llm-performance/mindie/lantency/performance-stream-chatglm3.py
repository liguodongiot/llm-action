
import requests
import json
import time
import numpy as np


url = "http://192.xxx.16.211:1025/v1/chat/completions"

# input_path = "/home/aicc/alpaca_data_1k.json"
input_path = "/home/aicc/alpaca_gpt4_data_input_1k.json"
list_str = json.load(open(input_path, "r"))


first_token_time_list = []
avg_token_time_list = []
total_time_list = []

count = 0

for line in list_str:
  # instruction = line.get("instruction")
  # inputs = line.get("input")
  instruction = line
  inputs = line

  count += 1
  if count > 1000:
    break

  print("--------------------", str(count))

  if len(inputs) == 0:
    continue
  content = f"<|user|>\n{instruction}<|assistant|>\n"
  payload = json.dumps({
    "model": "chatglm3-6b",
    "messages": [
      {
        "role": "user",
        "content": content
      }
    ],
    "max_tokens": 256,
    "top_p": 0.85,
    "n": 10,
    "logprobs": True,
    "stream": True
  })


  headers = {
    'Content-Type': 'application/json'
  }

  start_time = time.perf_counter()
  start = start_time

  response = requests.request("POST", url, headers=headers, data=payload, stream=True)
  response.raise_for_status()

  i = 0
  gen_time_list = []
  for chunk in response.iter_content(chunk_size=8192):
    end_time = time.perf_counter()
    print(chunk.decode('utf-8'))
    gen_time = end_time - start_time
    i+=1
    if i==1:
      first_token_time_list.append(gen_time)
      print("首Token时延：", round(gen_time, 4))
    else:
      gen_time_list.append(gen_time)
    start_time = end_time

  avg_token_time = sum(gen_time_list) / len(gen_time_list)
  print("Token间时延：", round(avg_token_time, 4))
  avg_token_time_list.append(avg_token_time)

  total_time = end_time - start
  print("端到端时延：", round(total_time, 4))
  total_time_list.append(total_time)


print("首Token时延---------------------")
print("最小值：", round(min(first_token_time_list), 4))
print("最大值：", round(max(first_token_time_list), 4))
print("TP50：", np.percentile(np.array(first_token_time_list), 50))
print("TP90：", np.percentile(np.array(first_token_time_list), 90))
print("TP99：", np.percentile(np.array(first_token_time_list), 99))
print("平均：", round(sum(first_token_time_list) / len(first_token_time_list), 4))


print("Token间时延---------------------")
print("最小值：", round(min(avg_token_time_list), 4))
print("最大值：", round(max(avg_token_time_list), 4))
print("TP50：", np.percentile(np.array(avg_token_time_list), 50))
print("TP90：", np.percentile(np.array(avg_token_time_list), 90))
print("TP99：", np.percentile(np.array(avg_token_time_list), 99))
print("平均：", round(sum(avg_token_time_list) / len(avg_token_time_list), 4))


print("端到端时延---------------------")
print("最小值：", round(min(total_time_list), 4))
print("最大值：", round(max(total_time_list), 4))
print("TP50：", np.percentile(np.array(total_time_list), 50))
print("TP90：", np.percentile(np.array(total_time_list), 90))
print("TP99：", np.percentile(np.array(total_time_list), 99))
print("平均：", round(sum(total_time_list) / len(total_time_list), 4))




