import requests
import json
import time
import numpy as np

#a800
#url = "http://10.193.195.xxx:9009/v1/chat/completions"

#4090
#url = "http://10.112.2.xxx:9009/v1/chat/completions"

#h800
url = "http://10.112.64.xxx:9009/v1/chat/completions"


# input_path = "/home/aicc/alpaca_data_1k.json"
input_path = "./alpaca_gpt4_data_input_1k.json"
list_str = json.load(open(input_path, "r"))

first_token_time_list = []
avg_token_time_list = []

intertoken_time_list = []
total_time_list = []
gen_token_len_list = []


prompt_tokens_list = []
completion_tokens_list = []
total_tokens_list = []



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

  #content = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
  content = f"{instruction}"

  payload = json.dumps({
    "model": "qwen1.5",
    "messages": [
      {
        "role": "user",
        "content": "保持健康的秘诀"
      },
      {
        "role": "assistant",
        "content": "保持健康是一个综合性的概念，涉及到饮食、运动、休息、心理等多个方面。以下是一些保持健康的基本秘诀：\n\n1. 均衡饮食：保证营养均衡，吃足够的蔬菜、水果、全谷类、优质蛋白质（如鱼、鸡、豆腐等）和健康脂肪（如坚果、鱼油）。避免过多摄入糖分、盐分和饱和脂肪。\n\n2. 适量运动：每周至少150分钟的中等强度有氧运动，如快步走、游泳、骑自行车等，或者75分钟的高强度运动，以及每周两次以上的肌肉力量训练。\n\n3. 充足睡眠：成年人每晚应保证7-9小时的睡眠，良好的睡眠有助于身体修复和恢复。\n\n4. 戒烟限酒：避免吸烟，限制酒精摄入，过量饮酒对健康有害。\n\n5. 定期体检：定期进行身体检查，早期发现并处理可能的健康问题。\n\n6. 保持良好的心理状态：保持积极乐观的心态，学会应对压力，进行适当的心理调适，如冥想、瑜伽等。\n\n7. 保持良好的生活习惯：避免熬夜，保持良好的个人卫生，定期洗手，避免接触传染病。\n\n"
      },
      {
        "role": "user",
        "content": "如何保持良好的心态"
      },{"role": "assistant",
        "content": "保持良好的心态是身心健康的重要组成部分，以下是一些方法：\n\n1. **正念冥想**：这是一种训练注意力和觉察力的技巧，可以帮助你专注于当下，减少对过去或未来的忧虑，提升内心的平静和专注。\n\n2. **积极思考**：尝试用积极的角度看待问题，看待困难和挑战时，看到它们作为学习和成长的机会，而不是障碍。\n\n3. **健康的生活方式**：均衡饮食、规律作息、适度运动，这些都有助于保持身体和精神的活力，减少压力。\n\n4. **社交互动**：和朋友、家人保持良好的沟通，社交活动可以提供情感支持，减轻压力。\n\n5. **放松技巧**：如深呼吸、瑜伽、按摩、温泉浴等，可以帮助放松身心，缓解紧张和压力。\n\n6. **设定目标**：有目标的生活可以带来方向感和动力，但要确保目标是实际可行的，避免过度压力。\n\n7. **心理咨询**：如果你发现自己难以应对情绪或压力，不要犹豫寻求专业的心理咨询师的帮助。\n\n8. **自我接纳**：接受自己的优点和缺点，对自己宽容，减少自我批评。\n\n9. **时间管理**：合理安排时间，避免过度工作或学习，给自己留"
      },
      {
        "role": "user",
        "content": content
      }
    ],
    "max_tokens": 256,
    "top_p": 0.85,
    #"n": 10,
    "stream": True
  })

  #print(payload)
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
    result = chunk.decode('utf-8')
    print(result)
    if "assistant" in result and "role" in result :
        continue

    gen_time = end_time - start_time
    start_time = end_time

    i+=1
    if i==1:
      first_token_time_list.append(gen_time)
      print("首Token时延：", round(gen_time, 4))
    else:
      gen_time_list.append(gen_time)

    if "usage" in result and "prompt_tokens" in result :
        result_new = result.lstrip("data: ")
        print("------------------------", result_new)
        json_data= eval(result_new)

        prompt_tokens_list.append(int(json_data["usage"]["prompt_tokens"]))
        completion_tokens_list.append(int(json_data["usage"]["completion_tokens"]))
        total_tokens_list.append(int(json_data["usage"]["total_tokens"]))
    #start_time = end_time

  avg_token_time = sum(gen_time_list) / len(gen_time_list)
  intertoken_time_list.extend(gen_time_list)
  gen_token_len_list.append(len(gen_time_list))
  print("Token间时延：", round(avg_token_time, 4))
  avg_token_time_list.append(avg_token_time)

  total_time = end_time - start
  print("端到端时延：", round(total_time, 4))
  total_time_list.append(total_time)




if len(prompt_tokens_list) != 0:
  print("Token输入输出长度---------------------")
  print("平均输入token长度：", round(sum(prompt_tokens_list) / len(prompt_tokens_list), 5))
  arr_np = np.array(prompt_tokens_list)
  print("输入token长度-均值：", round(np.mean(arr_np),5))
  print("输入token长度-方差：", round(np.var(arr_np),5))
  print("平均输出token长度：", round(sum(completion_tokens_list) / len(completion_tokens_list), 5))
  print("平均总token长度：", round(sum(total_tokens_list) / len(total_tokens_list), 5))


print("首Token时延---------------------")
print("最小值：", round(min(first_token_time_list), 5))
print("最大值：", round(max(first_token_time_list), 5))
print("TP50：", np.percentile(np.array(first_token_time_list), 50))
print("TP90：", np.percentile(np.array(first_token_time_list), 90))
print("TP99：", np.percentile(np.array(first_token_time_list), 99))
print("平均：", round(sum(first_token_time_list) / len(first_token_time_list), 5))


print("平均Token间时延-宏平均---------------------")
print("最小值：", round(min(avg_token_time_list), 4))
print("最大值：", round(max(avg_token_time_list), 4))
print("TP50：", np.percentile(np.array(avg_token_time_list), 50))
print("TP90：", np.percentile(np.array(avg_token_time_list), 90))
print("TP99：", np.percentile(np.array(avg_token_time_list), 99))
print("平均：", round(sum(avg_token_time_list) / len(avg_token_time_list), 4))


print("生成token长度---------------------")

print("最小值：", round(min(gen_token_len_list), 4))
print("最大值：", round(max(gen_token_len_list), 4))
print("TP50：", np.percentile(np.array(gen_token_len_list), 50))
print("TP90：", np.percentile(np.array(gen_token_len_list), 90))
print("TP99：", np.percentile(np.array(gen_token_len_list), 99))
print("平均：", round(sum(gen_token_len_list) / len(gen_token_len_list), 4))



print("Token间时延-微平均---------------------")
print("最小值：", round(min(intertoken_time_list), 4))
print("最大值：", round(max(intertoken_time_list), 4))
print("TP50：", np.percentile(np.array(intertoken_time_list), 50))
print("TP90：", np.percentile(np.array(intertoken_time_list), 90))
print("TP99：", np.percentile(np.array(intertoken_time_list), 99))
print("平均：", round(sum(intertoken_time_list) / len(intertoken_time_list), 4))

print("端到端时延---------------------")
print("最小值：", round(min(total_time_list), 4))
print("最大值：", round(max(total_time_list), 4))
print("TP50：", np.percentile(np.array(total_time_list), 50))
print("TP90：", np.percentile(np.array(total_time_list), 90))
print("TP99：", np.percentile(np.array(total_time_list), 99))
print("平均：", round(sum(total_time_list) / len(total_time_list), 4))


