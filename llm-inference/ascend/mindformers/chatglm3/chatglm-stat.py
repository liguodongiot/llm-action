import mindspore as ms
from mindformers import AutoConfig, AutoModel, AutoTokenizer
import time
import json
import numpy as np

input_path = "/root/workspace/data/alpaca_gpt4_data_input_2k.json"

# input_path = "/root/workspace/data/alpaca_10.json"


list_str = json.load(open(input_path, "r"))

# 指定图模式，指定使用训练卡id
ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=3)

tokenizer = AutoTokenizer.from_pretrained('/root/workspace/model/chatglm3-6b_ms')


# model的实例化有以下两种方式，选择其中一种进行实例化即可
# 1. 直接根据默认配置实例化
model = AutoModel.from_pretrained('/root/workspace/model/chatglm3-6b_ms')

"""
# 2. 自定义修改配置后实例化
config = AutoConfig.from_pretrained('/root/workspace/model/chatglm3-6b_ms/run_glm3_6b.yaml')
config.use_past = True                  # 此处修改默认配置，开启增量推理能够加速推理性能
config.seq_length = 2048                      # 根据需求自定义修改其余模型配置
config.checkpoint_name_or_path = "/root/workspace/model/chatglm3-6b_ms/glm3_6b.ckpt"

model = AutoModel.from_config(config)   # 从自定义配置项中实例化模型
"""

role="user"
text="可以帮我做一份旅游攻略吗？"
history=[]
inputs = tokenizer.build_chat_input(text, history=history, role=role)
inputs = inputs['input_ids']
input_token_lens = len(inputs[0])
start_time = time.perf_counter()
outputs = model.generate(inputs, do_sample=False, num_beams=1, top_k=1, top_p=1, temperature=1, repetition_penalty=1.0, max_new_tokens=128)
end_time = time.perf_counter()
first_gen_time = end_time - start_time
print("第一次生成时间：", first_gen_time)
outputs =outputs[0][len(inputs[0]):]
response = tokenizer.decode(outputs)
print(response)




first_token_time_list = []
total_token_time_list = []



for i, line in enumerate(list_str):
    inputs = tokenizer.build_chat_input(line, history=history, role=role)
    inputs = inputs['input_ids']
    input_token_lens = len(inputs[0])
    start_time = time.perf_counter()
    outputs = model.generate(inputs, do_sample=False, num_beams=1, top_k=1, top_p=1, temperature=1, repetition_penalty=1.0, max_new_tokens=1)
    end_time = time.perf_counter()
    output_token_lens = len(outputs[0])
    new_token_lens = output_token_lens - input_token_lens

    gen_time = end_time - start_time
    print("生成时间：", gen_time, "输入Token长度：", input_token_lens, "生成Token长度：", new_token_lens)
    outputs =outputs[0][len(inputs[0]):]
    response = tokenizer.decode(outputs)
    print(response)
    first_token_time_list.append(gen_time)
    print("\n-------------------\n")


new_token_lens_list = []

for i, line in enumerate(list_str):
    inputs = tokenizer.build_chat_input(line, history=history, role=role)
    inputs = inputs['input_ids']
    input_token_lens = len(inputs[0])
    start_time = time.perf_counter()
    outputs = model.generate(inputs, do_sample=False, num_beams=1, top_k=1, top_p=1, temperature=1, repetition_penalty=1.0, max_new_tokens=100)
    end_time = time.perf_counter()
    output_token_lens = len(outputs[0])
    new_token_lens = output_token_lens - input_token_lens

    gen_time = end_time - start_time
    print("生成时间：", gen_time, "输入Token长度：", input_token_lens, "生成Token长度：", new_token_lens)
    outputs =outputs[0][len(inputs[0]):]
    response = tokenizer.decode(outputs)
    print(response)
    total_token_time_list.append(gen_time)
    new_token_lens_list.append(new_token_lens)
    print("\n-------------------\n")

print(len(first_token_time_list), len(total_token_time_list))


avg_first_token_time = sum(first_token_time_list) / len(first_token_time_list)

avg_total_token_time = sum(total_token_time_list) / len(total_token_time_list)

avg_new_token_lens =  sum(new_token_lens_list) / len(new_token_lens_list)

avg_token_time_list = []

for i in range(len(list_str)):
    if (new_token_lens_list[i] <= 1):
        continue
    token_time = (total_token_time_list[i] - first_token_time_list[i]) / (new_token_lens_list[i]-1)
    avg_token_time_list.append(token_time)

avg_token_time = sum(avg_token_time_list) / len(avg_token_time_list)


print(" avg_first_token_time: ", avg_first_token_time,
" avg_token_time: ",avg_token_time,
" avg_total_token_time: ", avg_total_token_time, 
" avg_new_token_lens: ", avg_new_token_lens)


print("首Token时延---------------------")
print("最小值：", round(min(first_token_time_list), 2))
print("最大值：", round(max(first_token_time_list), 2))
print("TP50：", np.percentile(np.array(first_token_time_list), 50))
print("TP90：", np.percentile(np.array(first_token_time_list), 90))
print("TP99：", np.percentile(np.array(first_token_time_list), 99))


print("端到端时延---------------------")
print("最小值：", round(min(total_token_time_list), 2))
print("最大值：", round(max(total_token_time_list), 2))
print("TP50：", np.percentile(np.array(total_token_time_list), 50))
print("TP90：", np.percentile(np.array(total_token_time_list), 90))
print("TP99：", np.percentile(np.array(total_token_time_list), 99))


print("生成Token长度---------------------")
print("最小值：", round(min(new_token_lens_list), 2))
print("最大值：", round(max(new_token_lens_list), 2))
print("TP50：", np.percentile(np.array(new_token_lens_list), 50))
print("TP90：", np.percentile(np.array(new_token_lens_list), 90))
print("TP99：", np.percentile(np.array(new_token_lens_list), 99))




