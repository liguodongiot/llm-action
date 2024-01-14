


import mindspore as ms
from mindformers import AutoConfig, AutoModel, AutoTokenizer
import time



# 指定图模式，指定使用训练卡id
ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=0)

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


line = input()
while line:
    inputs = tokenizer.build_chat_input(line, history=history, role=role)
    inputs = inputs['input_ids']
    input_token_lens = len(inputs[0])
    start_time = time.perf_counter()
    outputs = model.generate(inputs, do_sample=False, num_beams=1, top_k=1, top_p=1, temperature=1, repetition_penalty=1.0, max_new_tokens=128)
    end_time = time.perf_counter()
    gen_time = end_time - start_time
    print("生成时间：", gen_time)

    outputs =outputs[0][len(inputs[0]):]
    response = tokenizer.decode(outputs)
    print(response)
    print("\n-------------------\n")
    line = input()


