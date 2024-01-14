from mindspore import context
from mindformers.pipeline import pipeline
from mindformers.models import LlamaConfig
from mindformers import MindFormerConfig

from baichuan2_7b import Baichuan7BV2ForCausalLM
from baichuan2_13b import Baichuan13BV2ForCausalLM
from baichuan2_tokenizer import Baichuan2Tokenizer
import time
import json
import numpy as np


#  GRAPH_MODE(0) or PYNATIVE_MODE(1)
context.set_context(device_id=2, mode=0)


input_path = "/root/workspace/data/alpaca_gpt4_data_input_2k.json"
gen_max_tokens = 100
#input_path = "/root/workspace/data/alpaca_10.json"


list_str = json.load(open(input_path, "r"))


model_dict = {
    "baichuan2_7b": Baichuan7BV2ForCausalLM,
    "baichuan2_13b": Baichuan13BV2ForCausalLM,
}


# init model
baichuan2_config_path = "/root/mindformers/research/baichuan2/run_baichuan2_7b.yaml"
baichuan2_config = MindFormerConfig(baichuan2_config_path)

baichuan2_config.model.model_config.batch_size = 1

baichuan2_model_config = LlamaConfig(**baichuan2_config.model.model_config)
model_name = baichuan2_config.trainer.model_name
baichuan2_network = model_dict[model_name](
    config=baichuan2_model_config
)

# init tokenizer
tokenizer = Baichuan2Tokenizer(
    vocab_file=baichuan2_config.processor.tokenizer.vocab_file
)



text="可以帮我做一份旅游攻略吗？"

# predict using generate
inputs_ids = tokenizer(text)["input_ids"]
# inputs_ids = tokenizer(text, max_length=64, padding="max_length")["input_ids"]

input_token_lens = len(inputs_ids)
start_time = time.perf_counter()
outputs = baichuan2_network.generate(inputs_ids,
                                     do_sample=False,
                                     num_beams=1, 
                                     top_k=1,
                                     top_p=1.0,
                                     repetition_penalty=1.0,
                                     temperature=1.0,
                                     max_new_tokens=64)
end_time = time.perf_counter()
first_gen_time = end_time - start_time
print("第一次生成时间：", first_gen_time)
outputs =outputs[0][len(inputs_ids):]
response = tokenizer.decode(outputs)
print(response)




first_token_time_list = []
total_token_time_list = []



for i, line in enumerate(list_str):
    inputs_ids = tokenizer(line)["input_ids"]
    input_token_lens = len(inputs_ids)
    
    start_time = time.perf_counter()
    outputs = baichuan2_network.generate(inputs_ids,
                                     do_sample=False,
                                     num_beams=1, 
                                     top_k=1,
                                     top_p=1.0,
                                     repetition_penalty=1.0,
                                     temperature=1.0,
                                     max_new_tokens=1)
    end_time = time.perf_counter()
    output_token_lens = len(outputs[0])
    new_token_lens = output_token_lens - input_token_lens

    gen_time = end_time - start_time
    print("生成时间：", gen_time, "输入Token长度：", input_token_lens, "生成Token长度：", new_token_lens)
    outputs =outputs[0][len(inputs_ids):]
    response = tokenizer.decode(outputs)
    print(response)
    first_token_time_list.append(gen_time)
    print("\n-------------------\n")


new_token_lens_list = []

for i, line in enumerate(list_str):
    inputs_ids = tokenizer(line)["input_ids"]
    input_token_lens = len(inputs_ids)

    start_time = time.perf_counter()
    outputs = baichuan2_network.generate(inputs_ids,
                                     do_sample=False,
                                     num_beams=1, 
                                     top_k=1,
                                     top_p=1.0,
                                     repetition_penalty=1.0,
                                     temperature=1.0,
                                     max_new_tokens=gen_max_tokens)
    end_time = time.perf_counter()

    output_token_lens = len(outputs[0])
    new_token_lens = output_token_lens - input_token_lens

    gen_time = end_time - start_time
    print("生成时间：", gen_time, "输入Token长度：", input_token_lens, "生成Token长度：", new_token_lens)
    outputs =outputs[0][len(inputs_ids):]
    response = tokenizer.decode(outputs)
    print(response)
    total_token_time_list.append(gen_time)
    new_token_lens_list.append(new_token_lens)
    print("\n-------------------\n")

print(len(first_token_time_list), len(total_token_time_list), len(new_token_lens_list))


avg_first_token_time = sum(first_token_time_list) / len(first_token_time_list)

avg_total_token_time = sum(total_token_time_list) / len(total_token_time_list)

# 平均生成Token长度
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
print("最小值：", round(min(first_token_time_list), 4))
print("最大值：", round(max(first_token_time_list), 4))
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
