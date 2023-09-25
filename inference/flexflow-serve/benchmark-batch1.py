import json
import flexflow.serve as ff
import torch
from transformers import LlamaTokenizer
import time
from statistics import mean
import numpy as np


input_data_path = "/workspace/data/alpaca_data.json"
llm_path = "/workspace/model/llama-7b-hf"
ssm_path = "/workspace/model/llama-68m"

list_data_dict = json.load(open(input_data_path, "r"))

prompts = []
i = 0

for temp in list_data_dict:
	prompts.append(temp)
	i+=1
	if i==10000:
		break


ff.init(
        num_gpus=1,
        memory_per_gpu=40000,
        zero_copy_memory_per_node=30000,
        tensor_parallelism_degree=1,
        pipeline_parallelism_degree=1
    )

# Specify the LLM
llm = ff.LLM(llm_path)

# Specify a list of SSMs (just one in this case)
ssms=[]
ssm = ff.SSM(ssm_path)
ssms.append(ssm)


# Create the sampling configs
generation_config = ff.GenerationConfig(
    do_sample=False, temperature=0.9, topp=0.8, topk=1
)

# Compile the SSMs for inference and load the weights into memory
for ssm in ssms:
    ssm.compile(generation_config)

# Compile the LLM for inference and load the weights into memory
llm.compile(generation_config, ssms=ssms)

gen_token_list = []
gen_time_list = []
error_input_list = []

for i in range(len(prompts)):
	prompt = prompts[i]
	print("-----------------------------------------------", i , "-----",prompt)
	input_prompt = prompt["instruction"]+"\n"
	tokenizer = LlamaTokenizer.from_pretrained(llm_path, torch_dtype=torch.float16)
	input_token_len = len(tokenizer(input_prompt).input_ids)
	try:
		start = time.perf_counter()
		llm_result = llm.generate([input_prompt])
		end = time.perf_counter()
		output_token_len = len(llm_result[0].output_tokens)
		gen_token_len = output_token_len - input_token_len
		gen_time = (end - start)*1000
	except:
		error_input_list.append(input_prompt)
		continue
		
	if gen_token_len > 0:
		gen_token_list.append(gen_token_len)
		gen_time_list.append(gen_time)
	else:
		error_input_list.append(input_prompt)


print("错误输入列表：", error_input_list)
print("均值：", round(mean(gen_time_list), 2))
print("最小值：", round(min(gen_time_list), 2))
print("最大值：", round(max(gen_time_list), 2))
print("TP50：", np.percentile(np.array(gen_time_list), 50))
print("TP90：", np.percentile(np.array(gen_time_list), 90))
print("TP99：", np.percentile(np.array(gen_time_list), 99))


token_num_pre_second = round((sum(gen_token_list) / sum(gen_time_list)) * 1000 , 3)
time_pre_token = round(sum(gen_time_list) / sum(gen_token_list), 2)

print("每秒生成的Token数：", token_num_pre_second)
print("每个Token生成的平均耗时(ms)：", time_pre_token)
print("生成Token长度均值：", round(mean(gen_token_list), 2))
print("TOKEN 最小值：", min(gen_token_list))
print("TOKEN 最大值：", max(gen_token_list))
