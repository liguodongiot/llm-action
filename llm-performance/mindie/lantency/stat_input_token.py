

import json
from transformers import AutoConfig, AutoModel, AutoTokenizer
import numpy as np

# input_path = "/home/aicc/alpaca_data_1k.json"
input_path = "/home/aicc/alpaca_gpt4_data_input_1k.json"
list_str = json.load(open(input_path, "r"))


model_path_dict = {
    "Qwen1.5-7B": "/home/aicc/model_from_hf/Qwen1.5-7B-Chat",
    "Qwen1.5-14B": "/home/aicc/model_from_hf/Qwen1.5-14B-Chat",
    "Qwen1.5-72B": "/home/aicc/model_from_hf/Qwen1.5-72B",
    "Qwen-72B": "/home/aicc/model_from_hf/qwen-72b-chat-hf",
    "Baichuan2-7B": "/home/aicc/model_from_hf/Baichuan2-7B-Chat",
    "Baichuan2-13B": "/home/aicc/model_from_hf/Baichuan2-13B-Chat",
    "chatglm3-6b": "/home/aicc/model_from_hf/chatglm3-6b-chat-full"
}

for key, value in model_path_dict.items():
    count = 0
    tokenizer = AutoTokenizer.from_pretrained(value, trust_remote_code=True)
    input_len_list = []

    for line in list_str:
        instruction = line
        inputs = line

        count += 1
        if count > 1000:
            print("--------------------", str(count))
            break

        if len(inputs) == 0:
            continue

        if "Qwen1.5" in key:
            content = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
        elif "Qwen-72B" in key:
            content = f"{instruction}"
        elif "Baichuan2" in key: 
            content = f"<reserved_106>{instruction}<reserved_107>"
        elif "chatglm3" in key:
            content = f"<|user|>\n{instruction}<|assistant|>\n"

        inputs_ids = tokenizer(content)["input_ids"]
        input_token_lens = len(inputs_ids)
        input_len_list.append(input_token_lens)


        avg_len = sum(input_len_list)/len(input_len_list)


    print("--------", key)
    print(input_len_list)

    print("最小值：", round(min(input_len_list), 4))
    print("最大值：", round(max(input_len_list), 4))
    print("TP50：", np.percentile(np.array(input_len_list), 50))
    print("TP90：", np.percentile(np.array(input_len_list), 90))
    print("TP99：", np.percentile(np.array(input_len_list), 99))
    print("平均：", round(avg_len, 4))

