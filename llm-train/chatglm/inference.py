import os
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer


#MODEL_PATH = "/data/nfs/llm/model/chatglm-6b"
#CHECKPOINT_PATH = "/home/guodong.li/output/adgen-chatglm-6b-pt-128-2e-2/"

MODEL_PATH = "/home/guodong.li/output/adgen-chatglm-6b-ft-1e-4/"


# 载入Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

"""

config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, pre_seq_len=128)
model = AutoModel.from_pretrained(MODEL_PATH, config=config, trust_remote_code=True)


"""
config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True)


"""
prefix_state_dict = torch.load(os.path.join(CHECKPOINT_PATH, "pytorch_model.bin"))
new_prefix_state_dict = {}

for k, v in prefix_state_dict.items():
    if k.startswith("transformer.prefix_encoder."):
        new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
"""



print(f"Quantized to 4 bit")
#model = model.quantize(4)
model = model.half().cuda()


"""
model.transformer.prefix_encoder.float()
"""

model = model.eval()


print("用户：你好\n")
response, history = model.chat(tokenizer, "你好", history=[])
print("ChatGLM-6B：\n",response)
print("\n------------------------------------------------\n用户：")

line = input()
while line:
    # response, history = model.chat(tokenizer, line, history=history)
    response, history = model.chat(tokenizer, line, history=[])
    print("ChatGLM-6B：\n", response)
    print("\n------------------------------------------------\n用户：")
    line = input()

