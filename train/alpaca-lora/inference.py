import sys
from transformers import LlamaForCausalLM, AutoTokenizer
import torch

device = torch.device("cuda:2") if torch.cuda.is_available() else torch.device("cpu")


tokenizer_path="/data/nfs/guodong.li/pretrain/hf-llama-model/tokenizer"
model_path = "/home/guodong.li/output/hf_65b_ckpt" # You can modify the path for storing the local model

model =  LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, load_in_8bit=True, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
print("Human:")
line = input()
while line:
        inputs = 'Human: ' + line.strip() + '\n\nAssistant:'
        input_ids = tokenizer(inputs, return_tensors="pt").input_ids
        input_ids = input_ids.to(device)
        outputs = model.generate(input_ids, max_new_tokens=500, do_sample = True, top_k = 30, top_p = 0.85, temperature = 0.5, repetition_penalty=1., eos_token_id=2, bos_token_id=1, pad_token_id=0)
        rets = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print("Assistant:\n" + rets[0].strip().replace(inputs, ""))
        print("\n------------------------------------------------\nHuman:")
        line = input()
