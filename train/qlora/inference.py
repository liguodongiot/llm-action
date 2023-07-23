from transformers import AutoModelForCausalLM, LlamaTokenizer
import torch

model_id = "/data/nfs/guodong.li/pretrain/hf-llama-model/llama-7b"
merge_model_id = "/home/guodong.li/output/llama-7b-merge"

#model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(merge_model_id, load_in_4bit=True, device_map="auto")

tokenizer = LlamaTokenizer.from_pretrained(model_id)

#print(model)

device = torch.device("cuda:0")

#model = model.to(device)

text = "Hello, my name is "
inputs = tokenizer(text, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=20, do_sample=True, top_k=30, top_p=0.85)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

print("\n------------------------------------------------\nInput: ")

line = input()
while line:
  inputs = tokenizer(line, return_tensors="pt").to(device)
  outputs = model.generate(**inputs, max_new_tokens=20, do_sample=True, top_k=30, top_p=0.85)
  print("Output: ",tokenizer.decode(outputs[0], skip_special_tokens=True))
  print("\n------------------------------------------------\nInput: ")
  line = input()
