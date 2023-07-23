import torch
from peft import PeftModel

model_id = "/data/nfs/guodong.li/pretrain/hf-llama-model/llama-7b"
lora_weights = "/home/guodong.li/output/llama-7b-qlora/checkpoint-1000/adapter_model"

#model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, device_map="auto")
model = PeftModel.from_pretrained(
    model,
    lora_weights,
)



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
