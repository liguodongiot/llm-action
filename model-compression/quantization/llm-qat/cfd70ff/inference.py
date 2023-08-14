from transformers import LlamaTokenizer
from models.configuration_llama import LlamaConfig
from models.modeling_llama_quant import LlamaForCausalLM as LlamaForCausalLMQuant
import torch

PATH_TO_CONVERTED_TOKENIZER = "/home/guodong.li/tmp/llama/models/7B-finetuned"
tokenizer = LlamaTokenizer.from_pretrained(
	pretrained_model_name_or_path=PATH_TO_CONVERTED_TOKENIZER,
	model_max_length=30,
	padding_side="right",
	use_fast=False,
)

config = LlamaConfig.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)
model = LlamaForCausalLMQuant.from_pretrained(
	pretrained_model_name_or_path=PATH_TO_CONVERTED_TOKENIZER,
	config=config,
	torch_dtype=torch.bfloat16,
	low_cpu_mem_usage=True,
	device_map="auto"
)
model = model.cuda()

prompt = "Hey, are you consciours? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs.input_ids.cuda()
# Generate
generate_ids = model.generate(input_ids, max_length=30)
tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
