from mindspore import context
from mindformers.pipeline import pipeline
from mindformers.models import LlamaConfig
from mindformers import MindFormerConfig

from baichuan2_7b import Baichuan7BV2ForCausalLM
from baichuan2_13b import Baichuan13BV2ForCausalLM
from baichuan2_tokenizer import Baichuan2Tokenizer
import time


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


line = input()
while line:
    inputs_ids = tokenizer(line)["input_ids"]
    
    input_token_lens = len(inputs_ids)
    start_time = time.perf_counter()
    outputs = baichuan2_network.generate(inputs_ids,
                                        do_sample=False,
                                        top_k=1,
                                        top_p=1.0,
                                        repetition_penalty=1.05,
                                        temperature=1.0,
                                        max_length=64)
    end_time = time.perf_counter()
    gen_time = end_time - start_time
    print("生成时间：", gen_time)
    outputs =outputs[0][len(inputs_ids):]
    response = tokenizer.decode(outputs)
    print(response)
    print("\n-------------------\n")
    line = input()
