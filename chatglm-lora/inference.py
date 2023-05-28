from transformers import AutoModel,AutoTokenizer
import torch
from peft import PeftModel
import json
from cover_alpaca2jsonl import format_example

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


model = AutoModel.from_pretrained("/data/nfs/llm/model/chatglm-6b", trust_remote_code=True, load_in_8bit=True, device_map='auto', revision="")
tokenizer = AutoTokenizer.from_pretrained("/data/nfs/llm/model/chatglm-6b", trust_remote_code=True,  revision="")

model = PeftModel.from_pretrained(model, "/home/guodong.li/data/chatglm-6b-lora")


# TODO
instructions = json.load(open("/data/nfs/guodong.li/data/alpaca_data_cleaned.json"))
answers = []


with torch.no_grad():
    for idx, item in enumerate(instructions[:3]):
        feature = format_example(item)
        input_text = feature['context']
        ids = tokenizer.encode(input_text)
        input_ids = torch.LongTensor([ids])
        input_ids = input_ids.to(device)
        out = model.generate(
            input_ids=input_ids,
            max_length=150,
            do_sample=False,
            temperature=0
        )
        out_text = tokenizer.decode(out[0])
        answer = out_text.replace(input_text, "").replace("\nEND", "").strip()
        item['infer_answer'] = answer
        print(out_text)
        print(f"### {idx+1}.Answer:\n", item.get('output'), '\n\n')
        answers.append({'index': idx, **item})

