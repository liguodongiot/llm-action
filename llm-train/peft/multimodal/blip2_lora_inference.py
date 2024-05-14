
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForVision2Seq, AutoProcessor

from peft import LoraConfig, get_peft_model
from peft import PeftModel, PeftConfig
import torch
import requests
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration



peft_model_id = "/workspace/output/multimodal/blip2"
config = PeftConfig.from_pretrained(peft_model_id)
processor = Blip2Processor.from_pretrained(config.base_model_name_or_path)

model = AutoModelForVision2Seq.from_pretrained(config.base_model_name_or_path, load_in_8bit=True, device_map="auto")
model = PeftModel.from_pretrained(model, peft_model_id)


train_dataset_path = "/workspace/data/pytorch_data/multimodal/blip2/ybelkada___football-dataset/default-80f5618dafa96df9/0.0.0/0111277fb19b16f696664cde7f0cb90f833dec72db2cc73cfdf87e697f78fe02"

dataset = load_dataset(train_dataset_path, split="train")

# Let's load the dataset here!
#dataset = load_dataset("ybelkada/football-dataset", split="train")


item = dataset[0]

device = "cuda" if torch.cuda.is_available() else "cpu"
model.eval()


encoding = processor(images=item["image"], padding="max_length", return_tensors="pt")
# remove batch dimension
encoding = {k: v.squeeze() for k, v in encoding.items()}
encoding["text"] = item["text"]

print(encoding.keys())

processed_batch = {}
for key in encoding.keys():
    if key != "text":
        processed_batch[key] = torch.stack([example[key] for example in [encoding]])
    else:
        text_inputs = processor.tokenizer(
            [example["text"] for example in [encoding]], padding=True, return_tensors="pt"
        )
        processed_batch["input_ids"] = text_inputs["input_ids"]
        processed_batch["attention_mask"] = text_inputs["attention_mask"]


pixel_values = processed_batch.pop("pixel_values").to(device, torch.float16)
print("----------")
generated_output = model.generate(pixel_values=pixel_values)
print(processor.batch_decode(generated_output, skip_special_tokens=True))




