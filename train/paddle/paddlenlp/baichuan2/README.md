


```
from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan2-7B-Base")
model = AutoModelForCausalLM.from_pretrained("baichuan-inc/Baichuan2-7B-Base", dtype="float16")
```