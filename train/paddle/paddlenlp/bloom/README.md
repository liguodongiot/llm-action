



```
cd /home/guodong.li/virtual-venv
virtualenv -p /usr/bin/python3.10 paddle-venv-py310-cu117
source /home/guodong.li/virtual-venv/paddle-venv-py310-cu117/bin/activate
```


```
from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")
model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m", dtype="float16")


from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("/home/guodong.li/.paddlenlp/models/bigscience/bloomz-560m")
model = AutoModelForCausalLM.from_pretrained("/home/guodong.li/.paddlenlp/models/bigscience/bloomz-560m", dtype="float16")

input_features = tokenizer("hi, my name is", return_tensors="pd")
outputs = model.generate(**input_features, max_length=128)
tokenizer.batch_decode(outputs[0])
```


```
python -u  -m paddle.distributed.launch --gpus "0,1,2,3" finetune_generation.py ./bloom/sft_argument.json
```



