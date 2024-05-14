



- LORA: https://github.com/baichuan-inc/Baichuan2/blob/main/fine-tune/fine-tune.py

```
from peft import LoraConfig, TaskType, get_peft_model

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["W_pack"],
            inference_mode=False,
            r=1,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

```

## 4bit/8bit/16bit对应的线程层类


```

from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from transformers import (
    set_seed,
    HfArgumentParser,
    TrainingArguments,
    AutoModelForCausalLM
)

device_map = {'': 0}
model = AutoModelForCausalLM.from_pretrained(
    "/home/guodong.li/workspace/model/bloom-2b6-zh",
    device_map=device_map,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    ),
)

print(model)


BloomForCausalLM(
  (transformer): BloomModel(
    (word_embeddings): Embedding(46145, 2560)
    (word_embeddings_layernorm): LayerNorm((2560,), eps=1e-05, elementwise_affine=True)
    (h): ModuleList(
      (0): BloomBlock(
        (input_layernorm): LayerNorm((2560,), eps=1e-05, elementwise_affine=True)
        (self_attention): BloomAttention(
          (query_key_value): Linear4bit(in_features=2560, out_features=7680, bias=True)
          (dense): Linear4bit(in_features=2560, out_features=2560, bias=True)
          (attention_dropout): Dropout(p=0.0, inplace=False)
        )
        (post_attention_layernorm): LayerNorm((2560,), eps=1e-05, elementwise_affine=True)
        (mlp): BloomMLP(
          (dense_h_to_4h): Linear4bit(in_features=2560, out_features=10240, bias=True)
          (gelu_impl): BloomGelu()
          (dense_4h_to_h): Linear4bit(in_features=10240, out_features=2560, bias=True)
        )
      )
     (29): BloomBlock(
       ...
      )
    )
    (ln_f): LayerNorm((2560,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=2560, out_features=46145, bias=False)
)




from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from transformers import (
    set_seed,
    HfArgumentParser,
    TrainingArguments,
    AutoModelForCausalLM
)

device_map = {'': 0}
model = AutoModelForCausalLM.from_pretrained(
    "/home/guodong.li/workspace/model/bloom-2b6-zh",
    device_map=device_map,
    load_in_8bit=True,
    torch_dtype=torch.float16,
)

print(model)


BloomForCausalLM(
  (transformer): BloomModel(
    (word_embeddings): Embedding(46145, 2560)
    (word_embeddings_layernorm): LayerNorm((2560,), eps=1e-05, elementwise_affine=True)
    (h): ModuleList(
      (0): BloomBlock(
        (input_layernorm): LayerNorm((2560,), eps=1e-05, elementwise_affine=True)
        (self_attention): BloomAttention(
          (query_key_value): Linear8bitLt(in_features=2560, out_features=7680, bias=True)
          (dense): Linear8bitLt(in_features=2560, out_features=2560, bias=True)
          (attention_dropout): Dropout(p=0.0, inplace=False)
        )
        (post_attention_layernorm): LayerNorm((2560,), eps=1e-05, elementwise_affine=True)
        (mlp): BloomMLP(
          (dense_h_to_4h): Linear8bitLt(in_features=2560, out_features=10240, bias=True)
          (gelu_impl): BloomGelu()
          (dense_4h_to_h): Linear8bitLt(in_features=10240, out_features=2560, bias=True)
        )
      )
      ...
      (29): BloomBlock(
        ...
      )
    )
    (ln_f): LayerNorm((2560,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=2560, out_features=46145, bias=False)
)



from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from transformers import (
    set_seed,
    HfArgumentParser,
    TrainingArguments,
    AutoModelForCausalLM
)

device_map = {'': 0}
model = AutoModelForCausalLM.from_pretrained(
    "/home/guodong.li/workspace/model/bloom-2b6-zh",
    device_map=device_map,
    torch_dtype=torch.float16,
)

print(model)

BloomForCausalLM(
  (transformer): BloomModel(
    (word_embeddings): Embedding(46145, 2560)
    (word_embeddings_layernorm): LayerNorm((2560,), eps=1e-05, elementwise_affine=True)
    (h): ModuleList(
      (0): BloomBlock(
        (input_layernorm): LayerNorm((2560,), eps=1e-05, elementwise_affine=True)
        (self_attention): BloomAttention(
          (query_key_value): Linear(in_features=2560, out_features=7680, bias=True)
          (dense): Linear(in_features=2560, out_features=2560, bias=True)
          (attention_dropout): Dropout(p=0.0, inplace=False)
        )
        (post_attention_layernorm): LayerNorm((2560,), eps=1e-05, elementwise_affine=True)
        (mlp): BloomMLP(
          (dense_h_to_4h): Linear(in_features=2560, out_features=10240, bias=True)
          (gelu_impl): BloomGelu()
          (dense_4h_to_h): Linear(in_features=10240, out_features=2560, bias=True)
        )
      )
      (29): BloomBlock(
        ...
      )
    )
    (ln_f): LayerNorm((2560,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=2560, out_features=46145, bias=False)
)
```


## 是否进行计算梯度

```

train_loss =  lw[0] *  loss0 + lw[1] * loss1 + lw[2] * loss2 

# loss backward
for name, parms in model.named_parameters():	
   	print('\nBefore backward\n')
    print('-->name:', name)
    print('-->para:', parms)
    print('-->grad_requirs:',parms.requires_grad)
    print('-->grad_value:',parms.grad)
    print("===========================")
#
train_loss.backward()
#
for name, parms in model.named_parameters():	
    print('\nAfter backward\n')
    print('-->name:', name)
    print('-->para:', parms)
    print('-->grad_requirs:',parms.requires_grad)
    print('-->grad_value:',parms.grad)
    print("===========================")
```



