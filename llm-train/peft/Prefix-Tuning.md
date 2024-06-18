



```
from peft import PrefixEncoder, PrefixTuningConfig

config = PrefixTuningConfig(
    peft_type="PREFIX_TUNING",
    task_type="SEQ_2_SEQ_LM",
    num_virtual_tokens=20,
    token_dim=768,
    num_transformer_submodules=1,
    num_attention_heads=12,
    num_layers=12,
    encoder_hidden_size=768,
)

prefix_encoder = PrefixEncoder(config)
```



https://huggingface.co/JackFram/llama-68m
