




- https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py
- TrainingArguments




## 保存模型

- model.save_pretrained('./path_to_model/')
- model.config.to_json_file("config.json")




## RoPE



rope_scaling
```
Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
accordingly.
Expected contents:
    `rope_type` (`str`):
        The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
        'llama3'], with 'default' being the original RoPE implementation.
    `factor` (`float`, *optional*):
        Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
        most scaling types, a `factor` of x will enable the model to handle sequences of length x *
        original maximum pre-trained length.
```


















