
## Alpaca-LoRA

- 源码: https://github.com/tloen/alpaca-lora
- commit id : 9de612e582ab86013b5d1c3be6b0ed9f5ab2065a





## LoRA


### 7B
```

```


### 13B

```
torchrun --nproc_per_node=8 --master_port=29005 finetune_metrics_epoch.py \
--base_model '/data/nfs/guodong.li/pretrain/hf-llama-model/llama-13b' \
--data_path '/home/guodong.li/llama-mp/GPT-4-LLM/data/alpaca_gpt4_data_zh.json' \
--output_dir '/home/guodong.li/output/alpaca-lora-13b-dp-zh' \
--batch_size 48 \
--micro_batch_size 6 \
--num_epochs 10 \
--cutoff_len=512 \
--group_by_length \
--lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
--lora_r=16
```

| 模型 | 显存 | 耗时 |
| --- | --- | --- |
| 13B | 8 * 76G |  4：14分钟|


### 30B



### 65B










