
## Alpaca-LoRA

- 源码: https://github.com/tloen/alpaca-lora
- commit id : 9de612e582ab86013b5d1c3be6b0ed9f5ab2065a





## LoRA


### 7B

```
torchrun --nproc_per_node=8 --master_port=29005 finetune_metrics_epoch.py \
--base_model '/data/nfs/guodong.li/pretrain/hf-llama-model/llama-7b' \
--data_path '/home/guodong.li/llama-mp/GPT-4-LLM/data/alpaca_gpt4_data_zh.json' \
--output_dir '/home/guodong.li/output/alpaca-lora-7b-dp-zh' \
--batch_size 80 \
--micro_batch_size 10 \
--num_epochs 10 \
--cutoff_len=512 \
--group_by_length \
--lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
--lora_r=16
```



| 模型 | 显存 | 耗时 | 数据量  |
| --- | --- | --- |  --- |
| 7B | 8 * 74G |  2小时5分钟 | 46818 |

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

| 模型 | 显存 | 耗时 | 数据量  |
| --- | --- | --- | --- |
| 13B | 8 * 76G |  2小时10分钟 | 46818 |


### 30B

```
torchrun --nproc_per_node=8 --master_port=29005 finetune.py \
--base_model '/data/nfs/guodong.li/pretrain/hf-llama-model/llama-30b' \
--data_path '/data/nfs/guodong.li/data/alpaca_data_cleaned.json' \
--output_dir '/home/guodong.li/output/alpaca-lora-30b-dp' \
--batch_size 96 \
--micro_batch_size 6 \
--num_epochs 3 
```



### 65B


```
torchrun --nproc_per_node=8 --master_port=29005 finetune.py \
--base_model '/data/nfs/guodong.li/pretrain/hf-llama-model/llama-65b' \
--data_path '/home/guodong.li/llama-mp/GPT-4-LLM/data/alpaca_gpt4_data_zh.json' \
--output_dir '/home/guodong.li/output/alpaca-lora-65b-dp-zh' \
--batch_size 8 \
--micro_batch_size 1 \
--num_epochs 3 
```

## 测试用例

```
请给我讲一个温馨的睡前故事
如何快速提升自己的写作能力？
计算以下表达式：(6+2)*(2-2)。
What are the five characteristics of a good argument?
```





