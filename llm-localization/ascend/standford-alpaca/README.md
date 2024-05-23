


```
git lfs clone https://www.modelscope.cn/qwen/Qwen1.5-14B-Chat.git
```


```
docker exec -it pytorch_ubuntu_dev bash
conda activate llm-dev 
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /workspace/code/stanford_alpaca
```


## qwen1.5

```

torchrun --nproc_per_node=8 --master_port=29001 train.py \
    --model_name_or_path /workspace/model/Qwen1.5-7B-Chat/ \
    --data_path ./alpaca_data_1k.json \
    --fp16 True \
    --output_dir /workspace/output/alpaca \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'Qwen2DecoderLayer'


```


### 7b-zero2

```

- ds train_micro_batch_size_per_gpu=2 vs hf per_device_train_batch_size=8
- ds gradient_accumulation_steps=8 vs hf gradient_accumulation_steps=1
- ds train_batch_size=128 vs hf train_batch_size (calculated)=64


torchrun --nproc_per_node=8 --master_port=29001 train.py \
    --model_name_or_path /workspace/model/Qwen1.5-7B-Chat/ \
    --data_path ./alpaca_data_1k.json \
    --fp16 True \
    --output_dir /workspace/output/alpaca2 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps=8 \
    --num_train_epochs 1 \
    --per_device_eval_batch_size 4 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --deepspeed ds_config_zero2.json


```

### 14b-zero2

```

rm -rf /workspace/output/alpaca-qwen14/*
torchrun --nproc_per_node=8 --master_port=29001 train.py \
    --model_name_or_path /workspace/model/Qwen1.5-14B-Chat/ \
    --data_path /workspace/data/alpaca_gpt4_data_zh_5k.json \
    --fp16 True \
    --output_dir /workspace/output/alpaca-qwen14 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps=8 \
    --num_train_epochs 1 \
    --per_device_eval_batch_size 4 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --deepspeed ds_config_zero2.json
```

### zero3

```
rm -rf /workspace/output/alpaca2/*
torchrun --nproc_per_node=8 --master_port=29001 train.py \
    --model_name_or_path /workspace/model/Qwen1.5-7B-Chat/ \
    --data_path /workspace/data/alpaca_gpt4_data_zh_5k.json \
    --fp16 True \
    --output_dir /workspace/output/alpaca2 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps=8 \
    --num_train_epochs 1 \
    --per_device_eval_batch_size 4 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --deepspeed ds_config_zero3.json



torchrun --nproc_per_node=8 --master_port=29001 train.py \
    --model_name_or_path /workspace/model/Qwen1.5-7B-Chat/ \
    --data_path /workspace/data/alpaca_gpt4_data_zh_5k.json \
    --fp16 True \
    --output_dir /workspace/output/alpaca2 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps=8 \
    --num_train_epochs 1 \
    --per_device_eval_batch_size 4 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --deepspeed ds_config.json
```



## baichuan2



```

torchrun --nproc_per_node=8 --master_port=29001 train.py \
    --model_name_or_path /workspace/model/Baichuan2-7B-Chat \
    --data_path ./alpaca_data_1k.json \
    --fp16 True \
    --output_dir /workspace/output/baichuan2 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'DecoderLayer'

```


```

mkdir -p /workspace/output/baichuan2-13b
torchrun --nproc_per_node=8 --master_port=29001 train.py \
    --model_name_or_path /workspace/model/Baichuan2-13B-Chat \
    --data_path ./alpaca_data_1k.json \
    --fp16 True \
    --output_dir /workspace/output/baichuan2-13b \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing true \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'BaichuanLayer'


```



### zero2

```
rm -rf /workspace/output/baichuan2-13b/*

nohup \ 
torchrun --nproc_per_node=8 --master_port=29001 train.py \
    --model_name_or_path /workspace/model/Baichuan2-7B-Chat \
    --data_path /workspace/data/alpaca_gpt4_data_zh_5k.json \
    --fp16 True \
    --output_dir /workspace/output/baichuan2-ds-7b/ \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps=8 \
    --num_train_epochs 1 \
    --per_device_eval_batch_size 4 \
    --gradient_checkpointing true \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --deepspeed ds_config_zero2.json \
> baichuan2-ds-7b.log 2>&1 &

tail -100f baichuan2-ds-7b.log





mkdir -p /workspace/output/baichuan2-ds-13b/

nohup \ 
torchrun --nproc_per_node=8 --master_port=29001 train.py \
    --model_name_or_path /workspace/model/Baichuan2-13B-Chat \
    --data_path /workspace/data/alpaca_gpt4_data_zh_5k.json \
    --fp16 True \
    --output_dir /workspace/output/baichuan2-ds-13b/ \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps=8 \
    --num_train_epochs 1 \
    --per_device_eval_batch_size 4 \
    --gradient_checkpointing true \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --deepspeed ds_config_zero2.json \
> baichuan2-ds-13b.log 2>&1 &

tail -100f baichuan2-ds-13b.log
```



### zero3


```

rm -rf /workspace/output/baichuan2-ds-7b/*
mkdir -p /workspace/output/baichuan2-ds-7b/

rm -rf /workspace/output/baichuan2-ds-7b/*
nohup \ 
torchrun --nproc_per_node=8 --master_port=29001 train.py \
    --model_name_or_path /workspace/model/Baichuan2-7B-Chat \
    --data_path /workspace/data/alpaca_gpt4_data_zh_5k.json \
    --fp16 True \
    --output_dir /workspace/output/baichuan2-ds-7b/ \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps=8 \
    --num_train_epochs 1 \
    --per_device_eval_batch_size 4 \
    --gradient_checkpointing true \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --deepspeed ds_config_zero3.json \
> baichuan2-ds-7b.log 2>&1 &

tail -100f baichuan2-ds-7b.log




TypeError: Old language model head is of type <class 'transformers_modules.Baichuan2-7B-Chat.modeling_baichuan.NormHead'>, which is not an instance of <class 'torch.nn.modules.linear.Linear'>. You should either use a different resize function or make sure that `old_lm_head` are an instance of <class 'torch.nn.modules.linear.Linear'>.






rm -rf /workspace/output/baichuan2-13b/*
torchrun --nproc_per_node=8 --master_port=29001 train.py \
    --model_name_or_path /workspace/model/Baichuan2-13B-Chat \
    --data_path /workspace/data/alpaca_gpt4_data_zh_5k.json \
    --fp16 True \
    --output_dir /workspace/output/baichuan2-13b/ \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps=8 \
    --num_train_epochs 1 \
    --per_device_eval_batch_size 4 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --deepspeed ds_config_zero3.json


TypeError: Old language model head is of type <class 'transformers_modules.Baichuan2-13B-Chat.modeling_baichuan.NormHead'>, which is not an instance of <class 'torch.nn.modules.linear.Linear'>. You should either use a different resize function or make sure that `old_lm_head` are an instance of <class 'torch.nn.modules.linear.Linear'>.

https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat/blob/ea66ced17780ca3db39bc9f8aa601d8463db3da5/modeling_baichuan.py#L495

https://huggingface.co/Qwen/Qwen-7B-Chat/blob/93a65d34827a3cc269b727e67004743b723e2f83/modeling_qwen.py#L980

```














