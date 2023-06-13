
# qlora

- 源码地址：https://github.com/artidoro/qlora
- commit id：cc488110b5ea23594a418daca7085000a9420625




## LLaMA 65B 微调

### 单GPU

```
CUDA_VISIBLE_DEVICES=0 python qlora.py \
    --model_name_or_path /data/nfs/guodong.li/pretrain/hf-llama-model/llama-65b \
    --dataset /data/nfs/guodong.li/data/alpaca_data_cleaned.json \
    --output_dir /home/guodong.li/output/llama-65b-qlora \
    --logging_steps 10 \
    --save_strategy steps \
    --data_seed 42 \
    --save_steps 100 \
    --save_total_limit 2 \
    --evaluation_strategy steps \
    --eval_dataset_size 128 \
    --max_eval_samples 200 \
    --per_device_eval_batch_size 1 \
    --max_new_tokens 32 \
    --dataloader_num_workers 3 \
    --group_by_length \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --do_mmlu_eval \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --source_max_len 16 \
    --target_max_len 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_steps 200 \
    --eval_steps 50 \
    --learning_rate 0.0001 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.05 \
    --weight_decay 0.0 \
    --seed 0 \
    --report_to tensorboard
```

### 多GPU

```
python qlora.py \
    --model_name_or_path /data/nfs/guodong.li/pretrain/hf-llama-model/llama-65b \
    --dataset /data/nfs/guodong.li/data/alpaca_data_cleaned.json \
    --output_dir /home/guodong.li/output/llama-65b-qlora \
    --logging_steps 10 \
    --save_strategy steps \
    --data_seed 42 \
    --save_steps 100 \
    --save_total_limit 2 \
    --evaluation_strategy steps \
    --eval_dataset_size 128 \
    --max_eval_samples 200 \
    --per_device_eval_batch_size 1 \
    --max_new_tokens 32 \
    --dataloader_num_workers 3 \
    --group_by_length \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --do_mmlu_eval \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --source_max_len 16 \
    --target_max_len 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_steps 200 \
    --eval_steps 50 \
    --learning_rate 0.0001 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.05 \
    --weight_decay 0.0 \
    --seed 0 \
    --report_to tensorboard
```



