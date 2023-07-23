LR=1e-4

MASTER_PORT=$(shuf -n 1 -i 10000-65535)

deepspeed --num_gpus=8 --master_port $MASTER_PORT main.py \
    --deepspeed deepspeed.json \
    --do_train \
    --train_file /data/nfs/llm/data/AdvertiseGen/train.json \
    --test_file /data/nfs/llm/data/AdvertiseGen/dev.json \
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path /data/nfs/llm/model/chatglm-6b \
    --output_dir /home/guodong.li/output/adgen-chatglm-6b-ft-$LR \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 64 \
    --per_device_train_batch_size 30 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --predict_with_generate \
    --num_train_epochs 10 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate $LR \
    --fp16

