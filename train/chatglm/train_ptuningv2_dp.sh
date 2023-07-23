PRE_SEQ_LEN=128
LR=2e-2

deepspeed --include localhost:1,2,3 --master_port 29001 main.py \
    --deepspeed deepspeed.json \
    --do_train \
    --train_file /data/nfs/llm/data/AdvertiseGen/train.json \
    --validation_file /data/nfs/llm/data/AdvertiseGen/dev.json \
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path /data/nfs/llm/model/chatglm-6b \
    --output_dir /home/guodong.li/output/adgen-chatglm-6b-pt \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 64 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --num_train_epochs 10 \
    --logging_steps 10 \
    --save_steps 100 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN
