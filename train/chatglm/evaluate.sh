PRE_SEQ_LEN=128
CHECKPOINT=adgen-chatglm-6b-pt-128-2e-2
STEP=3000


# --model_name_or_path /data/nfs/llm/model/chatglm-6b \
# --ptuning_checkpoint /home/guodong.li/output/adgen-chatglm-6b-pt-128-2e-2/checkpoint-500 \
#   --predict_with_generate \


CUDA_VISIBLE_DEVICES=1 python3 main.py \
    --do_predict \
    --validation_file /data/nfs/llm/data/AdvertiseGen/dev.json \
    --test_file /data/nfs/llm/data/AdvertiseGen/dev.json \
    --overwrite_cache \
    --prompt_column content \
    --response_column summary \
    --model_name_or_path /home/guodong.li/output/adgen-chatglm-6b-ft-1e-4 \
    --output_dir /home/guodong.li/output/adgen-chatglm-6b-ft-1e-4 \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 64 \
    --per_device_eval_batch_size 1 \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4

