#!/bin/bash

# Runs the "345M" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

#CHECKPOINT_PATH=<Specify path>
#VOCAB_FILE=<Specify path to file>/gpt2-vocab.json
#MERGE_FILE=<Specify path to file>/gpt2-merges.txt
#DATA_PATH=<Specify path and file prefix>_text_document

CHECKPOINT_PATH=/workspace/model/megatron-models/345m
VOCAB_FILE=/workspace/model/gpt2-vocab/gpt2-vocab.json
MERGE_FILE=/workspace/model/gpt2-vocab/gpt2-merges.txt
#DATA_PATH=/workspace/data/merged_cleand.json
DATA_PATH=/workspace/data/my-gpt2_text_document
MODEL_PATH=/workspace/model/megatron-models/output

GPT_ARGS="
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size 1 \
    --global-batch-size 2 \
    --lr 0.00015 \
    --train-iters 5000 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --data-impl mmap \
    --split 700,200,100
"

OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10
"

torchrun pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --save $MODEL_PATH \
    --load $CHECKPOINT_PATH
