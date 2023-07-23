#!/bin/bash

# Runs the "345M" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6069
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

#CHECKPOINT_PATH=<Specify path>
#VOCAB_FILE=<Specify path to file>/gpt2-vocab.json
#MERGE_FILE=<Specify path to file>/gpt2-merges.txt
#DATA_PATH=<Specify path and file prefix>_text_document


CHECKPOINT_PATH=/workspace/model/megatron-models/345m-init-mp
VOCAB_FILE=/workspace/model/gpt2-vocab/gpt2-vocab.json
MERGE_FILE=/workspace/model/gpt2-vocab/gpt2-merges.txt
DATA_PATH=/workspace/data/my-gpt2_text_document
MODEL_PATH=/workspace/model/megatron-models/345m-init-mp

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 2 \
    --sequence-parallel \
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size 4 \
    --global-batch-size 16 \
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
    --save-interval 2000 \
    --eval-interval 1000 \
    --eval-iters 10
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $MODEL_PATH \
    --load $CHECKPOINT_PATH
