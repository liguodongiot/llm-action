#!/bin/bash

#SBATCH --job-name=megatron-multinode-ib-30b-2 #作业名称
#SBATCH --partition=h800-ib-2 #分区
#SBATCH --output=log/%j.out #日志
#SBATCH --error=log/%j.out #日志

#SBATCH -N 8  # 指定机器数量
#SBATCH -c 80
#SBATCH --gres=gpu:8 #每台机器使用四张卡

export NCCL_DEBUG=info

#export NCCL_IB_DISABLE=0
#export NCCL_IB_HCA=mlx5_0
#export NCCL_PXN_DISABLE=1
#export NCCL_IB_TIMEOUT=22
#export NCCL_IB_RETRY_CNT=13
#export NCCL_IB_PCI_RELAXED_ORDERING=1

export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=bond0



MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST"|head -1)
MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$MASTER_HOST" hostname --ip-address|awk '{print $1}') #从中获取一个IP作为通信IP
MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID|tail -c 4)) #master通信端口

ENDPOINT_URL=$MASTER_HOST:$MASTER_PORT

echo "$MASTER_HOST --- ""$MASTER_ADDR --- ""$MASTER_PORT"


srun --mpi=pmix_v3 singularity run --nv \
--pwd /workspace/code/Megatron-DeepSpeed-llama-20230815 \
-B /data/hpc/home/guodong.li/workspace:/workspace:rw  \
megatron-deepspeed-v4-gcc.sif \
torchrun --nnodes 8 --nproc_per_node 8 --rdzv_id=$SLURM_JOBID --rdzv_backend=c10d --rdzv_endpoint=$ENDPOINT_URL  pretrain_gpt.py --tensor-model-parallel-size 16 --pipeline-model-parallel-size 4 --num-layers 80 --hidden-size 8192 --ffn-hidden-size 22016 --num-attention-heads 64 --micro-batch-size 1 --global-batch-size 8 --seq-length 2048 --max-position-embeddings 2048 --train-iters 1500 --save ./tmp-llama-65b --load ./tmp-llama-65b --data-path 1 /workspace/data/gpt2-data/my-gpt2_text_document 2 /workspace/data/gpt2-data/my-gpt2-1_text_document --data-impl mmap --tokenizer-type GPTSentencePieceTokenizer --tokenizer-model /workspace/model/llama-tokenizer/tokenizer.model --split 900,50,50 --distributed-backend nccl --lr 3e-4 --lr-decay-style cosine --min-lr 3e-5 --weight-decay 0.1 --clip-grad 1 --lr-warmup-iters 500 --optimizer adam --adam-beta1 0.9 --adam-beta2 0.95 --log-interval 1 --save-interval 2000 --eval-interval 1000 --eval-iters 10 --fp16 --no-query-key-layer-scaling --attention-dropout 0 --hidden-dropout 0 --use-rotary-position-embeddings --untie-embeddings-and-output-weights --swiglu --normalization rmsnorm --disable-bias-linear --deepspeed-activation-checkpointing --zero-stage=0 --deepspeed_config=./tmp/deepspeed.json --deepspeed


