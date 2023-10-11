#!/bin/bash

#SBATCH --job-name=multinode-example
#SBATCH --partition=a800 #分区
#SBATCH --output=log/%j.out #日志
#SBATCH --error=log/%j.err #日志

#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=4

NODELIST=$(scontrol show hostname $SLURM_JOB_NODELIST)
# 对第一个节点赋值为主节点
MASTER_NODE=$(head -n 1 <<< "$NODELIST")
# 计数器
NODE_COUNT=0
# 一共的节点数
NODE_NUM=($(echo $NODELIST | tr " " "\n" | wc -l))

# 打印
echo $SLURM_NODEID
echo $NODELIST
echo $MASTER_NODE
echo $NODE_NUM


export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=bond0



for NODE in $NODELIST; do
    if [ "$NODE" == "$MASTER_NODE" ]; then
         srun --nodes=1 --ntasks=1 -w $NODE \
         singularity run --nv \
        --pwd /workspaces/examples-main/distributed/minGPT-ddp/mingpt \
        -B /data/hpc/home/guodong.li/:/workspaces:rw \
        pytorch-multinode.sif \
        torchrun --nproc_per_node=4 \
        --nnodes=$NODE_NUM \
        --node_rank=0 \
        --master_addr=$MASTER_NODE \
        --master_port=29500 \
        main.py &
    else
        ((NODE_COUNT++))
        srun --nodes=1 --ntasks=1 -w $NODE \
        singularity run --nv \
        --pwd /workspaces/examples-main/distributed/minGPT-ddp/mingpt \
        -B /data/hpc/home/guodong.li/:/workspaces:rw \
        pytorch-multinode.sif \
        torchrun --nproc_per_node=4 \
        --nnodes=$NODE_NUM \
        --node_rank=$NODE_COUNT \
        --master_addr=$MASTER_NODE \
        --master_port=29500 \
        main.py &
    fi
done
wait

