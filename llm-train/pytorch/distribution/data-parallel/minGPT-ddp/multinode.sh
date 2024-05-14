#!/bin/bash

#SBATCH --job-name=multinode-example
#SBATCH --partition=a800 #分区
#SBATCH --output=log/%j.out #日志
#SBATCH --error=log/%j.err #日志

#SBATCH -N 2
#SBATCH --ntasks=2
#SBATCH --gres=gpu:4
##SBATCH --cpus-per-task=4

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


srun --mpi=pmix_v3 singularity run --nv --pwd /workspaces/examples-main/distributed/minGPT-ddp/mingpt -B /data/hpc/home/guodong.li/:/workspaces:rw pytorch-multinode.sif torchrun --nproc_per_node=4 main.py
