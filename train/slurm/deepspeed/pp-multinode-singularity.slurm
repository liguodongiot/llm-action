#!/bin/sh

#SBATCH --job-name=multinode-deepspeed-singularity        # name

#SBATCH -N 2
#SBATCH --ntasks=4
#SBATCH --gres=gpu:2
###SBATCH --cpus-per-task=1

#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --partition=a800 # 分区
#SBATCH --output=log/%j.out # 日志
#SBATCH --error=log/%j.err # 日志

#export GPUS_PER_NODE=2
#export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
#export MASTER_PORT=9903

export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=bond0
export CC=/opt/hpcx/ompi/bin/mpicc
export CUDA_LAUNCH_BLOCKING=1

srun --mpi=pmi2 singularity run --nv --pwd /workspaces/DeepSpeedExamples-20230430/training/pipeline_parallelism \
-B /data/hpc/home/guodong.li/:/workspaces:rw \
deepspeed.sif \
python -m torch.distributed.run \
--nproc_per_node 2 \
train.py --deepspeed_config=ds_config.json -p 2 --steps=200

