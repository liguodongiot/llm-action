#!/bin/sh

#SBATCH --job-name=multinode-deepspeed        # name

#SBATCH -N 2
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4

#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --partition=a800 # 分区
#SBATCH --output=log/%j.out # 日志
#SBATCH --error=log/%j.err # 日志


export GPUS_PER_NODE=2
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901


module load anaconda/3-2023.03

source activate
conda activate liguodong-310

module load cuda-cudnn8.9/11.7.1

cd /data/hpc/home/guodong.li/DeepSpeedExamples-20230430/training/pipeline_parallelism

srun --jobid $SLURM_JOBID bash -c 'python -m torch.distributed.run \
 --nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
 --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
train.py --deepspeed_config=ds_config.json -p 2 --steps=200'