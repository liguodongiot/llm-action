

- srun文档： https://slurm.schedmd.com/srun.html
- SCOW（web管理）：https://github.com/PKUHPC/SCOW



## MUNGE-免密

MUNGE (MUNGE Uid 'N' Gid Emporium)是一种用于创建和验证凭证的身份验证服务。它允许进程在
一组具有公共用户和组的主机中验证另一个本地或远程进程的UID和GID。





## pytorch




### 单机多卡
```

```


### 多机多卡 (slurm+torchrun)



- https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/slurm/sbatch_run.sh


```
#!/bin/bash

#SBATCH --job-name=multinode-example
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

srun torchrun \
--nnodes 4 \
--nproc_per_node 1 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:29500 \
/shared/examples/multinode_torchrun.py 50 10
```


## deepspeed


### 单机多卡

```
deepspeed --include localhost:0,1,2,3 train.py --deepspeed_config=ds_config.json -p 2 --steps=200
```


### 多机多卡

```
python -m torch.distributed.run --nproc_per_node=2 --nnode=2 --node_rank=0 --master_addr=10.99.2.xx \
--master_port=9901 train.py --deepspeed_config=ds_config.json -p 2 --steps=200


python -m torch.distributed.run --nproc_per_node=2 --nnode=2 --node_rank=1 --master_addr=10.99.2.xx \
--master_port=9901 train.py --deepspeed_config=ds_config.json -p 2 --steps=200
```


### 单机多卡+docker

```
sudo docker run -it --rm --gpus all \
--network=host \
--shm-size 4G \
-v /data/hpc/home/guodong.li/:/workspaces \
-v /data/hpc/home/guodong.li/.cache/:/root/.cache/ \
-w /workspaces/DeepSpeedExamples-20230430/training/pipeline_parallelism \
deepspeed/deepspeed:v072_torch112_cu117 /bin/bash

deepspeed --include localhost:4,5,6,7 --master_port 29001 train.py --deepspeed_config=ds_config.json -p 2 --steps=200
```


### 单机多卡+singularity


```
docker tag deepspeed/deepspeed:v072_torch112_cu117 harbor.aip.io/base/deepspeed:torch112_cu117
sudo docker push harbor.aip.io/base/deepspeed:torch112_cu117
SINGULARITY_NOHTTPS=1 singularity build deepspeed.sif docker://harbor.aip.io/base/deepspeed:torch112_cu117


singularity run --nv \
--pwd /workspaces/DeepSpeedExamples-20230430/training/pipeline_parallelism \
-B /data/hpc/home/guodong.li/:/workspaces:rw \
deepspeed.sif


export NCCL_IB_DISABLE=1 && export NCCL_SOCKET_IFNAME=bond0 && export CC=/opt/hpcx/ompi/bin/mpicc && deepspeed --include localhost:4,5,6,7 --master_port 29001 train.py --deepspeed_config=ds_config.json -p 2 --steps=200
```

### 单机多卡+singularity+slurm 


```
sbatch pp-standalone-singularity.slurm


squeue
scancel -v xx
```


---


```
srun --mpi=list 
```




### 多机多卡+singularity+slurm 





- --mpi：指定mpi类型为pmi2

```
sbatch pp-multinode-singularity.slurm
```














