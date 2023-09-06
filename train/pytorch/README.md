

- 镜像：https://hub.docker.com/r/pytorch/pytorch


- https://github.com/pytorch/examples
- https://github.com/pytorch/examples.git
- https://github.com/pytorch/pytorch

---



- torch.distributed.get_rank() # 取得当前进程的全局序号
- torch.distributed.get_world_size() # 取得全局进程的个数
- torch.cuda.set_device(device) # 为当前进程分配GPU
- torch.distributed.new_group(ranks) # 设置组
- torch.cuda.current_device()


---






- https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
- 



## PyTorch 分布式训练



- PyTorch 分布式训练（一）：概述
- PyTorch 分布式训练（二）：数据并行
- PyTorch 分布式训练（三）：分布式自动微分
- PyTorch 分布式训练（四）：分布式优化器
- PyTorch 分布式训练（五）：分布式 RPC 框架
- 






## 问题排查


- 将环境变量 NCCL_DEBUG 设置为 INFO 以打印有助于诊断问题的详细日志。（export NCCL_DEBUG=INFO）
- 显式设置网络接口。（export NCCL_SOCKET_IFNAME=eth0）


