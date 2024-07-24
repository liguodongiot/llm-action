

NCCL 通信库仅针对 Nvidia Spectrum-X 和 Nvidia InfiniBand 进行了优化。

博通 Tomahawk 5 以太网方案，客户需要有足够的工程能力来为 Tomahawk 5 适配及优化英伟达的 NCCL 通信库。


- 环境变量：https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html




NCCL_DEBUG=WARN


NCCL_SOCKET_IFNAME==ens1f0



ldconfig -p | grep libnccl

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH


```
yum install libnccl libnccl-devel libnccl-static
```





