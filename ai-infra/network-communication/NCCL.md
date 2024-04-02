




- 环境变量：https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html




NCCL_DEBUG=WARN


NCCL_SOCKET_IFNAME==ens1f0



ldconfig -p | grep libnccl

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH


```
yum install libnccl libnccl-devel libnccl-static
```





