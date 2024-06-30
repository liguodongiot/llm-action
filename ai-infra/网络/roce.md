


AI场景下高性能网络技术RoCE v2介绍: https://mp.weixin.qq.com/s/XyMFst3w-d65u4fU7cgLPA


RoCE是基于 Ethernet的RDMA，RoCEv1版本基于网络链路层，无法跨网段，基本无应用。
RoCEv2基于UDP，可以跨网段具有良好的扩展性，而且吞吐，时延性能相对较好，所以是被大规模采用的方案。



## ofed驱动

使用RoCE v2之前需要安装相关驱动，也就是ofed软件栈。OFED (OpenFabrics Enterprise Distribution) 是一个开源的软件栈，用于在高性能计算 (HPC) 和数据中心环境中实现高性能网络通信。它是一组用于 InfiniBand 和以太网 RDMA (Remote Direct Memory Access) 技术的软件包和驱动程序的集合。



## 性能测试-perftest

perftest是ofed性能测试工具集。专门用于测试RDMA的性能

```
带宽测试
ib_send_bw -d mlx5_0

客户端
ib_send_bw -d mlx5_1 10.251.30.207
```








