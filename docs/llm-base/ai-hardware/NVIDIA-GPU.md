




Nvidia下游市场分为四类：游戏、专业可视化、数据中心、汽车，各市场重点产品如下：

游戏：GeForce RTX/GTX系列GPU（PCs）、GeForce NOW（云游戏）、SHIELD（游戏主机）；

专业可视化：Quadro/RTX GPU（企业工作站）；

数据中心：基于GPU的计算平台和系统，包括DGX（AI服务器）、HGX（超算）、EGX（边缘计算）、AGX（自动设备）；

汽车：NVIDIA DRIVE计算平台，包括AGX Xavier（SoC芯片）、DRIVE AV（自动驾驶）、DRIVE IX（驾驶舱软件）、Constellation（仿真软件）



消费级：https://www.nvidia.cn/geforce/graphics-cards/40-series/rtx-4090/

生产级：https://www.nvidia.cn/data-center/a100/





不同GPU型号的计算能力：https://developer.nvidia.com/cuda-gpus#compute




## DGX

- https://docs.nvidia.com/dgx-systems/
- https://docs.nvidia.com/dgx/pdf/dgx2-user-guide.pdf




## DGX-H100


- 官网详细介绍：https://docs.nvidia.com/dgx/dgxh100-user-guide/introduction-to-dgxh100.html
- AI芯片白皮书下载：https://www.nvidia.cn/data-center/dgx-a100/



HGX H100 8-GPU 是新一代 Hopper GPU 服务器的关键组成部分。它拥有八个 H100 张量核 GPU 和四个第三代 NV 交换机。每个 H100 GPU 都有多个第四代 NVLink 端口，并连接到所有四个 NVLink 交换机。每个 NVSwitch 都是一个完全无阻塞的交换机，完全连接所有八个 H100 Tensor Core GPU 。


NVSwitch 的这种完全连接的拓扑结构使任何 H100 都可以同时与任何其他 H100 通信。值得注意的是，这种通信以每秒 900 千兆字节（ GB/s ）的 NVLink 双向速度运行，这是当前 PCIe Gen4 x16 总线带宽的 14 倍多。

第三代 NVSwitch 还为集体运营提供了新的硬件加速，多播和 NVIDIA 的网络规模大幅缩减。结合更快的 NVLink 速度，像all-reduce这样的普通人工智能集体操作的有效带宽比 HGX A100 增加了 3 倍。集体的 NVSwitch 加速也显著降低了 GPU 上的负载。



HGX H100 拥有八个 H100 Tensor Core GPU 和 四个第三代 NV 交换机。每个 H100 GPU 都有多个第四代 NVLink 端口，并连接到所有四个 NVLink 交换机。每个 NVSwitch 都是一个完全无阻塞的交换机，完全连接所有八个 H100 Tensor Core GPU 。

NVSwitch 的这种完全连接的拓扑结构使任何 H100 都可以同时与任何其他 H100 通信。值得注意的是，这种通信以每秒 900 千兆字节（ GB/s ）的 NVLink 双向速度运行，这是当前 PCIe Gen4 x16 总线带宽的 14 倍多。



## DGX-A100


- https://docs.nvidia.com/dgx/dgxa100-user-guide/introduction-to-dgxa100.html


- [NVIDIA GPU A100 Ampere(安培) 架构深度解析](https://blog.csdn.net/han2529386161/article/details/106411138)
- [GPU 进阶笔记（一）：高性能 GPU 服务器硬件拓扑与集群组网（2023）](https://arthurchiao.art/blog/gpu-advanced-notes-1-zh/)

- [GPU 进阶笔记（二）：华为 GPU 相关（2023）](https://arthurchiao.art/blog/gpu-advanced-notes-2-zh/)

- [NVIDIA DGX H100 介绍](https://www.foresine.com/news/465-cn.html)

## H200

- https://www.nvidia.com/en-us/data-center/h200/













