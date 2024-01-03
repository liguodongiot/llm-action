



- centos.install.mellanox.gpudirect.md
- Assueme NVIDIA Driver and CUDA already successfully installed.
- https://gist.github.com/1duo/666d749ac7bf24ac4cc4f67984756edf



- InfiniBand Software：https://developer.nvidia.com/networking/infiniband-software


## Linux Drivers

### NVIDIA MLNX_OFED

OpenFabrics Alliance (www.openfabrics.org) 的 OFED 通过高性能 I/O 供应商的协作开发和测试得到了强化。

MLNX_OFED 是经过 NVIDIA 测试和封装的 OFED 版本，支持两种使用相同 RDMA（远程 DMA）的互连类型和称为 OFED verbs 的内核旁路 API – InfiniBand 和以太网。

支持高达 400Gb/s 的 InfiniBand 和超过 10/25/40/50/100/200/400GbE 的 RoCE（基于融合以太网标准的 RDMA）。


### Linux 内置驱动程序

适用于以太网和 InfiniBand 适配器的 Linux 驱动程序，也可在所有主要发行版（RHEL、SLES、Ubuntu 等）的收件箱中找到。



## libibverbs

libibverbs：用于直接用户空间使用 RDMA (InfiniBand/iWARP/RoCE) 硬件的库和驱动程序



libibverbs 是一个库，允许程序使用 RDMA "verbs 从用户空间直接访问 RDMA（当前是 InfiniBand 和 iWARP）硬件。





## MLNX_OFED GPUDirect RDMA

nvidia-peer-memory


GPU-GPU 通信的最新进展是 GPUDirect RDMA。 该技术在 GPU 内存与 NVIDIA 网络适配器设备之间提供直接的 P2P（点对点）数据路径。 这显着减少了 GPU-GPU 通信延迟，并完全卸载了 CPU，将其从网络上的所有 GPU-GPU 通信中移除。 

GPU Direct 利用 NVIDIA 网络适配器的 PeerDirect RDMA 和 PeerDirect ASYNC™ 功能。



## NVIDIA HPC-X

提高消息通信的可扩展性和性能。
NVIDIA® HPC-X® 是一个综合软件包，包括消息传递接口 (MPI)、对称分层内存 (SHMEM) 和分区全局地址空间 (PGAS) 通信库以及各种加速包。 这个功能齐全、经过测试和打包的工具包使 MPI 和 SHMEM/PGAS 编程语言能够实现高性能、可扩展性和效率，并确保通信库通过 NVIDIA Quantum InfiniBand 网络解决方案得到全面优化。




