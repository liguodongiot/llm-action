


- https://docs.nvidia.com/gpudirect-storage/design-guide/index.html
- https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html
- https://docs.nvidia.com/gpudirect-storage/getting-started/index.html



随着 AI 和 HPC 数据集的大小不断增加，为给定应用程序加载数据所花费的时间开始对整个应用程序的性能造成压力。 在考虑端到端应用程序性能时，快速的 GPU 通过缓慢的 I/O 将显著降低GPU的利用率。

I/O 是将数据从存储加载到 GPU 进行处理的过程，历来由 CPU 控制。 随着计算从较慢的 CPU 转移到更快的 GPU，I/O 越来越成为整体应用程序性能的瓶颈。

正如 GPUDirect RDMA（远程直接内存地址）在网络接口卡 (NIC) 和 GPU 内存之间直接移动数据时改善了带宽和延迟一样，一种名为 GPUDirect Storage 的新技术支持本地或远程存储（例如：NVMe 或 NVMe over Fabric (NVMe-oF)）与GPU内存之间的直接移动数据。 




## GPUDirect Storage

在GPU加速系统当中，所有的IO操作都会先经过主机端，也就是需要经过CPU指令把数据传到主机内存里，然后才会到达GPU，CPU通常会通过“bounce buffer”来实现数据传输，“bounce buffer”是系统内存中的一块区域，数据在传输到GPU之前会在这里保存一个副本。很明显，这种中转会引额外延迟和内存消耗，降低运行在GPU上的应用程序的性能，还会占用CPU资源，这就是GPUDirect Storage要解决的问题。






## GDS 的工作原理

NVIDIA 力求尽可能采用现有标准，并在必要时扩展这些标准。 POSIX 标准的 pread 和 pwrite 提供存储和 CPU 缓冲区之间的复制，但尚未启用到 GPU 缓冲区的复制。 Linux 内核中不支持 GPU 缓冲区的缺点将随着时间的推移得到解决。

一种名为 dma_buf 的解决方案正在开发中，该解决方案可以在 NIC 或 NVMe 和 GPU 等 PCIe 总线上的对等设备之间进行复制，以解决这一问题。 

与此同时，GDS 的性能提升空间太大，无法等待上游解决方案传播给所有用户。 许多供应商都提供了支持 GDS 的替代解决方案，如：MLNX_OFED。 GDS 解决方案涉及新的 API：cuFileRead 或 cuFileWrite，它们与 POSIX pread 和 pwrite 类似。

动态路由、NVLink 的使用以及只能从 GDS 获得的用于 CUDA 流的异步 API 等优化使得 cuFile API 成为 CUDA 编程模型的持久特性，即使在解决了 Linux 文件系统中的缺陷之后也是如此。

以下是 GDS 实施的作用。 
首先，当前 Linux 实现的根本问题是将 GPU 缓冲区地址作为 DMA 目标向下通过虚拟文件系统 (VFS) 传递，以便本地 NVMe 或网络适配器中的 DMA 引擎可以执行与 GPU 内存之间的传输。 这会导致错误情况。 我们现在有一个解决这个问题的方法：传递 CPU 内存中缓冲区的地址。

当使用 cuFileRead 或 cuFileWrite 等 cuFile API 时，libcufile.so 用户级库捕获 GPU 缓冲区地址并替换传递给 VFS 的代理 CPU 缓冲区地址。 就在缓冲区地址用于 DMA 之前，启用 GDS 的驱动程序对 nvidia-fs.ko 的调用会识别 CPU 缓冲区地址并再次提供替代 GPU 缓冲区地址，以便 DMA 可以正确进行。

libcufile.so 中的逻辑执行前面描述的各种优化，例如：动态路由、预固定缓冲区的使用和对齐。 

图 2 显示了用于此优化的堆栈。 cuFile API 是 Magnum IO 灵活抽象架构原则的一个示例，可实现特定于平台的创新和优化，例如选择性缓冲和 NVLink 的使用。

图 2. GDS 软件堆栈，其中应用程序使用 cuFile API，并且支持 GDS 的存储驱动程序调用 nvidia-fs.ko 内核驱动程序来获取正确的 DMA 地址。