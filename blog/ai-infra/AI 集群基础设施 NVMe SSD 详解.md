
随着 AI 和 HPC 数据集的大小不断增加，为给定应用程序加载数据所花费的时间开始对整个应用程序的性能造成压力。 在考虑端到端应用程序性能时，快速的 GPU 通过缓慢的 I/O 将显著降低GPU的利用率。

I/O 是将数据从存储加载到 GPU 进行处理的过程，历来由 CPU 控制。 随着计算从较慢的 CPU 转移到更快的 GPU，I/O 越来越成为整体应用程序性能的瓶颈。

正如 GPUDirect RDMA（远程直接内存地址）在网络接口卡 (NIC) 和 GPU 内存之间直接移动数据时改善了带宽和延迟一样，一种名为 GPUDirect Storage 的新技术支持本地或远程存储（例如：NVMe 或 NVMe over Fabric (NVMe-oF)）与GPU 内存之间的直接移动数据。

本文将针对硬盘的发展、NVMe的技术原理以及NVMe在AI服务器及DeepSpeed框架中的应用进行详细的介绍。

> 文章较长，建议先点赞收藏，后续再慢慢观看。另外，我撰写的**大模型相关的博客及配套代码**均整理放置在Github：[llm-action](https://github.com/liguodongiot/llm-action/tree/main)，有需要的朋友自取。

## 衡量硬盘传输速度的三大要素

要知道一块硬盘真正的性能，就要知道如何衡量一块硬盘速度？通常情况下，我们从以下三个方面进行评估：通讯协议（总线）、物理接口标准、传输通道。

### 传输通道/总线

总线是计算机系统内部各个组件之间传输数据的通道。硬盘通过总线与其他系统组件（如主板、内存等）进行通信。总线的速度直接影响着硬盘的数据传输速度。

常见的总线/通道类型包括SATA（Serial ATA）和PCI Express。SATA是较为常见的硬盘连接总线，而PCI Express则通常用于更高性能的存储解决方案。其他的通道有SAS通道（企业级别硬盘用的通道）、FC通道（光纤通道）等。

### 通讯协议

协议定义了数据在总线上的传输方式和规则。其内容主要包括设备间如何相互识别、如何建立链接、使用的讯号类型、数据的编码解码方式、数据传输的类型、数据传 输的方式以及物理层面上的电压、电流、保持时间和截止时间等。

只有当两个设备之间的协议相同或者相容时，才可以正常进行通讯。不同协议能够支持的最大传输速率也不同。

常见的通信协议如下：

| 协议名称 | 应用场合 |                                                                            说明                                                                           |
| :--: | :--: | :-----------------------------------------------------------------------------------------------------------------------------------------------------: |
|  IDE |  民用  |                                                               机械硬盘时代，用于数据操作，传输的协议。目前已被淘汰。                                                               |
| AHCI |  民用  |                                       仍然是机械硬盘时代的主流数据传输协议。但SATA固态硬盘和极少数PCIe固态硬盘可以使用此协议，用SATA通道优化后的效率相比IDE提升10-30%。                                       |
| NVMe |  民用  |                            由于机械硬盘和固态硬盘的工作模式发生巨大变化，需要一种全新的针对固态的传输层协议，NVMe 应运而生。NVMe协议相比传统的AHCI协议具有更低的延迟和更高的数据传输速度，适用于高性能要求的应用。                           |
| SCSI |  服务器 | SCSI（Small Computer System Interface）协议是一种通用的存储协议，最初是一种并行接口，后来也演变为串行接口（SAS，Serial Attached SCSI）。 常见于企业级固态硬盘，消费级市场不常见。但其设计并非专门针对固态硬盘，可能在利用硬件特性方面不如NVMe。 |

不同的硬盘可以采用不同的数据传输协议。例如，SATA 上的硬盘通常使用AHCI（Advanced Host Controller Interface）协议，而PCIe（PCI Express）上的硬盘可能使用NVMe（Non-Volatile Memory Express）协议。

### 物理接口

接口是硬盘与计算机或其他设备之间物理连接的方式。硬盘接口通常是指连接到主板的插槽或端口。常见的硬盘接口包括 M.2、SATA、PCI Express等。

不同的接口也会影响硬盘的数据传输速度，因为不同接口支持的总线和协议可能有所不同。

常见的硬盘接口类型：

*   **PCI Express（PCIe）：** PCI Express接口用于连接高性能固态硬盘（NVMe SSD）和一些扩展卡。NVMe（Non-Volatile Memory Express）是一种用于固态存储的高效协议，通常与PCI Express接口结合使用，提供更高的传输速度和更低的延迟。
*   **M.2（NGFF）：** M.2是一种小型、高密度的接口，用于连接固态硬盘和一些无线通信设备。M.2接口通常支持**AHCI协议**（走SATA通道）和**NVMe协议**（走PCI Express通道），适用于轻薄型笔记本电脑和台式机。
*   **SATA（Serial ATA）：** SATA是目前最为普遍的硬盘接口之一，用于连接传统的机械硬盘（HDD）和固态硬盘（SSD）。它是一种串行数据传输接口，提供了不同版本，如：SATA I（1.5 Gbit/s）、SATA II（3 Gbit/s）、SATA III（6 Gbit/s）等。
*   **SAS（Serial Attached SCSI）：** SAS是一种用于连接企业级硬盘的接口，提供了更高的性能和可靠性。SAS接口通常用于服务器和存储系统，支持更高的数据传输速度和更多的同时连接。
*   **mSATA：** mSATA是一种用于连接小型存储设备的接口，通常用于一些较老的笔记本电脑和嵌入式系统。它支持SATA。
*   **IDE（PATA）：** IDE（Integrated Drive Electronics）是一种较老的硬盘接口，也称为PATA（Parallel ATA）。这种接口用于连接传统的机械硬盘和光盘驱动器，已经逐渐被SATA接口所取代。
*   **SCSI**：SCSI是一种用于连接计算机和外部设备的通用接口标准，也用于连接一些服务器和存储设备上的硬盘。SCSI有多个版本，包括并行SCSI和串行SCSI。

总之，硬盘接口技术在不断进化革新，从早期的IDE、SCSI接口到主流的SATA、SAS接口，再到M.2、PCIe接口。

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/d4f1e6d05a0c49b58a20a7593e74f8a5~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1144\&h=720\&s=2475831\&e=png\&b=fdfdfd)

可以看到，SATA、PCIe 即是一种**物理接口标准**，也是**总线（通道）标准**，其中，SATA接口使用AHCI通讯协议；PCIe接口通常使用NVMe通讯协议，也可以使用AHCI通讯协议。

## 硬盘

### 机械硬盘

机械硬盘（HDD）是一种带有机械马达结构的存储装置，主要带有马达、盘片、磁头、缓存。

![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/0b870c11c5dd4bd3a5d2495e12958eed~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=920\&h=720\&s=1991213\&e=png\&b=eeeceb)

### 固态硬盘

固态硬盘（SSD）是一种不带有机械马达结构的存储装置，主要带有闪存、主控芯片、缓存。

![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/b2098c24372f471c9caf6b9dc8540163~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=986\&h=577\&s=565723\&e=png\&b=fcfbfc)

### 固态硬盘外形尺寸

固态硬盘存在多种外形和尺寸，如：2.5 英寸、M.2、mSATA 和 U.2。

**2.5"(2.5 英寸)**：最常见的固态硬盘类型，适用于多数笔记本电脑或台式机。其外形类似传统机械硬盘 (HDD) 并通过 SATA 线缆连接，因此使用起来与众多现有产品非常类似。

**M.2**：另一种外形尺寸， M.2 已变成纤薄便携式计算机和笔记本电脑的标配存储类型。这种小巧的外形尺寸常常类似于一片口香糖，在多数情况下可轻松安装到主板上。它具备各种不同长度，可实现不同的固态硬盘存储容量；硬盘越长，可搭载的 NAND 闪存芯片越多，从而实现更高存储容量。

**mSATA**：mSATA 或 mini-SATA 是全尺寸 SATA 固态硬盘的缩小版。它像 M.2 一样使用紧凑的外形尺寸，但两者不可互换。M.2 硬盘支持 SATA 和 PCIe 两种接口选项，而 mSATA 仅支持 SATA。这种外形尺寸专为空间受限的小型系统设计。

**U.2** ：它看起来像 2.5 英寸硬盘，但略微厚一点。它使用不同的连接方式，并通过 PCIe 接口发送数据。U.2 固态硬盘技术通常用于需要更大存储容量的高端工作站、服务器和企业应用。它支持更高工作温度，比 M.2 外形尺寸更利于散热。

![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/0b07f53e501e46ec9a9ae22bbddc2214~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=450\&h=800\&s=163542\&e=png\&b=ffffff)

### 固态硬盘接口类型

常见的固态硬盘接口类型：

*   **SATA（Serial ATA）：** SATA接口是一种常见的硬盘接口，广泛用于连接传统的机械硬盘和固态硬盘。SATA接口有不同的版本，包括：SATA I（1.5 Gbit/s）、SATA II（3 Gbit/s）、SATA III（6 Gbit/s）等。
*   **PCI Express（PCIe）：** PCI Express接口是一种高速串行总线，用于连接固态硬盘和其他高性能设备。通过PCIe接口连接的固态硬盘通常使用NVMe（Non-Volatile Memory Express）协议，提供更高的数据传输速度和更低的延迟。
*   **M.2（NGFF）：** M.2是一种小型、高密度的接口，广泛用于连接固态硬盘和无线通信设备。M.2接口支持SATA和PCI Express协议，适用于轻薄型笔记本电脑和台式机。
*   **mSATA：** mSATA是一种较老的小型接口，用于连接固态硬盘。它采用SATA协议，通常用于一些较老的笔记本电脑和嵌入式系统。
*   **U.2：** U.2是一种适用于企业级硬盘的接口，支持SATA和PCI Express协议。U.2接口通常用于连接高性能企业级固态硬盘，提供更大的功率和散热能力。
    ![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/58606ac104bc4a09b9db26560d488277~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=500\&h=300\&s=121643\&e=png\&b=fcfbfb)

### M.2 固态硬盘类型

M.2 固态硬盘包含 SATA 和 NVMe（使用PCIe） 两种类型。

注意：M.2 固态硬盘仅兼容支持 M.2 插槽的主板。检查计算机的主板，确保包含 M.2 插槽。

M.2接口类型分为Socket 2和Socket 3：

*   Socket 2：也可以叫做B key，支持sata，pcie x2 通道。
*   Socket 3：也可以叫做M key，支持sata，pcie x4 通道。

一开始，B key的只能插在B key（Socket 2）的接口中，M key的只能插在 M key(Socket 3)的接口中，但是**随着M key接口的普及，越来越多电脑主板只有M key 接口，B key的固态硬盘根本插不上去，于是厂商们又设计了一个B\&M key接口的固态硬盘（SSD）**。

B\&M key接口即可以插上B key也可以插上M key。**B\&M key支持的通道和B key支持的通道一样，都是sata和pcie x2 通道**，但是 **B\&M key可以兼容 M key 和 B key两种，而B key只能兼容B key一种，这就导致了B key毫无优势，B key被B\&M key取代**，现在市面上只有B\&M key和M key两种M.2 ssd卖, B key的 M.2  SSD 已经绝迹。

注意：固态硬盘（SSD）的金手指有B key，B key ，B\&M key三种，但是主板上的M.2接口只有B key和M key两种。

**SATA M.2 固态硬盘**：

**SATA 固态硬盘是性能最低的固态硬盘，采用的接口与机械硬盘相同**。尽管如此，SATA 固态硬盘的带宽是旋转式机械硬盘的三到四倍。SATA M.2 固态硬盘使用的 SATA 接口最大数据传输速率为 6Gbps。SATA 固态硬盘比 NVMe 固态硬盘更加普及、更加便宜。如果计算机没有空间安装 2.5 英寸固态硬盘，SATA M.2 固态硬盘可能是 2.5 英寸固态硬盘的出色替代选项。 

如图所示，同时具有 M 键和 B 键的 M.2 固态硬盘将是 SATA 固态硬盘。

![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/2e453a69c4e34355bd553d859b5ef721~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=450\&h=250\&s=169390\&e=png\&b=e1e5e9)

SATA 一直是用于存储技术的主要接口。使用 SATA 线缆的 SATA 硬盘需要两根线缆才能工作。一根用于将数据传输到主板，另一根用于连接 PSU（电源）。当使用多个 SATA 存储硬盘时，杂乱无章的线缆是可能影响 PC 机箱性能的问题之一。包括超级本在内的纤薄笔记本电脑和便携式计算机甚至没有空间来安放 SATA 线缆，所以需要采用 M.2 的外形尺寸。SATA M.2 外形尺寸的固态硬盘解决了这个问题，它没有其他 SATA 存储硬盘所使用的两个线缆连接。

**NVMe M.2 固态硬盘**：

如图所示，只有 M 键的 M.2 固态硬盘将是 NVMe 固态硬盘。  **NVMe M.2 固态硬盘采用了专为固态硬盘设计的 NVMe 协议**。与 PCIe 总线配合，NVMe 固态硬盘可以提供市面上最新水平的性能和速度。NVMe 固态硬盘利用 PCIe 插槽直接与系统 CPU 进行通信。基本上，它让闪存可以作为固态硬盘通过 PCIe 插槽进行运行，而不必使用速度比 NVMe 慢得多的 SATA 通信驱动程序。

![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/090ec4516e6e474491c5fd07969ac893~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=450\&h=550\&s=355389\&e=png\&b=f1f3f7)

NVMe M.2 固态硬盘的性能比 SATA M.2 固态硬盘高得多。通过利用 PCIe 总线，NVMe M.2 固态硬盘拥有高达 20Gbps 的理论传输速度，比 SATA M.2 固态硬盘的 6Gbps 快。**PCIe 总线支持 1x、4x、8x 和 16x 通道**。PCIe 3.0 的有效传输速度高达每通道 985 MB/秒，这意味着潜在传输速度高达 16GB/秒。不过，**当使用 M.2 外形尺寸与 PCIe 总线时，只能访问 x2 和 x4 通道**，这使得最大传输速度为 4GB/秒。

**SATA 固态硬盘与 NVMe M.2 固态硬盘的区别**：

SATA 固态硬盘与 NVMe M.2 固态硬盘之间的主要区别在于接口技术和性能水平。SATA M.2 固态硬盘仍然采用 SATA 接口设计，这无法改进速度与性能，毕竟这不是 NVMe M.2 固态硬盘。

NVMe允许SSD通过PCIe总线直接连接到CPU以通过高速通道传输数据。单个第四代PCIe通道可以传输高达**2,000 MB/s**的数据，NVMe SSD最多使用其中四个。相比之下，SATA只有一个lane，最高可以传输**600MB/s**。

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/8daa735866ed40eb8be6a293ff2dda77~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1508\&h=596\&s=336666\&e=png\&b=d9e2fb)

SATA SSD AHCI驱动程序只有一个可用队列，每个队列有32个命令。而 NVMe 允许多达65,535个队列，每个队列的最大深度为65,536个命令。

使用NVMe技术，由于I/O处理门铃信号的高性能，CPU可以更有效地管理队列，从而降低CPU开销。低CPU开销会导致CPU周期减少。相比之下，SATA SSD在I/O处理中产生较高的CPU周期。

与SATA SSD相比，NVMe技术缩短和优化了数据路径，从而降低了延迟。它产生大约2.8微秒的延迟，而SATA SSD有大约6微秒的延迟——比NVMe SSD长了近3微秒。

NVMe SSD最适合企业工作负载处理和人工智能、机器学习项目、实时分析、大数据传输和DevOps。它们通常用于数据中心、高端笔记本电脑和预制台式电脑。而 SATA SSD 最适合小数据分析和各种轻量级存储应用程序。它们主要用于预算笔记本电脑和服务器。

在价格方面，与SATA SSD相比，NVMe SSD更贵。

![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/c4d9099fe41641d28489fa59350d9179~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1510\&h=568\&s=401661\&e=png\&b=d8e0fb)

总之，由于 NVMe 利用 PCIe 插槽，它传输的数据量是同等 SATA 产品的 25 倍。除了更多数据，NVMe 命令的速度是 AHCI 驱动程序命令的 2 倍。此外，NVMe 的每秒输入/输出操作 (IOPS) 超过 100 万，是 AHCI 硬盘的 900%。得益于自身的兼容性，NVMe 还直接与系统 CPU 通信，具有惊人的速度。同时，NVMe 硬盘兼容所有主要的操作系统。

![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/f8d2effd638e41988430717dd9436606~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=992\&h=620\&s=86030\&e=png\&b=ffffff)

**通信驱动器程序：AHCI 与 NVMe 对比**：

通信驱动程序被操作系统用来与存储设备交换数据。NVMe 驱动程序比常见于 SATA 接口的 AHCI 驱动程序速度快。

*   NVMe专为采用闪存技术的 SSD 设计，速度远超专为采用旋转磁盘技术的普通机械硬盘设计的 AHCI 驱动程序。
*   NVMe 拥有 64000 个命令队列，可以每个队列发送 64000 条命令，而 AHCI 只有一个命令队列，每个队列只能发送 32 条命令。
*   利用 AHCI 驱动程序，命令利用高 CPU 周期，延迟为 6 微秒，而 NVMe 驱动程序命令利用低 CPU 周期，延迟为 2.8 微秒。

NVMe驱动程序直接与系统 CPU 通信，而 AHCI 必须与 SATA 控制器通信。AHCI 的 IOPS（每秒输入/输出操作）最高 10 万，而 NVMe 的 IOPS 超过 100 万。IOPS（每秒输入/输出操作，发音是 i-ops）是用来对计算机存储设备进行基准测试的常见性能衡量指标。

![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/5eb17b23064f4b9ea40d8331a5f5056b~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=932\&h=874\&s=368173\&e=png\&b=fbfbfb)

### 常见硬盘规格

| 名称             | 接口           | 总线        | 协议   | 接口理论速率上限 |
| -------------- | ------------ | --------- | ---- | -------- |
| 西部数据蓝盘1TB      | SATA         | SATA3.0   | AHCI | 600MB/s  |
| 固态铠侠TC10(SATA) | SATA         | SATA3.0   | AHCI | 600MB/s  |
| 三星860EVO(M.2)  | M.2 B\&M-Key | SATA3.0   | AHCI | 600MB/s  |
| 三星SM951(AHCI)  | M.2 M-Key    | PCIe3.0×4 | AHCI | 4GB/s    |
| 三星XP941        | M.2 M-Key    | PCIe2.0×4 | NVMe | 2GB/s    |
| 西部数据SN500      | M.2 B\&M-Key | PCIe3.0×2 | NVMe | 2GB/s    |
| 西部数据SN750      | M.2 M-Key    | PCIe3.0×4 | NVMe | 4GB/s    |
| 三星980PRO       | M.2 M-Key    | PCIe4.0×4 | NVMe | 8GB/s    |
| 三星983ZET       | PCIe         | PCIe3.0×4 | NVMe | 4GB/s    |
| Intel SSD I910 | PCIe         | PCIe2.0×8 | SCSI | 4GB/s    |
| 希捷银河4T         | SAS          | SAS3.0    | SCSI | 1.2GB/s  |
| Intel P4510    | U.2          | PCIe3.0×4 | NVMe | 4GB/s    |

## NVMe

### NVMe 简介

NVMe （non-volatile memory express）是一种高性能、NUMA（非统一内存访问）优化的、高度可扩展的**存储协议**，用于连接主机和内存子系统。NVMe是专门为NAND、闪存等非易失性存储设计的，NVMe协议建立在高速PCIe通道上。它可以使我们能够充分利用SSD和存储类内存（SCM）的速度。

### NVMe 的主要特征

1.  为PCIe制定的标准接口协议。解除了旧标准施放在SSD上的各种限制。
2.  支持所有常见的操作系统。
3.  良好的可拓展性。
4.  采用多队列设计，支持64K命令队列：提高数据传输速度，因为数据是利用芯片和块以分散形式写入 SSD 的，而不是像机械硬盘一样在旋转的磁盘上写入数据。
5.  可以使用低CPU周期为每个队列发送64K命令；
6.  延迟约为2.8微秒；
7.  可以直接与系统CPU通信；
8.  可以实现超过一百万的IOPS。

### NVMe 的发展史

在过去的十年中，存储技术发生了翻天覆地的变化。随着固态硬盘开始取代机械硬盘成为主要的存储设备，我们急需一个新的接口标准来利用更快的速度和功能。

传统的SATA接口与AHCI标准其实是为了机械硬盘而设计的，早期的SSD性能不高，即使使用这些传统的接口和协议，也不觉得有什么问题，但是随着SSD的性能逐渐增强，传统的标准已经不再适用，进而成为了限制SSD的一大瓶颈。NVMe是第一个真正满足高速存储介质需求的协议。

2009年下半年，NVM Express工作组（NVMHCI）开始制定NVMe规范，NVM Express工作组包含90多家公司成员，Intel是主要领头人，小组成员包括美光、戴尔、三星、Marvell、NetAPP、EMC、IDT等公司，目的就是为SSD建立新的存储规范标准，让它在老旧的SATA与AHCI中解放出来。

2011年，NVMe 1.0标准正式出炉，该标准是根据闪存存储的特点量身定制的，新的标准解除了旧标准施放在SSD上的各种限制。2012、2014、2017和2019年相继推出了1.1、1.2、1.3、1.4版本。

![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/81425d6ef93342c898e74859f98201ae~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=2130\&h=880\&s=631178\&e=png\&b=fefefe)

NVMe从1.0发展到1.4，逐渐形成 **NVMe Base Specification（NVMe）、NVMe Management Interface Specification（NVMe-MI）和NVMe Over Fabrics Specification（NVMe-oF）** 三大Spec合集，它们看似独立，但又彼此关联，相互依赖。

![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/ab898d2fa3ce4d51a122fee21eb0e645~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1023\&h=590\&s=95570\&e=png\&b=fefefe)

随着Spec的增多，如何组织这些Spec就成了新的问题。在NVMe 1.4架构下，如果想要添加新的命令集，或区分不同的Transports协议，使用当前Spec架构就会带来诸多不便，牵一发而动全身。因此，NVMe 2.0最重要的使命，就是通过对Spec结构进行调整以方便在最小化已有方案影响的前提下进行新的开发。

新的Spec架构更加趋于模块化，将Commend Set和Transport Spec从原有三大合集中独立出来，与Base Spec、NVMe-MI一同，构成了新的NVMe 2.0协议族，也方便未来有更多新的功能添加进来。

![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/21e1e6838355412f8d3d8b2e9acfbeae~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1080\&h=304\&s=79723\&e=png\&b=f9f1f0)

此外，NVM Express还对NVMe设备的启动做了规范（NVM Express Boot Specification）。因此，最新的NVMe协议族如下：

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/31e56e5c1c474334a13de29b59c56757~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1025\&h=426\&s=84180\&e=png\&b=efe8e2)

| 名词                                                           | 说明                                           |
| ------------------------------------------------------------ | -------------------------------------------- |
| I/O Command Set Specifications                               | 定义扩展NVM Express基本规范的数据结构、功能、日志页、命令和状态值。      |
| NVM Express Base Specification                               | 定义了主机软件通过各种基于存储器的传输和基于消息的传输与非易失性存储器子系统通信的协议。 |
| Transport Specifications                                     | 定义包括控制器属性的NVMe协议到特定传输的绑定。                    |
| The NVM Express Management Interface (NVMe-MI) Specification | 定义了所有NVM Express子系统的可选管理接口。                  |
| NVM Express Boot Specification (NVMe Boot)                   | 定义了从NVM Express接口启动的结构和指南。                   |

2021年，NVMe 2.0 协议族证书发布，它由8个具体的协议规范组成，其中，**NVM Command Set、Zoned Namespace Command Set、Key Value Command Set**共同组成了新的Command Set 协议族，Transport Spec 也被细分为PCIe、RDMA和TCP三种。

**NVM Commend Set Spec、PCIe Transport Spec** 是NVMe最开始想要实现的目标，在此基础上又有6个Spec被开发出来，可见NVMe的技术发展确实是非常快的。

![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/d170dac1e7314181a8a83aa7f81d4b0f~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1040\&h=401\&s=81203\&e=png\&b=fdfdfd)

NVMe 2.0定义了3种命令类型：**Admin命令、I/O命令和Fabrics命令**。作用跟之前版本一样。

*   Admin命令用于控制器的管理，有专用的Admin SQ/CQ来实现命令传递。
*   I/O命令使用I/O SQ/CQ，如我们熟悉的读和写，都由I/O命令完成，在NVMe 2.0协议族中，I/O命令集又分为**NVM命令集、Zoned Namespace命令集和Key Value命令集**三种。
*   Fabrics命令则专用于NVMe over Fabrics。

NVMe Transport是基于物理连接属性抽象的协议层，分为Memory-Based、Message-Based、Message/Memory混合型。

*   **Memory-Based Transport**：指Host和NVM Subsystem之间的命令、应答和数据是由显性的内存读写命令来完成的，代表协议：NVMe over PCIe Transport Spec。
*   **Message-Based Transport**：指Host和NVM Subsystem之间通过发送Message封装命令和响应数据，代表协议：Fiber Channel、TCP Transport Spec。
*   **Message/Memory 混合型 Transport**：组合使用Message和显性的内存读写命令，代表协议：RDMA Transport Spec。

![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/a094e5b008ac4ac9aa499115b34d185f~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=906\&h=365\&s=104141\&e=png\&b=fffefe)

### NVMe SSD 外形尺寸

NVMe SSD 存在多种不同的外形尺寸，但具体取决于用例或应用。

*   个人/客户端产品使用 BGA 和 M.2 外形尺寸。
*   数据中心/服务器应用使用 M.2、U.2、U.3 和 EDSFF 外形尺寸。

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/889afa23013847d8b1d58db88e041d17~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=900\&h=430\&s=124121\&e=png\&b=eeeeee)

### NVMe 协议在协议栈中的位置

NVMe是一种Host与SSD之间通讯的协议，它在协议栈中隶属高层。NVMe在协议栈中处于应用层，底层通常是PCIe协议栈，NVMe协议能够正常运行，是依赖于底层PCIe协议提供的一些功能，而PCIe协议是目前计算机硬件与CPU之间互联互通的关键，诸如网卡、显卡等等都在利用这套协议，可以说底层利用PCIe协议栈是最简单快速的方法。

![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/0226f894935a4b1d95a774f502d36cdf~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1320\&h=696\&s=448777\&e=png\&b=faf9f9)

理论上，NVMe协议不一定跑在PCIe协议栈之上，只要底层能够完成同样的任务，替换成别的协议框架也是可以的，就比如把NVMe移植到手机上，就可能发生一些变化。

**NVMe over Fabrics**就换掉了NVMe下层部分协议，具体如下图所示。

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/4d6b132435764979bab986f5885b5127~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=720\&h=565\&s=71985\&e=png\&b=8d8d8d)

### NVMe 命令

NVMe制定了Host与SSD之间通讯的命令，NVMe有两种命令：

*   一种叫Admin Command，用于Host管理和控制SSD；
*   另外一种就是I/O Command，用于Host和SSD之间数据的传输。

### NVMe 队列

NVMe 的队列分为2种，其中一种是用于管理的队列，称为Admin Queue（管理队列），仅有一个，另外一种是命令队列（Command Queue），最多可以有65535个。

其中，命令队列的数量和模式都是通过管理队列来设置的。每一个队列实际上是一个队列对，也就是包括两个队列，分别是提交队列（Submission Queue）和完成队列（Completion Queue）。**提交队列**用于主机端向NVMe设备发送NVMe命令，而**完成队列**则用于NVMe设备向主机反馈命令执行情况。

![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/7028fa5e992d468baff80a2287ae9a0e~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=552\&h=202\&s=122591\&e=png\&b=e3e2e2)

*   Admin Submission Queue和对应的Admin Completion Queue用来管理和控制主控器(如：创建和删除IO队列，终止命令等)，只有属于Admin Command Set的命令才会被提交到Admin Submission Queue。Admin Queue的ID都是0。
*   IO Submission Queues和对应的IO Completion Queues用来处理IO命令，规范定义了一种IO Command Set，叫做NVM Command Set，与IO队列一起使用。系统在创建Submission Queue前必须先创建相关的Completion Queue，同时，删除Submission Queue操作也要先于相关的Completion Queue。

实际上NVMe还有另外一种模式，就是多个提交队列共享同一个完成队列的情况。

![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/f433a128f7534631a63b3db4eb7bd837~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=552\&h=202\&s=116964\&e=png\&b=e3e2e2)

NVMe是通过队列传递控制命令和命令等内容。而提交队列和完成队列就是内存的一个区域，在数据结构原理上这里的队列其实是一个环形缓冲区。

![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/bc455f929dbb4084b56fc0bd4029cae6~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=619\&h=440\&s=124578\&e=png\&b=fcfcfc)

**NVMe通过一种门铃机制(Doorbell)来告知控制器命令队列是否有新数请求/命令**。也就是说每个队列都有一个门铃指针。对于发送队列来说，这个指针表示的是发送队列的尾指针。主机端将数据写入到发送队列后，**更新映射到位于设备寄存器空间中的门铃的尾指针**。此时，在控制器端就知道有新的请求/命令到来，接下来就可以进行对其进行处理。

当控制器完成一个NVMe请求时，通过完成队列来把完成的结果告知主机端。与发送队列不同，完成队列是通过\*\*中断机制（可以是INTx，MSI或MSIx）\*\*告诉主机端。

![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/9313c23aa41d483d9431ea26f5514565~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1280\&h=475\&s=2436481\&e=png\&b=ede2d5)

### 带 NVMe SSD 的 PCIe 拓扑

SSD 作为一个 PCIe Endpoint 通过 PCIe 连着 Root Complex （RC）；然后，RC连接着CPU和内存。

![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/6294983025214f4a9051d68a63e5efcf~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=683\&h=457\&s=187824\&e=png\&b=f4ebe5)

核心部分说明：

*   **Submission Queue（SQ）**：位于Host内存中，由系统放置命令，**Host要发送命令时，先把准备好的命令放在SQ中**，然后通知SSD来取；
*   **Completion Queue（CQ）**：也是位于Host内存中，**由控制器（NVMe）放置完成信息**。一个命令执行完成，成功或失败，SSD总会往CQ中写入命令完成状态。
*   **Doorbell Register（DB）**： 位于SSD的控制器内部，在Host发送命令时，不是直接往SSD中发送命令的，而是把命令准备好放在自己的内存中，Host就是通过写SSD端的DB寄存器来告知SSD命令已经处理完毕。
*   **Root Complex（RC）**：CPU和PCle总线之间的接口，可能包含几个组件(处理器接口、DRAM接口等)，甚至可能包含几个芯片。主要负责PCIe报文的解析和生成。RC接受来自CPU的IO指令，生成对应的PCIe报文，或者接受来自设备的PCIe TLP报文，解析数据传输给CPU或者内存。
*   **Switch**：提供扩展或聚合能力，并允许更多的设备连接到一个PCle端口。
*   NVM 子系统（Subsystem）：
*   **PCIe Endpoint** ： PCIe设备；处于PCIe总线系统拓扑结构中的最末端，一般作为总线操作的发起者或者终结者。显然，Endpoint只能接受来自上级拓扑的数据包或者向上级拓扑发送数据包。细分Endpoint类型的话，分为 Lagacy PCIe Endpoint 和 Native PCIe Endpoint。
    *   **Lagacy PCIe Endpoint**：指那些原本准备设计为PCI-X总线接口的设备，但是却被改为PCIe接口的设备。
    *   **Native PCIe Endpoint**：标准的PCIe设备。

**Lagacy PCIe Endpoint 和 Native PCIe Endpoint的区别**：

*   Lagacy PCIe Endpoint可以使用一些在 Native PCIe Endpoint 禁止使用的操作，如IO Space和Locked Request等。
*   Native PCIe Endpoint 则全部通过Memory Map来进行操作，因此，Native PCIe Endpoint 也被称为 Memory Mapped Devices（MMIO Devices）。

### NVMe 子系统

NVMe子系统直接通过PCIe总线和主机连接，路径中不再需要HBA（主机总线适配器）卡，降低了系统开销。
NVMe 子系统内部组成：

*   至少一个PCIe port，用于外部连接。
*   至少一个NVMe controller，该controller是实现了NVMe逻辑的PCI   function
*   名字空间标识(NSID)
*   名字空间(NS)：指一定量的NVM(Non-Volatile Memory)集合，这些NVM可被格式化为许多个逻辑块。一个NVMe控制器能支持多个不同命名空间ID(简称：NSID)标识的NS。
*   NAND Flash介质

![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/5e222796e0ca492bbc32e1376db19408~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=702\&h=555\&s=52114\&e=png\&b=f1f1d6)

注意：在系统向某个NS提交IO命令之前，**这个NS必须与某个控制器关联**。若NVM子系统支持NS管理，则NVM子系统内的NSID必须是唯一的(不管NS连接的是哪个控制器)；若不支持，则不要求私有NS的ID唯一。

### NVMe SSD 架构

NVMe SSD可以分为三部分，host端的驱动（NVMe官网以及linux、Windows已经集成了相应的驱动）、PCIe+NVMe实现的控制器以及FTL+NAND Flash的存储介质。

![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/6078fba77dfb4ebdbc7344a83aa00d13~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1080\&h=691\&s=482608\&e=png\&b=fefbfa)

### NVMe 工作流程

NVMe 工作流程大体为主机端（Host）通过创建 Admin 命令和 I/O 命令提交到队列和更新门铃寄存器（Doorbell Register），以建立与SSD之间的通信，实现指令的发送和完成信息处理过程。

NVMe协议所规范的标准工作流程如图所示。

![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/95910b9d4a774b7d9f592bf6784ba92d~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=978\&h=532\&s=186877\&e=png\&b=fcfbfb)

具体工作流程按以下过程进行：

1.  主机控制器将指令写入相应的提交队列（SQ）；
2.  主机更新SSD端门铃寄存器（DB）以通知SSD有新的待执行指令；
3.  SSD通过检查门铃寄存器获取指令数量，并将指令从对应的提交队列中读出；
4.  SSD控制器依次解析并执行指令；
5.  指令执行完成后，SSD控制器将提交指令对应的完成信息依次写入主机端完成队列（CQ）队尾；
6.  SSD发送中断信号，通知主机指令执行完成；
7.  主机控制器从队列队首开始依次检查完成队列中新的完成信息，并分析指令的执行情况；
8.  检查完成后，主机发送门铃信息至完成队列的门铃寄存器（DB），以通知SSD其返回的完成信息已检查完成。

### NVMe over Fabric

NVMe over Fabrics，简称 NVMe-oF，它是 NVM Express工作组在2016年发布的规范，通过网络将主机（如服务器）连接到存储。

早期的 NVMe over PCIe 局限在主机的本地盘使用。而通过 Fabrics（如RDMA或光纤通道）代替PCIe，可帮助主机访问节点外的NVMe SSD资源，NVMe-oF极大地增强了灵活性和扩展性，**将NVMe低延时、高并发等特性，从服务器级别，扩展到整个数据中心级别**。

具体如下图（NVMe Transports）所示，左边表示在PCIe上进行传输，右边为在Fabrics上的几种类型传输方式。

![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/95645cba06c740e1a71e89398ea58742~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1280\&h=581\&s=2235309\&e=png\&b=e8e8e8)

构建NVMe-oF的Fabrics网络，也即主机和外地盘的连接通道，有多种方式，如：**RDMA (NVMe /RDMA)，Fibre Channel (NVMe/FC) 、TCP (NVMe/TCP)** ，各自适用不同领域。

*   NVMe/RDMA：性能高、成本也较高，用于HPC、分布式数据库、AI机器学习等场景。
*   NVMe/FC：非常适合运行关键型任务工作负载的现有大型 FC 基础设施。
*   NVMe/TCP：在性能和成本之间取得了平衡，被定位为通用 NVMe-oF 的主力。

![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/19a3962bac40477cb59f609dd42db961~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1066\&h=478\&s=212045\&e=png\&b=fafafa)

**为什么出现NVMe Over Fabrics？**

1）更快的存储，需要更快的网络

从HDD到SSD，再发展到PM (Persistent Memory)，性能的改善(延时缩短)比最初提升到数千倍！

2）更快更大容量的存储，需要给多个主机共享

不过考虑到效率、弹性、可靠性、可用性、可运维性等需求，开始出现了JBOF、EBOF等新型架构。

通常一个闪存盘箱(JBOF或EBOF)的SSD个数在20以上，也即能提供的总IOPS高达1600万！

常规的主机（服务器）很难吸收或全部利用这么高的磁盘性能。合理的架构应该是一些主机构成的集群，通过Fabrics共享使用一个或多个全闪存盘箱。

当网络或说是传输通道延时超低，带宽也高时，主机可不配置本地盘。只留少数启动盘，甚至通过使用NVMe Server Boot Cards，连启动盘都不需要。例如，Marvell推出了RAID 1 Accelerator来提高NVMe启动盘的冗余保护。

这其实就是 Disaggregation of Compute and Storage（计算和存储的解耦），即大家近年来经常听到的存算分离。NVMe-oF的优势只有在计算和存储完全分开时才能完全发挥出来。也就是说，通过网络将一个NVMe SSD池提供给一个服务器池，这种方式允许按需提供计算和存储。计算和存储的分解提升了存储的可伸缩性和可共享性，并支持可组合性，如下图所示。

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/d4c4aec528bc48bf87a02991d6a62f3b~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=900\&h=376\&s=246631\&e=png\&b=faf6f5)

存储分离的另一个部分是存储服务（即数据保护、复制、压缩等）。存储服务可以由服务器管理（如：加载模型），也可以卸载到靠近实际存储的数据处理单元（DPU）。 加载模型会消耗额外的 CPU 周期和网络带宽，但可以最大限度地降低成本，而卸载模型会增加成本，并且根据配置的不同，可能会产生瓶颈，因此必须做出权衡。 由于 TCO（总成本）优势，对大规模低成本存储的追求引领了加载附加存储的策略。

**为什么需要网络呢？ 主机直接在本地盘使用不行吗？**

当然可以，确实有些要求延时低且冗余要求不高的场景，如：AI训练、NoSQL(数据冗余在应用软件层实现)等。

### EBOF、JBOF 与 JBOD 的区别？

JBOD（Just a Bunch of Disks）通常用于在PCIe上使用NVMe扩展机架中的存储。而EBOF或JBOF可以使用NVMe-oF在数据中心之间扩展存储。

全闪存阵列（bunch of flash）有两种方式接入到 NVMe-oF：通过网络接入(EBOF(Ethernet Bunch of Flash))和直接连接 (JBOF(just a bunch of flash))。JBOF使用PCIe交换机向SSD扩展，而EBOF使用以太网交换机向SSD扩展。JBOF和EBOF都使用NVMe-oF连接到服务器。

![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/d0966f32b4e94ba2adca3c443ddde8d8~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=900\&h=454\&s=107210\&e=png\&b=fefefe)

除了以太网和PCIe交换之外，这两种方法的主要区别在于从NVMe到NVMe-oF的转换发生在哪里。

*   在JBOF上，转换或桥接是在外围使用一个或多个DPU (x DPU到y SSD, x:y比率)。
*   在EBOF上，桥接在SSD载体完成(x桥接到x SSD, 1:1的比例)。

虽然JBOF有使用DPU的处理能力来运行存储服务的优势，但它确实存在一个潜在的瓶颈，并且和EBOF模型相比，带来了额外的成本，具有一些新功能。当桥接器与固态硬盘的比例不是1:1时，成本权衡和瓶颈问题就开始显现出来了。

### 如何选择 NVMe-over-Fabrics 方案？

NVMe 可以通过 FC、启用了 RDMA 的以太网或使用 TCP/IP 的标准以太网进行传输，下图展示了目前可用的主流 NVMe 光纤连接方案。那么不同方案间的主要差异是什么呢？

![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/f4d24028e25d4745b5a755b38790eccc~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=720\&h=745\&s=1612626\&e=png\&b=fbfafa)

**NVMe-over-FC (FC-NVMe)**：

对于已经部署了光纤通道存储网络(SAN)基础设施的用户而言，FC-NVMe 当属最优方案。使用 16Gb FC 或 32Gb FC 主机总线适配器（HBA）和SAN交换机，即可将 NVMe 协议封入 FC 框架内部。通过升级至最新的HBA固件和驱动程序则能获取 Linux 服务器上的 FC-NVMe 支持。因此，投资新型 16Gb 或 32Gb FC HBA和 SAN 基础设施，能够为应用今后推出的 FC-NVMe 存储阵列做好提前准备。另外值得注意的是，SCSI (FCP) 和 NVMe (FC-NVMe) 可以共存于相同的FC光纤网络中，因此，基于 FC-SCSI 的老存储可以与全新的 NVMe 存储同时运行。

**使用RDMA的NVMe-over-Ethernet光纤**：

RDMA 有两种不同的部署方式，名称分别为RoCE(v1/v2)和iWARP。 然而非常遗憾，以上两种协议无法实现交互操作。下面我将简要说明两种协议各自的优劣势：

**a. NVMe-over-RoCE (NVMe/RoCE)** ：

如果您使用的是只有以太网的网络，NVMe-over-RoCE 是共享存储或超融合基础设施 (HCI) 连接的最佳方案。正因如此，目前已有多家存储阵列供应商公布了他们的计划，及表示支持 NVMe-over-RoCE 连接。 RoCE 能够提供最低的以太网络延迟，并且对于跳数不超过 2 个的小规模存储网络，能达到非常优异的运行效果。顾名思义，RoCE 需要聚合或无损的以太网络才能正常运行。此外，该方案还需启用实现额外的网络功能，包括数据中心桥接(DCB)、优先流控制(PFC)，以及其他一些更复杂的组织架构和网络拥塞管理机制。如果低延迟是您的首要目标，那么 NVMe-over-RoCE 很可能是您的最优选择，尽管其网络复杂性也相对较高。

**b. NVMe-over-iWARP (NVMe/iWARP)** ：

iWARP RDMA 协议运行于标准 TCP/IP 网络之中，因此其部署操作也更加简单。 尽管该协议的延迟性能不及 RoCE，但更加易用的特性以及更低的管理难度依然具有巨大的吸引力。在现阶段，存储阵列供应商尚未设计出支持 iWARP 的阵列，因此目前的 iWARP 最适合软件定义或者基于 Microsoft Azure Stack HCI / Storage Spaces Direct (S2D) 等 HCI 解决方案。

**NVMe-over-TCP(NVMe/TCP)**

NVMe-over-TCP 还处于研发萌芽阶段。该方案于 2018 年 11 月获批，在不进行任何必要调整工作的情况下即可在现有的以太网基础设施中运行（这一点利用了 TCP/IP 广泛的普遍性）。 NVMe-over-TCP 发挥的性能表现可能在速度上不及 NVMe-over-RDMA 或 FC-NVMe，但在标准以太网卡和以太网网络交换机上就可以轻松实现部署。无需大量的硬件投资，即可享受 NVMe SSD 存储的主要优势。
Marvell FastLinQ 10/25/50/100GbE 等部分网卡还能利用网卡内置的TCP/IP协议栈的硬件卸载(offload)功能，发挥为 NVMe/TCP报文卸载并加速的潜力。

### NVMe 常用命令

以下是一些比较常用的命令，大部分命令需要在root用户下运行。

*   **nvme list**：查看所有连接到当前系统的nvme设备：名称，序列号，大小，LBA 和 serial
*   **nvme id-ctrl**：展示nvme控制器和它所支持的一些特性
*   **nvme id-ns**：展示 nvme 的命名空间，优化特性和支持特性
*   **nvme format**：安全擦除SSD上的数据，格式化LBA大小或保护信息以实现端到端数据保护
*   **nvme sanitize**：安全的擦除SSD上的所有数据
*   **nvme smart-log**：查看NVME的smart log信息：page的健康状态，温度，稳定性情况等
*   **nvme fw-log**：查看NVME的固件日志，会打印每个entry的健康情况
*   **nvme error-log**：NVME的异常日志
*   **nvme reset**：重置NVME的控制器
*   **nvme help**：查看帮助信息
*   **nvme delete-ns**：指定设备删除一个命名空间
*   **nvme create-ns**：指定设备创建命名空间。比如可以为一个设备创建一个较小大小的命名空间，从而提升SSD的稳定性，性能和延时
*   **nvme fw-download**：为一个设备下载一个新的固件系统
*   **nvme fw-commit**：让固件立即运行

## NVMe 在 AI 服务器中的应用

### GPUDirect 简介

GPUDirect 是 NVIDIA 开发的一项技术，可实现 GPU 与其他设备（例如网络接口卡 (NIC) 和存储设备）之间的直接通信和数据传输，而不涉及 CPU。

传统上，当数据需要在 GPU 和另一个设备之间传输时，数据必须通过 CPU，从而导致潜在的瓶颈并增加延迟。使用 GPUDirect，网络适配器和存储驱动器可以直接读写 GPU 内存，减少不必要的内存消耗，减少 CPU 开销并降低延迟，从而显著提高性能。GPU Direct 技术包括 GPUDirect Storage、GPUDirect RDMA、GPUDirect P2P 和 GPUDirect Video。

### GPUDirect Storage 简介

GPUDirect Storage （GDS）允许本地或远程存储(例如：NVMe 或 NVMe over Fabric (NVMe-oF))和 GPU 之间进行直接数据传输，绕过 CPU，**使 NIC 或存储附近的直接内存访问 (DMA) 引擎能够将数据通过直接路径移入或移出 GPU 内存**，从而减少数据传输的延迟和 CPU 开销。

通过 GPUDirect Storage，GPU 可以直接从存储设备（如：固态硬盘（SSD）或非易失性内存扩展（NVMe）驱动器）访问数据，而无需将数据先复制到 CPU 的内存中。这种直接访问能够实现更快的数据传输速度，并更高效地利用 GPU 资源。

GPUDirect Storage方案用到了两项高端技术，一个是RDMA，一个是NVMe（NVMe-oF），其中，RDMA被封装在GPUDirect的协议中，依靠各种网络适配器工作（比如Mellanox的NIC），既可以访问远程的存储也可以访问本地的存储设备。

下图展示了 GPU 内存和 NVMe 驱动器之间是否使用 GPUDirect Storage 的对比。从存储器到 GPU 的直接内存访问缓解了 CPU I/O 瓶颈，并提高了 I/O 带宽和容量。

![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/02e2b74589394624b353e6bcf42e7511~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=2397\&h=1202\&s=80887\&e=png\&b=ffffff)

GPUDirect Storage 的主要特点和优势包括：

*   **减少 CPU 参与**：通过绕过 CPU，实现 GPU 和存储设备之间的直接通信，GPUDirect Storage 减少了 CPU 开销，并释放 CPU 资源用于其他任务，从而改善系统的整体性能。
*   **低延迟数据访问**：GPUDirect Storage 消除了数据通过 CPU 的传输路径，从而最小化了数据传输的延迟。这对于实时分析、机器学习和高性能计算等对延迟敏感的应用非常有益。
*   **提高存储性能**：通过允许 GPU 直接访问存储设备，GPUDirect Storage 实现了高速数据传输，可以显著提高存储性能，加速数据密集型工作负载的处理速度。
*   **增强的可扩展性**：GPUDirect Storage 支持多 GPU 配置，允许多个 GPU 同时访问存储设备。这种可扩展性对于需要大规模并行处理和数据分析的应用至关重要。
*   **兼容性和生态系统支持**：GPUDirect Storage 设计用于与各种存储协议兼容，包括：NVMe、NVMe over Fabrics和网络附加存储（NAS）。它得到了主要存储供应商的支持，并集成到流行的软件框架（如：NVIDIA CUDA）中，以简化与现有的 GPU 加速应用程序的集成。

### GPUDirect Storage 工作原理

NVIDIA 力求尽可能采用现有标准，并在必要时扩展这些标准。 POSIX 标准的 pread 和 pwrite 提供存储和 CPU 缓冲区之间的复制，但尚未启用到 GPU 缓冲区的复制。 Linux 内核中不支持 GPU 缓冲区的缺点将随着时间的推移得到解决。一种名为 **dma\_buf** 的解决方案正在开发中，该解决方案可以在 NIC 或 NVMe 和 GPU 等 PCIe 总线上的对等设备之间进行复制，以解决这一问题。

与此同时，GDS 的性能提升空间太大，无法等待上游解决方案传递给所有用户。 许多供应商都提供了支持 GDS 的替代解决方案，如：MLNX\_OFED。

GDS 解决方案涉及新的 API：cuFileRead 或 cuFileWrite，它们与 POSIX pread 和 pwrite 类似。

像**动态路由、NVLink 的使用以及只能从 GDS 获得的用于 CUDA 流的异步 API** 等优化使得 cuFile API 成为 CUDA 编程模型的持久特性，即使在解决了 Linux 文件系统中的缺陷之后也是如此。

以下是 GDS 的实现：

首先，当前 Linux 实现的根本问题是将 GPU 缓冲区地址作为 DMA 目标向下通过虚拟文件系统 (VFS) 传递，以便本地 NVMe 或网络适配器中的 DMA 引擎可以执行与 GPU 内存之间的传输，这会导致错误情况。我们现在有一个解决这个问题的方法：传递 CPU 内存中缓冲区的地址。

当使用 cuFileRead 或 cuFileWrite 等 cuFile API 时，libcufile.so 用户级库**捕获 GPU 缓冲区地址**并**替换**传递给 VFS 的**代理 CPU 缓冲区地址**。 就在缓冲区地址用于 DMA 之前，启用 GDS 的驱动程序对 nvidia-fs.ko 的调用会**识别 CPU 缓冲区地址**并**再次提供替代** GPU 缓冲区地址，以便 DMA 可以正确进行。

libcufile.so 中的逻辑将执行前面描述的各种优化，例如：动态路由、prepinned缓冲区的使用、对齐等。

下图为 GDS 软件堆栈，其中：应用程序使用 cuFile API，并且支持 GDS 的存储驱动程序调用 nvidia-fs.ko 内核驱动程序来获取正确的 DMA 地址。

![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/c0671b6cf14243ccaee026619fbd24b3~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1227\&h=1064\&s=104304\&e=png\&b=fefefe)

### NVIDIA DGX 中集成 NVME SSD

NVIDIA DGX 系列是一系列专为深度学习工作负载而设计的高性能计算系统。下面为DGX 系列中搭载的 NVME SSD 详情。

| DGX 系列   | 存储 (OS)                          | 存储 (Data Cache)                   |
| -------- | -------------------------------- | --------------------------------- |
| DGX-1    | 1 x 480 GB, 6 Gb/s, SATA 3.0 SSD | 4 x 1.92 TB, 6 Gb/s, SATA 3.0 SSD |
| DGX-2    | 2 x 960GB NVMe SSDs              | 8 x 3.84 TB NVMe SSD              |
| DGX A100 | 2 x 1.92 TB NVMe M.2 SSD         | 4 x 3.84 TB NVMe U.2 SED          |
| DGX H100 | 2 x 1.92 TB NVMe M.2 SSD (ea)    | 8 x 3.84 TB NVMe U.2 SED          |

**DGX-2 中包含 NVMe SSD：**

DGX-2 机柜包含两个 CPU ，每个 CPU 都有两个 PCIe 子树实例，如图所示。从存储或系统内存到 GPUs 的多条 PCIe 路径由两级的 PCIe 交换机支持。

NVIDIA DGX-2 由 16 个 Tesla V100 组成，包含 30TB NVMe SSD 数据缓存存储（ 8x 3.84TB ）和 1.5TB 系统内存。启用驱动器的 DMA 操作允许快速访问内存，同时增加带宽、降低延迟。

跟原来不支持 GPUDirect Storage 的 DGX-2 系统相比，GPUDirect Storage 的吞吐带宽能提升8倍。

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/f0f78b0c75134fbe9313bb612341686c~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=2397\&h=1202\&s=86869\&e=png\&b=fffefe)

**DGX-A100 中包含 NVMe SSD：**

DGX A100 有 8个 A100 GPU，每个A100 GPU拥有12个NVLink端口，每个GPU拿出2个Link端口与一个NVSwitch连接，一共连接了6个NVSwitch。 同时，每个A100 GPU通过PCIE Switch连接了 200Gb/s 网卡（NIC）和 NVMe。

![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/47e765e2ee68482abbd3622287aa91f3~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1083\&h=720\&s=2343859\&e=png\&b=fefdfd)

**DGX-H100 中包含 NVMe SSD：**

HGX H100 拥有八个 H100 Tensor Core GPU 和 四个第三代 NV 交换机。每个 H100 GPU 都有多个第四代 NVLink 端口，并连接到所有四个 NVLink 交换机。每个 NVSwitch 都是一个完全无阻塞的交换机，完全连接所有八个 H100 Tensor Core GPU 。

NVSwitch 的这种完全连接的拓扑结构使任何 H100 都可以同时与任何其他 H100 通信。值得注意的是，这种通信以每秒 900 千兆字节（ GB/s ）的 NVLink 双向速度运行，这是当前 PCIe Gen4 x16 总线带宽的 14 倍多。

同时，每个 H100 GPU 通过 ConnectX-7 智能网卡连接  InfiniBand 及Ethernet（400Gb/s）。

![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/24412bf7ca844e029a9ff7007af6eb95~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1235\&h=720\&s=2672709\&e=png\&b=fcfafa)

## NVMe 在 DeepSpeed 中的应用

GPU 集群在存储器方面具有高度异构性。除了GPU内存外，还有 CPU 内存以及无限大（Infinity）的 NVMe 磁盘空间。ZeRO-Infinity 通过利用这些异构存储器，突破 GPU 内存壁垒。

讲述 ZeRO-Infinity 之前，先来看看 ZeRO 和 ZeRO-Offload 技术原理。

### ZeRO 技术原理

在 ZeRO 中，通过三个阶段依次对优化器状态（一阶动量、二阶动量）、梯度、参数的切割，解决了传统数据并行中冗余存储的问题，提高了 GPU 的内存使用效率。

![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/941d242f6b2546fb80e3017efc6cbb66~tplv-k3u1fbpfcp-watermark.image#?w=1280\&h=617\&s=3164841\&e=png\&b=fefefe)

**ZeRO-1**：

ZeRO-1没有将模型本身进行分片，也没有将Gradient进行分片，而是只将优化器进行分片。训练过程与DDP类似。

1.  forward过程由每个rank的GPU独自完整的完成，然后进行backward过程。在backward过程中，梯度通过allReduce进行同步。
2.  Optimizer state 使用贪心策略基于参数量进行分片，以此确保每个rank几乎拥有相同大小的优化器内存。
3.  每个rank只负责更新当前优化器分片的部分，由于每个rank只有分片的优化器state，所以当前rank忽略其余的state。
4.  在更新过后，通过广播或者allGather的方式确保所有的rank都收到最新更新过后的模型参数。

ZeRO-1 非常适合使用类似Adam进行优化的模型训练，因为Adam拥有额外的参数m（momentum）与v（variance），特别是FP16混合精度训练。ZeRO-1 不适合使用SGD类似的优化器进行模型训练，因为SGD只有较少的参数内存，并且由于需要更新模型参数，导致额外的通讯成本。ZeRO-1只是解决了Optimizer state的冗余。

**ZeRO-2**：

相比于ZeRO-1，ZeRO-2除了对optimizer state进行切分，还对Gradient进行了切分。

像ZeRO-1一样将optimizer的参数进行分片，并安排在不同的rank上。在backward过程中，**gradients被reduce操作到对应的rank上，取代了all-reduce**，以此减少了通讯开销。 每个rank独自更新各自负责的参数。在更新操作之后，广播或allGather保证所有的ranks接收到更新后的参数。

**ZeRO-3**：

为了进一步节省更多的内存，ZeRO-3提出进行模型参数的分片。类似以上两种分片方式，ranks负责模型参数的切片。可以进行参数切片的原因主要有以下两点：

1.  All-Reduce操作可以被拆分为Reduce与allgather操作的结合。
2.  模型的每一层拥有该层的完整参数，并且整个层能够直接被一个GPU装下。所以计算前向的时候，除了当前rank需要的层之外，其余的层的参数可以抛弃。从这个层面上来说，Zero相当于数据并行+模型并行。

更具体说明查看之前的文章：[大模型分布式训练并行技术（二）-数据并行](https://zhuanlan.zhihu.com/p/650002268)

### ZeRO-Offload 技术原理

ZeRO-Offload 的动机是解决 GPU 内存墙的问题。具体而言，传统分布式训练中的 3D 并行（数据、张量、流水线）解决了模型无法存放在单张 GPU 内存中的问题，但是因为模型的参数、梯度、优化器状态总要存放在 GPU 内存中，需要要求所有的 GPU 的内存总和大于上述模型状态所需内存。但是模型规模的增长速度要远远大于 GPU 内存大小的增长速度，二者之间越来越大的差距使得上述要求已经很难得到满足。为此，ZeRO-Offload 利用异构设备训练（Heterogeneous DL training ）的思想，即利用 CPU 内存来减少 GPU 内存的压力，并集成到了 ZeRO-2 中。

ZeRO-Offload 分为 Offload Strategy 和 Offload Schedule 两部分，前者解决如何在 GPU 和 CPU 间划分模型的问题，后者解决如何调度计算和通信的问题。

在 Offload Strategy 方面，方法首先将训练过程抽象为数据流图，图中点分为两类，一类表示需要存储的模型状态，一类表示运算操作（前向传播、反向传播、参数更新等）；图中边表示节点之间传递的数据量。在数据流图的基础上，该方法提出了关于最优策略的 3 点见解：

1.  保证 CPU 的计算负担远远小于 GPU，从而防止 CPU 成为计算瓶颈；
2.  保证 CPU 和 GPU 之间的通信量最小；
3.  保证 GPU 的内存节省最大；

![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/a59e37656f3e4ae29345c46d3ba73cb2~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1494\&h=938\&s=278217\&e=png\&b=f2f2f2)

CPU 负责存储 FP32 的优化器状态和 FP16 的梯度，同时更新参数；GPU 负责存储 FP16 的梯度和参数，同时完成前向传播和反向传播。该方法声称这种策略在不增加 CPU 计算量和不增加 CPU 和 GPU 之间通信开销的前提下是最优的。

在 Offload Schedule 方面，分为单张 GPU 和多张 GPU 两部分。

单张 GPU 的调度如下图所示，核心是将 GPU 到 CPU 的梯度 offload 和 GPU 上反向传播时的梯度计算重叠，以及将 CPU 到 GPU 的参数 swap和 CPU 上的参数更新重叠。

![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/a859b37ad6bd468b80a9f0992d508d96~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1066\&h=362\&s=88128\&e=png\&b=fefbfb)

多张 GPU 的调度如下图所示，GPU 和 CPU 之间的通信量和 CPU 上的计算量没有改变，单纯和 ZeRO-2 的并行方式相结合。

![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/c842f89e854148f6b8666c7caf2a3127~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=946\&h=436\&s=85398\&e=png\&b=fefdfd)

### ZeRO-Infinity 技术原理

ZeRO-Infinity 则在 ZeRO-Offload 的基础上进一步优化，主要包括三个方面。

*   一是将和 ZeRO 的结合从 ZeRO-2 延伸到了 ZeRO-3，**解决了模型参数受限于单张 GPU 内存的问题**；
*   二是解决了 ZeRO-Offload 在训练 batch size 较小的时候效率较低的问题；
*   三是除 CPU 内存外，尝试**利用 NVMe 的空间进一步打破内存墙的限制训练超大模型**。

![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/034a508f0ec2470b93cd2896aa3ee864~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1120\&h=1064\&s=353789\&e=png\&b=faf4f3)

### ZeRO-Infinity 应用示例

在 DeepSpeed AI框架中，进行模型训练时，可通过配置
deepspeed\_config文件通过 ZeRO-Infinity 技术来使用 NVMe。

比如：将模型参数卸载到 CPU/NVMe。仅适用于 ZeRO Stage 3。

```json
{
  "offload_param": {
    "device": "[cpu|nvme]",
    "nvme_path": "/local_nvme",
    "pin_memory": [true|false],
    "buffer_count": 5,
    "buffer_size": 1e8,
    "max_in_cpu": 1e9
  }
}
```

参数说明：

*   **device**：用于卸载模型参数的设备内存。 支持的选项有 cpu 和 nvme。
*   **pin\_memory**：转移到CPU锁页（page-locked）内存。这可能会提升吞吐量，但代价是增加了额外的内存开销。(默认值：false)
*   **nvme\_path**：用于参数卸载的 NVMe 设备的文件系统路径。
*   **buffer\_count**：用于参数卸载到 NVMe 的缓冲池中的缓冲区数量。
*   **buffer\_size**：用于参数卸载到 NVMe 的缓冲池中的缓冲区大小。
*   **max\_in\_cpu**：启用卸载到 NVMe 时要在 CPU 内存中维护的参数元素数量。

同时，还支持将优化器计算卸载到 CPU 并将优化器状态卸载到 CPU/NVMe。 卸载到 CPU 适用于 ZeRO Stage 1、2、3。卸载到 NVMe 仅适用于 ZeRO Stage 3

```json
{
  "offload_optimizer": {
    "device": "[cpu|nvme]",
    "nvme_path": "/local_nvme",
    "pin_memory": [true|false],
    "ratio": 0.3,
    "buffer_count": 4,
    "fast_init": false
  }
}
```

参数说明：

*   **device**：用于卸载优化器状态的设备内存。 支持的选项有 cpu 和 nvme。 无论设备选项如何，优化器计算都会卸载到 CPU。
*   **nvme\_path**：用于优化器状态卸载的 NVMe 设备的文件系统路径。
*   **pin\_memory**：卸载到 CPU 锁页内存。这可以提高吞吐量，但代价是额外的内存开销。
*   **ratio**：CPU 侧参数更新（即优化器步骤）的比率。
*   **buffer\_count**：用于优化器状态卸载到 NVMe 的缓冲池中的缓冲区数量。 这至少应该是优化器为每个参数维护的状态数。 例如，Adam 优化器有 4 个状态（参数、梯度、动量和方差）。
*   **fast\_init**：卸载到 NVMe 时启用快速优化器初始化。

总的来说就是在使用 ZeRO Stage 1、2 时，可以使用 offload\_optimizer，优化器状态卸载到CPU。在使用 ZeRO Stage 3 时，可以同时使用offload\_optimizer 和 offload\_param 将优化器状态和模型参数卸载到CPU或NVMe。

## 总结

本文讲述了首先提到了衡量传输速度的三大要素（传输通道、通信协议、硬件接口），然后硬盘的发展进行了基本的介绍、之后对NVMe的技术原理以及NVMe在AI服务器及DeepSpeed框架中的应用进行了详细的讲述。

码字不易，如果觉得有帮助，欢迎点赞收藏加关注。

## 参考文档

*   [计算机固态硬盘（SSD）和机械硬盘（HDD）的区别](https://consumer.huawei.com/cn/support/content/zh-cn15819906/)
*   [了解 SSD 技术：NVMe、SATA、M.2](https://www.kingston.com/cn/ssd/what-is-nvme-ssd-technology)
*   [固态硬盘接口类型及区别](https://www.i-tc.com.cn/product-service/736.html)
*   [总线、协议、接口](https://zhuanlan.zhihu.com/p/561858380)
*   [M.2 固态硬盘的两种类型：SATA 和 NVMe](https://www.kingston.com/cn/blog/pc-performance/two-types-m2-vs-ssd)
*   [硬盘科普，M.2，PCI-E，NVMe 傻傻分不清](https://zhuanlan.zhihu.com/p/396745362)
*   [NVMe、AHCI、PCIe、SATA、NGFF接口、协议小结](https://blog.csdn.net/wujinglin7/article/details/122826608)
*   [M.2 固态硬盘的两种类型：SATA 和 NVMe](https://www.kingston.com/cn/blog/pc-performance/two-types-m2-vs-ssd)
*   [M.2 接口详解](https://zhuanlan.zhihu.com/p/491944861)
*   [「科普」NVMe2.0：闪存存储新篇章](https://baijiahao.baidu.com/s?id=1782475465795992948\&wfr=spider\&for=pc)
*   [什么是NVMe存储？了解新的行业标准](https://www.wbolt.com/what-is-nvme.html)
*   [NVMe技术分析之工作原理](https://blog.csdn.net/tiantianuser/article/details/117802938)
*   [固态硬盘常提到的NVMe协议是个啥？](https://www.zhihu.com/tardis/zm/art/137903162?source_id=1003)
*   [NVME协议解读（一）](https://blog.csdn.net/jingjiankai5228/article/details/121865937)
*   [理解NVMe的内部实现原理，这一篇就够了](https://zhuanlan.zhihu.com/p/71932654)
*   [收藏：NVMe协议基础原理介绍](https://blog.csdn.net/weixin_38889300/article/details/128349761)
*   [NVMe技术架构深度分析](https://blog.csdn.net/BtB5e6Nsu1g511Eg5XEg/article/details/84332456)
*   [深入剖析NVMe Over Fabrics](https://www.qinglite.cn/doc/300664763aa0aa7f1)
*   [如何选择最优的NVMe-over-Fabrics方案？](https://mp.weixin.qq.com/s/yNFcUoqcNVBGekxrrkoMOg)
*   [再见，Intel！GPU直连NVMe SSD！](https://www.sohu.com/a/442463342_505795)
*   [Ethernet Bunch of Flash in an NVMe-oF™ Network for Low-Cost Storage at Scale ](https://www.micron.com/about/blog/2020/november/ethernet-bunch-of-flash-in-an-nvme-of-network-for-low-cost-storage-at-scale)
*   [聊透 GPU 通信技术——GPU Direct、NVLink、RDMA](https://zhuanlan.zhihu.com/p/654417967)
*   [NVIDIA GPUDirect](https://developer.nvidia.com/gpudirect)
*   [GPUDirect Storage: A Direct Path Between Storage and GPU Memory](https://developer.nvidia.com/blog/gpudirect-storage/)
*   [Accelerating IO in the Modern Data Center: Magnum IO Storage](https://developer.nvidia.com/blog/accelerating-io-in-the-modern-data-center-magnum-io-storage/)
*   [NVIDIA Hopper 深入研究架构](https://developer.nvidia.com/zh-cn/blog/nvidia-hopper-architecture-in-depth/)
*   [介绍 NVIDIA HGX H100 ：用于人工智能和高性能计算的加速服务器平台](https://developer.nvidia.com/zh-cn/blog/introducing-nvidia-hgx-h100-an-accelerated-server-platform-for-ai-and-high-performance-computing/)
*   [DeepSpeed config-json](https://www.deepspeed.ai/docs/config-json/)
*   [关于Deepspeed的一些总结与心得](https://zhuanlan.zhihu.com/p/650824387)
*   [ZeRO-Offload 和 ZeRO-Infinity 流程解读](https://zhuanlan.zhihu.com/p/657946200)


