
GPU在高性能计算和深度学习加速中扮演着非常重要的角色， GPU的强大的并行计算能力，大大提升了运算性能。随着运算数据量的不断攀升，GPU间需要大量的交换数据，GPU通信性能成为了非常重要的指标。

NVIDIA推出的GPUDirect就是一组提升GPU通信性能的技术。


在分布式训练中，通信是必要环节，同时也是相比于单机训练而言多出来的系统开销。通信与计算的时间比例往往决定了分布式机器学习系统加速比的上限。

因此，分布式机器学习的关键是设计通信机制，从而降低通信与计算的时间比例，更加高效地训练出高精度模型。下面给大家一一介绍通信的软硬件、通信的实现方式、通信的步调、通信的频率等。


## 通信硬件

### 通信硬件的实现方式 

通信的实现方式分为两种类型：机器内通信与机器间通信。

机器内通信：

- 共享内存，比如：CPU与CPU之间的通信可以通过共享内存。
- PCIe，通常是CPU与GPU之间的通信。
- NVLink，通常是GPU与GPU之间的通信，也可以用于CPU与GPU之间的通信。

机器间通信：

- TCP/IP 网络协议。
- RDMA (Remote Direct Memory Access) 网络协议。
    -  InfiniBand
    -  iWARP
    -  RoCE

### PCIe

PCI-Express（peripheral component interconnect express），简称PCIe，是一种高速串行计算机扩展总线标准，主要用于扩充计算机系统总线数据吞吐量以及提高设备通信速度。

PCIE本质上是一种全双工的的连接总线，传输数据量的大小由通道数（lane，信道）决定的。

通常，1个连接通道lane称为X1，**每个通道lane由两对数据线组成，一对发送，一对接收，每对数据线包含两根差分线。即X1只有1个lane，4根数据线**，每个时钟每个方向1bit数据传输。依此类推，X2就有2个lane，由8根数据线组成，每个时钟传输2bit。类似的还有X12、X16、X32。


![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/d629223a46564875a2c1160e9521728d~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=720&h=1131&s=714141&e=png&b=4a5a66)


2003 年 PCIe 1.0 正式发布，可支持每通道传输速率为 250MB/s，总传输速率为 2.5 GT/s。

2007 年推出PCIe 2.0 规范。在 PCIe 1.0 的基础上将总传输速率提高了一倍，达到 5 GT/s，每通道传输速率从 250 MB/s 上升至 500 MB/s。

2022 年 PCIe 6.0 规范正式发布，总传输速率提高至 64 GT/s。

2022年6月，PCI-SIG联盟宣布PCIe 7.0版规范，单条通道（x1）单向可实现128GT/s传输速率，计划于2025年推出最终版本。

![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/f6a4a3004b3c4ceaa9567b2e7ef90eb0~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1406&h=431&s=102106&e=png&b=eeeeee)


PCIE体系架构一般包含根组件RC（rootcomplex），交换器switch，终端设备EP（endpoint）等类型的PCIE设备组成。RC在总线架构中只有一个，用于处理器和内存子系统与I/O设备之间的连接，而switch的功能通常是以软件形式提供的，它包括两个或更多的逻辑PCI到PCI的连接桥（PCI-PCI Bridge），以保持与现有PCI兼容。

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/2da8b7c4b2b044dfab80bb3c673043ed~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=852&h=600&s=203332&e=png&b=fefdfd)


### NVLink

算力的提升不仅依靠单张GPU卡的性能提升，往往还需要多GPU卡组合。在多GPU系统内部，GPU间通信的带宽通常在数百GB/s以上，PCIe总线的数据传输速率容易成为瓶颈，且PCIe链路接口的串并转换会产生较大延时，影响GPU并行计算的效率和性能。

GPU发出的信号需要先传递到PCIe Switch, PCIe Switch中涉及到数据的处理，CPU会对数据进行分发调度，这些都会引入额外的网络延迟，限制了系统性能。

![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/e5298c9a3382455cad518360bc16a74f~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=700&h=356&s=89728&e=png&b=f8efe8)


为此，NVIDIA推出了能够提升GPU通信性能的技术——GPUDirect P2P技术，使GPU可以通过PCI Express直接访问目标GPU的显存，避免了通过拷贝到CPU host memory作为中转，大大降低了数据交换的延迟，但受限于PCI Express总线协议以及拓扑结构的一些限制，无法做到更高的带宽。此后，NVIDIA提出了NVLink总线协议。


NVLink 是一种高速互连技术，旨在加快 CPU 与 GPU、GPU 与 GPU 之间的数据传输速度，提高系统性能。NVLink通过GPU之间的直接互联，可扩展服务器内的多GPU I/O，相较于传统PCIe总线可提供更高效、低延迟的互联解决方案。


![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/b46bcbcc54384297ab173b306a6ae58c~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=625&h=352&s=179166&e=png&b=353535)


NVLink的首个版本于2014年发布，首次引入了高速GPU互连。2016年发布的P100搭载了第一代NVLink，提供 160GB/s 的带宽，相当于当时 PCIe 3.0 x16 带宽的 5 倍。之后陆续发布了很多新版本，V100搭载的NVLink2将带宽提升到300GB/s ，A100搭载了NVLink3带宽为600GB/s。H100中包含18条第四代NVLink链路，总带宽达到900 GB/s，是PCIe 5.0带宽的7倍。


![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/6ac72f40a0374e7aa305034166b618fa~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=747&h=489&s=24935&e=png&b=eeeeee)

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/a24232222ccb49598a816bd923351aac~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1710&h=273&s=71583&e=png&b=ededed)

NVLink高速互联主要有两种：
- 第一种是以桥接器的形式实现。
- 另一种是在主板上集成 `NVLink` 接口。

![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/d07e4b62d39848b0a7aa9e37593ce6c4~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1033&h=381&s=382215&e=png&b=f9f6f6)

### NVSwitch

为了解决GPU之间通讯不均衡问题，NVIDIA引入NVSwitch。NVSwitch芯片是一种类似交换机的物理芯片（ASIC），通过NVLink接口可以将多个GPU高速互联到一起，可创建无缝、高带宽的多节点GPU集群，实现所有GPU在一个具有全带宽连接的集群中协同工作，从而提升服务器内部多个GPU之间的通讯效率和带宽。NVLink和NVSwitch的结合使NVIDIA得以高效地将AI性能扩展到多个GPU。


![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/936a389603f84830a4092394c7418358~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=688&h=401&s=48837&e=png&b=edeaea)


第一代 NVSwitch于2018年发布，采用台积电 12nm FinFET 工艺制造，共有 18 个 NVLink 2.0 接口。目前 NVSwitch 已经迭代至第三代。第三代 NVSwitch 采用台积电 4N 工艺（台积电 4N 工艺专为NVIDIA定制设计，并进行了一系列优化，但并非4nm，而是5nm，它与普通台积电5nm节点相比，可实现更好的电源效率与性能，并且密度有所提升）构建，每个 NVSwitch 芯片上拥有 64 个 NVLink 4.0 端口，GPU 间通信速率可达 900GB/s。


![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/b31291309f6d40ae92b525e78a6fcd70~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1546&h=207&s=56495&e=png&b=f3f3f3)


### Nvidia GPU 服务器 PCIe 版 和 SXM 版的区别

英伟达GPU卡间互连的内存插槽有2种，一种是PCIe口，一种是SXM口。

PCIe口是一个相对通用的协议，PCIe口相对慢一些，SXM是专门用来做卡间互连的，SXM协议是铺在电路板上，SXM协议做卡间互连会更快，对NVLink原生支持更好，显存带宽比PCIe高一些。PCIe和SXM都可以用NVLink，但是SXM是更好使用NVLink的方法。  

SXM架构是一种高带宽插座式解决方案，用于将 GPU连接到NVIDIA 专有的 DGX 和 HGX 系统。SXM 版 GPU 通过主板上集成的NVSwitch实现NVLink的连接，不需要通过主板上的PCIe进行通信，它能支持8块GPU卡的互联互通，实现了GPU之间的高带宽。未阉割的A100是600GB/s、H100是900GB/s，阉割过的A800、H800为400GB/s。


把 PCIe 版 GPU卡插到PCIe插槽上，就可以和CPU、同一个服务器上其他的GPU卡进行通信，也可以通过网卡与其他的服务器节点上的设备进行通信，这种就是PCIe的通信方式，但是这种传输速度不快。如果想要和SXM一样，有很快的传输速度，可以使用NVlink桥接器实现GPU和CPU之间的通信，但是和SXM不一样的地方就是它只能实现2块GPU卡之间的通信。即 PCIe 版只有成对的 GPU 通过 NVLink Bridge 连接，通过 PCIe 通道进行数据通信。同时，最新的PCIe网络带宽有128GB/s的限制。

![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/3bc4f115338245a88091182bddaaa1d1~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1982&h=677&s=821728&e=png&b=09428d)


### TCP/IP

TCP/TP 或传输控制协议/Internet 协议用于通过 Internet 互连网络设备。它确定了数据应该如何被打包、寻址、传输、路由和接收。TCP/IP 非常重视两台计算机之间的准确数据传输。如果系统在一次发送消息时遇到问题，则必须重新发送整个消息。

此外，TCP/IP 的功能分为四个不同的层：**数据链路层、互联网层、传输层和应用层**。数据在被另一端接收之前必须经过这四层。然后，TCP/IP 将通过以相反顺序传递层来重组数据并将其呈现给接收器。这样，您可以通过升级某些层而不是整个系统来提高数据中心的性能或安全性。

### RDMA

RDMA(远程直接数据存取)就是为了解决网络传输中服务器端数据处理的延迟而产生的，**无需使用CPU，就可以从一个主机或服务器的内存直接访问另一主机或服务器的内存**。它释放了CPU去执行其应做的工作，比如：运行应用程序和处理大量数据。这既提高了带宽又降低了延迟、抖动和 CPU 消耗。

对比传统的网络传输机制，RDMA无需操作系统和TCP/IP协议栈的介入。**RDMA的内核旁路机制，允许应用与网卡之间的直接数据读写**，将服务器内的数据传输时延降低到1us以下。同时，RDMA的内存零拷贝机制，允许接收端直接从发送端的内存读取数据，极大的减少了CPU的负担，提升CPU的效率。


![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/1e4c8404a9124f48ad6ee0e914489ca8~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1080&h=443&s=125224&e=png&b=ffffff)

大致有三类RDMA网络，分别是Infiniband、RoCE、iWARP。其中，Infiniband是一种专为RDMA设计的网络，从硬件级别保证可靠传输 ，而RoCE 和 iWARP都是基于以太网的RDMA技术，支持相应的verbs接口。



## InfiniBand

InfiniBand（直译为 “无限带宽” 技术，缩写为IB）是一个为大规模、易扩展机群而设计的**网络通信技术协议**。可用于计算机内部或外部的数据互连，服务器与存储系统之间直接或交换互连，以及存储系统之间的互连。

InfiniBand最重要的一个特点就是**高带宽**、**低延迟**，因此在高性能计算项目中广泛的应用。 主要用于高性能计算（HPC）、高性能集群应用服务器和高性能存储。


### InfiniBand 网络带宽的演进

下图展示了 InfiniBand 网络带宽从SDR、DDR、QDR、FDR、EDR、HDR到NDR的发展，其速度是基于 4x 链路速度。

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/52c2a6d6c45f4ac4be1b19c2032bd380~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1080&h=248&s=81845&e=png&b=fefefe)

- SDR（Single Data Rate）：2.5Gb/s (10Gb/s for 4x)。
- DDR（Double Data Rate）：5 Gb/s (20Gb/s for 4x)。
- QDR（Quad Data Rate）：10 Gb/s (40Gb/s for 4x)。
- FDR（Fourteen Data Rate）：14Gb/s (56Gb/s for 4x)。
- EDR（Enhanced Data Rate）：25 Gb/s (100Gb/s for 4x)。
- HDR（High Data Rate）：50 Gb/s (200Gb/s for 4x)。
- NDR（Next Data Rate）：100 Gb/s (400Gb/s for 4x)。


### InfiniBand 网络互连产品

InfiniBand 网络中，使用的线缆区别于传统的以太网线缆和光纤线缆。针对不同的连接场景，需使用专用的InfiniBand线缆。

InfiniBand网络互连产品包括：DAC高速铜缆、AOC有源线缆以及光模块。



### InfiniBand 的技术原理



### InfiniBand 的网络架构



### InfiniBand 的协议栈




### InfiniBand的商用产品


英伟达收购 Mellanox 之后，于2021年推出了自己的第七代 NVIDIA InfiniBand 架构：NVIDIA Quantum-2。

NVIDIA Quantum-2 平台包括：NVIDIA Quantum-2 系列交换机、NVIDIA ConnectX-7 InfiniBand 适配器、BlueField-3 InfiniBand DPU以及电缆。

![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/a93953336c7a426880e2f97a42898a29~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1499&h=696&s=662947&e=png&b=faf9f9)

NVIDIA Quantum-2 系列交换机采用紧凑型1U设计，包括风冷和液冷版本。交换机的芯片制程工艺为7nm，单芯片拥有570亿个晶体管（比A100 GPU还多）。单个交换机采用64个400Gb/s端口或128个200Gb/s端口的灵活搭配，提供总计 51.2Tb/s的双向吞吐量。NVIDIA NDR 400Gb/s InfiniBand 交换机如下图所示：

![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/6cbb986d98c04b42a5f1d8789b53a9ef~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=630&h=354&s=113706&e=png&b=1b1b1b)

NVIDIA ConnectX-7 InfiniBand 适配器，支持PCIe Gen4和Gen5，具有多种外形规格，可提供 400Gb/s 吞吐量。


![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/7e2025e0d5244d199fd554174dfe12f6~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=391&h=265&s=58835&e=png&b=fefefe)



## 通信软件

通信软件指用于分布式训练时，多个计算设备之间的集合通信。在分布式系统中，各个节点间往往存在大量的集合通信需求，而我们可以用消息传递接口 (Message Passing Interface，MPI，一套集合通信相关的接口标准) 来定义一些比较底层的消息通信行为。譬如 Reduce、AllReduce、Scatter、Gather、AllGather 等。

常见的集合通信库（如：Open MPI、Gloo、NCCL等）都在 MPI 的基础上，对各种集合通信的模式和算法作了各自的实现。


**Open MPI**：

Open MPI 是一个开源 MPI（消息传递接口 ）的实现，由学术，研究和行业合作伙伴联盟开发和维护。因此，Open MPI 可以整合高性能计算社区中所有专家，技术和资源，以构建可用的最佳 MPI 库。


**Gloo**：

Gloo 是 Facebook 开源的一套集体通信库，提供了对机器学习中有用的一些集合通信算法。如：Barrier，Broadcast，AllReduce。

**NCCL**：

NCCL 是英伟达基于 NVIDIA GPU 的一套开源的集合通信库，如其官网描述：NVIDIA 集合通信库（NCCL）实现了针对 NVIDIA GPU 性能优化的多 GPU 和多节点集合通信原语。NCCL 提供了诸如 All Gather，All Reduce，Broadcast，Reduce，Reduce-Scatter 等实现，这些实现优化后可以通过 PCIe 和 NVLink 等高速互联，从而实现高带宽和低延迟。

因为 NCCL 是 NVIDIA 基于自身硬件定制的，能做到更有针对性且更方便优化，故在英伟达硬件上，NCCL 的效果往往比其它的通信库更好。

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/eb1018950a05412b93e30ab79b808039~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=537&h=276&s=72386&e=png&b=fdfafa)



## InfiniBand 在 AI 集群中的应用


### GPUDirect 简介

GPUDirect 是 NVIDIA 开发的一项技术，可实现 GPU 与其他设备（例如网络接口卡 (NIC) 和存储设备）之间的直接通信和数据传输，而不涉及 CPU。

传统上，当数据需要在 GPU 和另一个设备之间传输时，数据必须通过 CPU，从而导致潜在的瓶颈并增加延迟。使用 GPUDirect，网络适配器和存储驱动器可以直接读写 GPU 内存，减少不必要的内存消耗，减少 CPU 开销并降低延迟，从而显著提高性能。GPU Direct 技术包括 GPUDirect Storage、GPUDirect RDMA、GPUDirect P2P 和 GPUDirect Video。


### GPUDirect Peer to Peer（P2P）简介

GPUDirect Peer-to-Peer(P2P) 技术主要用于单机GPU间的高速通信，它使得GPU可以通过PCI Express直接访问目标GPU的显存，避免了通过拷贝到CPU host memory作为中转，大大降低了数据交换的延迟。

以深度学习应用为例，主流的开源深度学习框架（如：TensorFlow、MXNet）都提供了对GPUDirect P2P的支持，NVIDIA开发的NCCL(NVIDIA Collective Communications Library)也提供了针对GPUDirect P2P的特别优化。

通过使用GPUDirect P2P技术可以大大提升深度学习应用单机多卡的扩展性，使得深度学习框架可以获得接近线性的训练性能加速比。

### GPUDirect RDMA 简介

所谓 GPUDirect RDMA，就是计算机1的GPU可以直接访问计算机2的GPU内存。而在没有这项技术之前，GPU需要先将数据从GPU内存搬移到系统内存，然后再利用RDMA传输到计算机2，计算机2的GPU还要做一次数据从系统内存到GPU内存的搬移动作。GPUDirect RDMA技术使得进一步减少了GPU通信的数据复制次数，通信延迟进一步降低。

![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/02ea5dd1ecf746cbaa6628ae78bf8db6~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=3840&h=2160&s=192623&e=png&a=1&b=2083bd)

使用 GPUDirect RDMA 两个GPU设备必须共享相同的上游 PCI Express root complex。

![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/f4c74697fc854927bc45a2dbc23aae6c~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=747&h=506&s=15351&e=png&b=ffffff)


### NVIDIA DGX 中集成 InfiniBand




## InfiniBand 在 AI 框架中的应用


## PyTorch


## DeepSpeed





## 总结




## 参考文档

- [带你了解PCIE通信原理](https://zhuanlan.zhihu.com/p/454282470)
- [电脑硬件冷知识：主板北桥芯片为何消失了，南桥也有同样的命运？](https://zhuanlan.zhihu.com/p/662904805)
- [必看: 原来PCIe技术原理这么简单](https://mp.weixin.qq.com/s/FlRc2q8r0fUOzxJFWulGfw)
- [AI网络互联，PCIe还是NVLink？](https://www.sdnlab.com/26316.html)
- [详谈RDMA技术原理和三种实现方式](https://mp.weixin.qq.com/s/FgKjDjZsPlweVJ03OVr3SA)
- [NVIDIA MLNX_OFED Documentation v5.8-3.0.7.0.101 for DGX H100 Systems](https://docs.nvidia.com/networking/display/mlnxofedv583070101/introduction)
- [态路小课堂丨关于InfiniBand网络相关内容简介！](https://baijiahao.baidu.com/s?id=1760941961023057651&wfr=spider&for=pc)
- [NCCL 环境变量](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [NVIDIA ConnectX InfiniBand 网卡](https://www.nvidia.cn/networking/infiniband-adapters/)
- [InfiniBand，到底是个啥？](https://mp.weixin.qq.com/s?__biz=MzI1NTA0MDUyMA==&mid=2456692454&idx=1&sn=031a11b931edee5504b15045cd863d37&chksm=fda68b81cad10297e4dd53bc97f63e0c47c26a27cdbb3c584cce6fc49fc6b4367b1531cbfcb6&scene=0&xtrack=1#rd)
- [浅析GPU通信技术（上）-GPUDirect P2P](https://developer.aliyun.com/article/591403)
- [浅析GPU通信技术（中）-NVLink](https://developer.aliyun.com/article/599183)
- [浅析GPU通信技术（下）-GPUDirect RDMA](https://developer.aliyun.com/article/603617)
- [GPUDirect RDMA 12.3 文档](https://docs.nvidia.com/cuda/gpudirect-rdma/index.html)





