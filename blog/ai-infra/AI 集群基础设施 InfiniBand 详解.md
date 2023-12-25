
GPU在高性能计算和深度学习加速中扮演着非常重要的角色， GPU的强大的并行计算能力，大大提升了运算性能。随着运算数据量的不断攀升，GPU间需要大量的交换数据，因此，GPU通信性能成为了非常重要的指标。

在 AI 集群中进行分布式训练时，通信是必要环节，同时也是相比于单机训练而言多出来的系统开销。通信与计算的时间比例往往决定了分布式机器学习系统加速比的上限。

因此，分布式机器学习的关键是设计通信机制，从而降低通信与计算的时间比例，更加高效地训练出高精度模型。

下面给大家介绍AI集群通信的软硬件、NCLL集合通信库、InfiniBand网络通信技术协议、AI集群和AI框架中对于InfiniBand的应用。

> 文章较长，建议先点赞收藏，后续再慢慢观看。另外，我撰写的**大模型相关的博客及配套代码**均整理放置在Github：[llm-action](https://github.com/liguodongiot/llm-action/tree/main)，有需要的朋友自取。

## 通信硬件

### 通信硬件的实现方式 

通信的实现方式分为两种类型：机器内通信与机器间通信。

机器内通信：

- 共享内存（QPI/UPI），比如：CPU与CPU之间的通信可以通过共享内存。
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

2007 年推出 PCIe 2.0 规范。在 PCIe 1.0 的基础上将总传输速率提高了一倍，达到 5 GT/s，每通道传输速率从 250 MB/s 上升至 500 MB/s。

2022 年 PCIe 6.0 规范正式发布，总传输速率提高至 64 GT/s。

2022年6月，PCI-SIG 联盟宣布 PCIe 7.0 版规范，单条通道（x1）单向可实现128 GT/s 传输速率，计划于2025年推出最终版本。

![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/f6a4a3004b3c4ceaa9567b2e7ef90eb0~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1406&h=431&s=102106&e=png&b=eeeeee)

**PCIe吞吐量(可用带宽)计算方法：**

`吞吐量=传输速率*编码方案`

传输速率为每秒传输量（GT/s），而不是每秒位数（Gbps），是因为传输量包括**不提供额外吞吐量的开销位**，比如：PCIe 1x和PCIe 2x使用8b/10b编码方案，导致占用了20%(=2/10)的原始信道带宽。

- GT/s，Giga transtion per second (千兆传输/秒)，即每一秒内传输的次数，重点在于描述物理层通信协议的速率属性，可以不和链路宽度等关联。
- Gbps，Giga Bits per second (千兆位/秒)。GT/s和Gbps之间不存在成比例的换算关系。

PCIe 2.0协议支持5.0GT/s，即每一条Lane上支持每秒钟传输5G个Bit，但这并不意味着PCIe 2.0协议的每一条Lane支持5Gbps的速率。为什么这么说呢，因为PCIe 2.0的物理层协议中使用的是8b/10b编码方案，即每传输8个Bit，需要发送10个Bit，这多出来的2Bit并不是对上层有意义的信息。那么，PCIe 2.0协议的每一条Lane支持`5*8/10=4Gbps=500MB/s`的速率。以一个PCIe 2.0 x8的通道为例，x8的可用带宽为`4*8=32Gbps=4GB/s`。

同理，PCIe 3.0协议支持8.0GT/s，即每一条Lane上支持每秒钟传输8G个Bit。而PCIe 3.0的物理层协议中使用的是128b/130b编码方案，即每传输128个Bit，需要发送130个Bit，那么，PCIe 3.0协议的每一条Lane支持`8*128/130=7.877GB/s=984.6MB/s`的速率。以一个PCIe 3.0 x16的通道为例，x16的可用带宽为`7.877*16=126.032 Gbps=15.754GB/s`。


**PCIE体系架构**：

PCIE体系架构一般包含根组件RC（root-complex），交换器switch，终端设备EP（endpoint）等类型的PCIE设备组成。RC在总线架构中只有一个，用于处理器和内存子系统与I/O设备之间的连接，而switch的功能通常是以软件形式提供的，它包括两个或更多的逻辑PCI到PCI的连接桥（PCI-PCI Bridge），以保持与现有PCI兼容。

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/2da8b7c4b2b044dfab80bb3c673043ed~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=852&h=600&s=203332&e=png&b=fefdfd)


### NVLink

**背景**：

算力的提升不仅依靠单张 GPU 卡的性能提升，往往还需要多 GPU 卡组合。在多 GPU 系统内部，GPU 间通信的带宽通常在数百GB/s以上，PCIe总线的数据传输速率容易成为瓶颈，且PCIe链路接口的串并转换会产生较大延时，影响GPU并行计算的效率和性能。

GPU发出的信号需要先传递到PCIe Switch, PCIe Switch中涉及到数据的处理，CPU会对数据进行分发调度，这些都会引入额外的网络延迟，限制了系统性能。

![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/e5298c9a3382455cad518360bc16a74f~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=700&h=356&s=89728&e=png&b=f8efe8)

为此，NVIDIA推出了能够提升GPU通信性能的技术——GPUDirect P2P技术，使GPU可以通过 PCI Express 直接访问目标GPU的显存，避免了通过拷贝到CPU host memory作为中转，大大降低了数据交换的延迟，但受限于PCI Express总线协议以及拓扑结构的一些限制，无法做到更高的带宽。此后，NVIDIA 提出了 NVLink 总线协议。


**NVLink简介**：

NVLink 是一种高速互连技术，旨在加快 CPU 与 GPU、GPU 与 GPU 之间的数据传输速度，提高系统性能。NVLink通过GPU之间的直接互联，可扩展服务器内的多GPU I/O，相较于传统PCIe总线可提供更高效、低延迟的互联解决方案。

![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/b46bcbcc54384297ab173b306a6ae58c~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=625&h=352&s=179166&e=png&b=353535)

NVLink的首个版本于2014年发布，首次引入了高速GPU互连。2016年发布的P100搭载了第一代NVLink，提供 160GB/s 的带宽，相当于当时 PCIe 3.0 x16 带宽（双向）的 5 倍。之后陆续发布了很多新版本，V100搭载的 NVLink2 将带宽提升到300GB/s ，A100搭载了NVLink3带宽为600GB/s。H100中包含18条第四代NVLink链路，总带宽（双向）达到 900 GB/s，是PCIe 5.0 x16带宽（双向）的7倍。

![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/6ac72f40a0374e7aa305034166b618fa~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=747&h=489&s=24935&e=png&b=eeeeee)

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/a24232222ccb49598a816bd923351aac~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1710&h=273&s=71583&e=png&b=ededed)

NVLink高速互联主要有两种：

- 第一种是以桥接器的形式实现。
- 另一种是在主板上集成 `NVLink` 接口。

![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/d07e4b62d39848b0a7aa9e37593ce6c4~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1033&h=381&s=382215&e=png&b=f9f6f6)

### NVSwitch

为了解决GPU之间通讯不均衡问题，NVIDIA引入NVSwitch。NVSwitch芯片是一种类似交换机的物理芯片（ASIC），通过NVLink接口可以将多个GPU高速互联到一起，可创建无缝、高带宽的多节点GPU集群，实现所有GPU在一个具有全带宽连接的集群中协同工作，从而提升服务器内部多个GPU之间的通讯效率和带宽。NVLink和NVSwitch的结合使NVIDIA得以高效地将AI性能扩展到多个GPU。

![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/936a389603f84830a4092394c7418358~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=688&h=401&s=48837&e=png&b=edeaea)


第一代 NVSwitch于2018年发布，采用台积电 12nm FinFET 工艺制造，共有 18 个 NVLink 2.0 接口。目前 NVSwitch 已经迭代至第三代。第三代 NVSwitch 采用台积电 4N 工艺（台积电 4N 工艺专为NVIDIA定制设计，并进行了一系列优化，它与普通台积电5nm节点相比，可实现更好的电源效率与性能，并且密度有所提升）构建，每个 NVSwitch 芯片上拥有 64 个 NVLink 4.0 端口，GPU 间通信速率可达 900GB/s。


![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/b31291309f6d40ae92b525e78a6fcd70~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1546&h=207&s=56495&e=png&b=f3f3f3)


### Nvidia GPU 服务器 PCIe 版 和 SXM 版的区别

英伟达GPU卡间互连的内存插槽有2种，一种是PCIe口，一种是SXM口。

PCIe口是一个相对通用的协议，PCIe口相对慢一些，SXM是专门用来做卡间互连的，SXM协议是铺在电路板上，SXM协议做卡间互连会更快，对NVLink原生支持更好，显存带宽比PCIe高一些。PCIe和SXM都可以用NVLink，但是SXM是更好使用NVLink的方法。  

SXM 架构是一种高带宽插座式解决方案，用于将 GPU 连接到 NVIDIA 专有的 DGX 和 HGX 系统。SXM 版 GPU 通过主板上集成的 NVSwitch 实现 NVLink 的连接，不需要通过主板上的PCIe进行通信，它能支持8块GPU卡的互联互通，实现了GPU之间的高带宽。未阉割的A100是600GB/s、H100是900GB/s，阉割过的A800、H800为400GB/s。

把 PCIe 版 GPU 卡插到 PCIe 插槽上，就可以和CPU、同一个服务器上其他的GPU卡进行通信，也可以通过网卡与其他的服务器节点上的设备进行通信，这种就是PCIe的通信方式，但是这种传输速度不快。如果想要和SXM一样，有很快的传输速度，可以使用NVlink桥接器实现GPU和CPU之间的通信，但是和SXM不一样的地方就是它只能实现2块GPU卡之间的通信。即 PCIe 版只有成对的 GPU 通过 NVLink Bridge 连接，通过 PCIe 通道进行数据通信。同时，最新的PCIe网络带宽有128GB/s的限制。

![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/3bc4f115338245a88091182bddaaa1d1~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1982&h=677&s=821728&e=png&b=09428d)


### TCP/IP

TCP/TP （或传输控制协议/Internet 协议）用于通过 Internet 互连网络设备。它确定了数据应该如何被打包、寻址、传输、路由和接收。TCP/IP 非常重视两台计算机之间的准确数据传输。如果系统在一次发送消息时遇到问题，则必须重新发送整个消息。

此外，TCP/IP 的功能分为四个不同的层：**数据链路层、互联网层、传输层和应用层**。数据在被另一端接收之前必须经过这四层。然后，TCP/IP 将通过以相反顺序传递层来重组数据并将其呈现给接收器。这样，您可以通过升级某些层而不是整个系统来提高数据中心的性能或安全性。

### RDMA

RDMA(远程直接数据存取)就是为了解决网络传输中服务器端数据处理的延迟而产生的，**无需使用CPU，就可以从一个主机或服务器的内存直接访问另一主机或服务器的内存**。它释放了CPU去执行其应做的工作，比如：运行应用程序和处理大量数据。这既提高了带宽又降低了延迟、抖动和 CPU 消耗。

对比传统的网络传输机制，RDMA无需操作系统和TCP/IP协议栈的介入。**RDMA的内核旁路机制，允许应用与网卡之间的直接数据读写**，将服务器内的数据传输时延降低到1us以下。同时，RDMA的内存零拷贝机制，允许接收端直接从发送端的内存读取数据，极大的减少了CPU的负担，提升CPU的效率。


![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/1e4c8404a9124f48ad6ee0e914489ca8~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1080&h=443&s=125224&e=png&b=ffffff)

大致有三类RDMA网络，分别是Infiniband、RoCE、iWARP。其中，Infiniband是一种专为RDMA设计的网络，从硬件级别保证可靠传输 ，而RoCE 和 iWARP都是基于以太网的RDMA技术，支持相应的verbs接口。

RDMA最早在Infiniband传输网络上实现，技术先进，但是价格高昂(**只有Mellanox（现已被英伟达收购）和Intel（2012年，英特尔公司出资收购了QLogic的InfiniBand技术）供应商提供全套网络解决方案**)，后来业界厂家把RDMA移植到传统Ethernet以太网上，降低了RDMA的使用成本，推动了RDMA技术普及。在Ethernet以太网上，根据协议栈融合度的差异，分为iWARP和RoCE两种技术，而RoCE又包括**RoCEv1和RoCEv2两个**版本(RoCEv2的最大改进是支持IP路由)。各RDMA网络协议栈的对比，如下图所示：

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/7d143d6d63a44f2caffe208201eec22d~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1024&h=532&s=463868&e=png&b=fdfbfb)

**IB（InfiniBand）：** 基于 InfiniBand 架构的 RDMA 技术，由 IBTA（InfiniBand Trade Association）提出。搭建基于 IB 技术的 RDMA 网络需要专用的 IB 网卡和 IB 交换机。

**iWARP（Internet Wide Area RDMA Protocal）：** 基于 TCP/IP 协议的 RDMA 技术，由 IETF 标 准定义。iWARP 支持在标准以太网基础设施上使用 RDMA 技术，但服务器需要使用支持iWARP 的网卡。

**RoCE（RDMA over Converged Ethernet）：** 基于以太网的 RDMA 技术，也是由 IBTA 提出。RoCE 支持在标准以太网基础设施上使用RDMA技术，但是需要交换机支持无损以太网传输，需要服务器使用 RoCE 网卡。

在三种主流的RDMA技术中，可以划分为两大阵营。一个是IB技术，另一个是支持RDMA的以太网技术(RoCE和iWARP)。其中, IBTA力挺的技术自然是IB和RoCE, Mellanox公司是这方面的急先锋。而iWARP则是IEEE/IETF力挺的技术，主要是Chelsio公司在推进。

在存储领域，支持RDMA的技术早就存在，比如：SRP(SCSI RDMA Protocol)和iSER(iSCSI Extensions for RDMA)。如今兴起的NVMe over Fabrics如果使用的不是FC网络的话，本质上就是 NVMe over RDMA。 换句话说，NVMe over InfiniBand, NVMe over RoCE 和 NVMe over iWARP 都是 NVMe over RDMA。


## InfiniBand

InfiniBand（直译为 “无限带宽” 技术，缩写为IB）是一个为大规模、易扩展机群而设计的**网络通信技术协议**。可用于计算机内部或外部的数据互连，服务器与存储系统之间直接或交换互连，以及存储系统之间的互连。

InfiniBand 最重要的一个特点就是**高带宽**、**低延迟**，因此在高性能计算项目中广泛的应用。 主要用于高性能计算（HPC）、高性能集群应用服务器和高性能存储。

### InfiniBand 链路速率

InfiniBand在物理层定义了多种链路速度，例如：1X，4X，12X。每个单独的链路是四线串行差分连接（每个方向两根线）。


![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/4470dc47a85b40d886470cdbf1212bd9~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=318&h=286&s=62025&e=png&b=cbd6fc)

以早期的SDR（单数据速率）规范为例，1X链路的原始信号带宽为2.5Gbps，4X链路是10Gbps，12X链路是30Gbps。  1X链路的实际数据带宽为2.0Gbps（因为采用8b/10b编码）。由于链路是双向的，因此相对于总线的总带宽是4Gbps。


随着时间的推移，InfiniBand的网络带宽不断升级，下图展示了 InfiniBand 从SDR、DDR、QDR、FDR、EDR发展到HDR、NDR的网络带宽，其速度是基于 4x 链路速度。

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/52c2a6d6c45f4ac4be1b19c2032bd380~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1080&h=248&s=81845&e=png&b=fefefe)

- SDR（Single Data Rate）：2.5Gb/s (10Gb/s for 4x)。
- DDR（Double Data Rate）：5 Gb/s (20Gb/s for 4x)。
- QDR（Quad Data Rate）：10 Gb/s (40Gb/s for 4x)。
- FDR（Fourteen Data Rate）：14Gb/s (56Gb/s for 4x)。
- EDR（Enhanced Data Rate）：25 Gb/s (100Gb/s for 4x)。
- HDR（High Data Rate）：50 Gb/s (200Gb/s for 4x)。
- NDR（Next Data Rate）：100 Gb/s (400Gb/s for 4x)。

![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/23c8aee1729541f0bf0b02c0fab555cb~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1080&h=632&s=139743&e=png&b=f7f7f7)


### InfiniBand 网络互连产品

InfiniBand 网络中，使用的线缆区别于传统的以太网线缆和光纤线缆。针对不同的连接场景，需使用专用的InfiniBand线缆。

InfiniBand网络互连产品包括：**DAC高速铜缆**、**AOC有源线缆**以及**光模块**。

DAC高速线缆和AOC有源光缆都是用于数据中心、高性能计算机等大容量储存器设备间的传输设备。

**DAC高速线缆**，也叫直连铜缆（Direct Attach Copper cable）, 它的线材是铜缆，是低压脉冲传输信号；因为材料的不同导致功耗、传输距离和价格的不同，DAC高速线缆的功耗比较低，但传输距离相对比较短，低于10米。价格方面相对便宜一些。

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/b017a8bae70644d8bc387856ea456539~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=800&h=800&s=137041&e=png&b=f8f6f9)

**AOC有源光缆**（Active Optial Cable），它的线材是光缆，为光信号传输，通过电-光-电的转换；功耗相对比较大些但传输的距离可达到100米，价格方面相对高些。

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/c9ad44b030d74bea94f59466dbe08850~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=710&h=533&s=241635&e=png&b=f8f7f7)


**光模块**的作用也是光电信号之间的转换，主要用于交换机与设备之间传输的载体，和光纤收发器的原理相同，只是光模块相比收发器更具效率性、安全性。光模块按照封装形式分类，常见的有 SFP，SFP+，XFP，SFP28,QSFP+,QSFP28 等。

**光纤收发器**是将短距离的电信号和长距离的光信号进行转换的设备，一般应用在远距离传输中，通过光纤进行传输，将电信号转换成光信号发送出去，同时，在接收端将接收到的光信号转换成电信号。在很多地方也被称之为光电转换器(Fiber Converter)。光纤收发器为需要将系统从铜线升级到光纤，为缺少资金、人力或时间的用户提供了一种廉价的方案。

![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/536b4be32d9449babb8907b900b0adac~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=771&h=470&s=217027&e=png&b=f0eeee)

**光模块与光纤收发器如何配对使用？**

1. 波长和传输距离必须一致，比如：采用1310nm波长，传输距离应该是10KM/20KM。 
2. 光纤跳线尾纤接口选择需注意，一般光纤收发器采用的SC口，光模块采用的是LC口。  
3. 速率必须一样，比如：千兆收发器对应 1.25G 光模块，百兆连百兆，千兆连千兆。
4、光模块类型需要采用相同类型，单纤对单纤，双纤对双纤。

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/5c6e868c217341cf93d59cb373a94d3c~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1294&h=482&s=422163&e=png&b=bdd3e0)


### InfiniBand 的网络架构

InfiniBand 是一种基于通道的结构，组成单元主要分为四类：

- HCA（Host Channel Adapter，主机通道适配器）
- TCA（Target Channel Adapter，目标通道适配器）
- InfiniBand link（连接通道，可以是电缆或光纤，也可以是板上链路）
- InfiniBand交换机和路由器（组网用的）

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/da3bbe48cb1947d7afe16ad22294e945~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1080&h=793&s=80831&e=png&b=ffffff)

通道适配器就是搭建InfiniBand通道用的。所有传输均以通道适配器开始或结束，以确保安全或在给定的QoS（服务质量）级别下工作。  

使用 InfiniBand 的系统可以由多个子网（Subnet）组成，每个子网最大可由 6 万多个节点组成。
- 子网内部，InfiniBand 交换机进行二级处理。
- 子网之间，使用路由器或网桥进行连接。


![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/9b241ef3b4294bf4ad4cf1d5203e9e53~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1049&h=519&s=63954&e=png&b=ffffff)

InfiniBand 的二级处理过程非常简单，每个 InfiniBand 子网都会设一个子网管理器，生成16位的 LID（本地标识符）。InfiniBand 交换机包含多个 InfiniBand 端口，并根据第二级本地路由标头中包含的LID，将数据包从其中一个端口转发到另一个端口。**除管理数据包外，交换机不会消耗或生成数据包**。

简单的处理过程，加上自有的Cut-Through技术，InfiniBand 将转发时延大幅降低至 100ns 以下，明显快于传统以太网交换机。  

在 InfiniBand 网络中，数据同样以数据包（最大4KB）的形式传输，采用的是串行方式。

### InfiniBand 的协议栈

InfiniBand 协议同样采用了分层结构，各层相互独立，下层为上层提供服务，如下图所示：

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/bff99416b2a843ebb08b36424e03383b~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1080&h=622&s=87147&e=png&b=fefefe)

- **物理层**定义了在线路上如何将比特信号组成符号，然后再组成帧、数据符号以及包之间的数据填充等，详细说明了**构建有效包的信令协议**等。
- **链路层**定义了数据包的格式以及数据包操作的协议，如：流控、 路由选择、编码、解码等。  
- **网络层**通过在数据包上添加一个40字节的全局的路由报头（Global Route Header, GRH）来进行路由的选择，对数据进行转发。**在转发的过程中，路由器仅仅进行可变的CRC校验，这样就保证了端到端的数据传输的完整性**。Infiniband报文封装格式如下图所示：

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/16014600fd2c4c1a8f52a3e2555a3efd~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1080&h=533&s=242259&e=png&b=fcfafa)

- **传输层**再将数据包传送到某个指定的队列偶（Queue Pair, QP）中，并指示 QP 如何处理该数据包。  

可以看出，InfiniBand 拥有自己定义的 1-4 层格式，是一个完整的网络协议。端到端流量控制，是 InfiniBand 网络数据包发送和接收的基础，可以实现无损网络。  

> QP（队列偶）说明：
> 
> QP是RDMA技术中通信的基本单元。队列偶就是一对队列，SQ（Send Queue，发送工作队列）和 RQ（Receive Queue，接收工作队列）。用户调用API发送接收数据的时候，实际上是将数据放入QP当中，然后以轮询的方式，将QP中的请求一条条的处理。

![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/51bb447ee5664c22b24cbebfe9af1a08~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1080&h=628&s=83343&e=png&b=ebebeb)



### Mellanox OFED 软件栈

Mellanox OFED 是一个单一的软件堆栈，包括驱动、中间件、用户接口，以及一系列的标准协议 IPoIB、SDP、SRP、iSER、RDS、DAPL(Direct Access Programming Library)，支持 MPI、Lustre/NFS over RDMA 等协议，并提供 Verbs 编程接口；Mellanox OFED 由开源 OpenFabrics 组织维护。

Mellanox OFED 软件堆栈是承载在 InfiniBand 硬件和协议之上的，软件通过协议和硬件进行有效的数据传输。

![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/5b954db7a5c44dc5870e82d2d2c0845e~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1136&h=720&s=2458522&e=png&b=e6e3e3)


### OpenSM 子网管理器

OpenSM 软件是符合InfiniBand的子网管理器(SM)，运行在Mellanox OFED软件堆栈进行 IB 网络管理，管理控制流走业务通道，属于带内管理方式。

OpenSM 包括**子网管理器、背板管理器和性能管理器**三个组件，绑定在交换机内部的必备部件。提供非常完备的管理和监控能力，如：**设备自动发现、设备管理、Fabric可视化、智能分析、健康监测**等等。

![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/63608c4c208f4ce5890da2e6ce52ae66~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1280&h=628&s=2416125&e=png&b=f1f0ef)


### InfiniBand 组网

InfiniBand 组网跟普通的交换机不太一样，InfiniBand 的组网成本很高。如果希望这个网络中任何两个计算节点的网卡之间互相无损地通信，需要使用一种叫做胖树（Fat Tree）的网络拓扑，大概是如下一种拓扑结构，方块是交换机，椭圆是计算节点。

胖树主要有两层，上面一层是核心层，不连任何计算节点，它的功能就是转发流量；下面一层是接入层，接入各类计算节点。

胖树拓扑成本高的主要原因是：某一个汇聚交换机上，假如有36个口，那如果为了达到无损速率，一半的口，也就是18个口可以给计算节点连，剩下一半要连到上层的核心交换机上。要知道，任何一根线，就是1万多块钱呢，如果达到无损，就要冗余地做这些连接。

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/77ea150e49e146d2a25a4e7dbbf4520e~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=591&h=331&s=47543&e=png&b=ffffff)


### NVIDIA InfiniBand 商用产品

Mellanox 在全球 InfiniBand 市场的占有率基本上无敌的存在，在英伟达收购 Mellanox 之后，也于2021年推出了自己的第七代 NVIDIA InfiniBand 架构：NVIDIA Quantum-2。

NVIDIA Quantum-2 平台包括：NVIDIA Quantum-2 系列交换机、NVIDIA ConnectX-7 InfiniBand 适配器、BlueField-3 InfiniBand DPU以及电缆。

![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/a93953336c7a426880e2f97a42898a29~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1499&h=696&s=662947&e=png&b=faf9f9)

**NVIDIA Quantum-2 系列交换机**采用紧凑型1U设计，包括风冷和液冷版本。交换机的芯片制程工艺为7nm，单芯片拥有570亿个晶体管（比A100 GPU还多）。单个交换机采用64个400Gb/s端口或128个200Gb/s端口的灵活搭配，提供总计 51.2Tb/s的双向吞吐量。NVIDIA NDR 400Gb/s InfiniBand 交换机如下图所示：

![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/6cbb986d98c04b42a5f1d8789b53a9ef~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=630&h=354&s=113706&e=png&b=1b1b1b)

**NVIDIA ConnectX-7 InfiniBand 适配器**支持PCIe Gen4和Gen5，具有多种外形规格，可提供 400Gb/s 吞吐量。

![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/7e2025e0d5244d199fd554174dfe12f6~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=391&h=265&s=58835&e=png&b=fefefe)



### InfiniBand 常用命令
 
- `ibv_asyncwatch`：监视 InfiniBand 异步事件  
- `ibv_devices` 或 `ibv_devinfo`： 列举 InfiniBand 设备或设备信息  - `ibstatus`：查询 IB 设备的基本状态
- `ibping`： 验证 IB 节点之间的连通性
- `ibtracert`：跟踪 IB 路径
- `iblinkinfo`：查看IB交换模块的所有端口的连接状态。此命令会将集群内所有的IB交换模块都进行列举。


## 通信软件

通信软件指用于分布式训练时，多个计算设备之间的集合通信。在分布式系统中，各个节点间往往存在大量的集合通信需求，而我们可以用消息传递接口 (Message Passing Interface，MPI，一套集合通信相关的接口标准) 来定义一些比较底层的消息通信行为。譬如 Reduce、AllReduce、Scatter、Gather、AllGather 等。

常见的集合通信库（如：Open MPI、Gloo、NCCL等）都在 MPI 的基础上，对各种集合通信的模式和算法作了各自的实现。

**Open MPI**：

Open MPI 是一个开源 MPI（消息传递接口 ）的实现，由学术，研究和行业合作伙伴联盟开发和维护。因此，Open MPI 可以整合高性能计算社区中所有专家，技术和资源，以构建可用的最佳 MPI 库。


**Gloo**：

Gloo 是 Facebook 开源的一套集体通信库，提供了对机器学习中有用的一些集合通信算法。如：Barrier，Broadcast，AllReduce。

**NCCL**：

NCCL（Nvidia Collective multi-GPU Communication Library）是英伟达基于 NVIDIA GPU 的一套开源的集合通信库，如其官网描述：NVIDIA 集合通信库（NCCL）实现了针对 NVIDIA GPU 性能优化的多 GPU 和多节点集合通信原语。NCCL 提供了诸如 All Gather，All Reduce，Broadcast，Reduce，Reduce-Scatter 等实现，这些实现优化后可以通过 PCIe、 NVLink、InfiniBand 等高速互联，从而实现高带宽和低延迟。

因为 NCCL 是 NVIDIA 基于自身硬件定制的，能做到更有针对性且更方便优化，故在英伟达硬件上，NCCL 的效果往往比其它的通信库更好。

NCCL主要做几件事：**探测计算节点的网络设备和拓扑结构**，使用算法自动调优选择一个最优的通信方式。

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/eb1018950a05412b93e30ab79b808039~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=537&h=276&s=72386&e=png&b=fdfafa)



## NCCL 集合通信库

### 通信原语

并行任务的通信一般可以分为 Point-to-point communication 和 Collective communication 。

P2P 通信这种模式只有一个sender和一个receiver，实现起来比较简单。

集合通信包含多个sender多个receiver，一般的通信原语包括broadcast，gather，all-gather，scatter，reduce，all-reduce，reduce-scatter，all-to-all等。


简单介绍几个常用的操作：

**Reduce**：从多个sender那里接收数据，最终combine到一个节点上。


![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/a6a99b0337024a50ae84d1aecd9ea61a~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=662&h=216&s=53187&e=png&b=f3f2f2)

**All-reduce**：从多个sender那里接收数据，最终combine到每一个节点上。


![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/45af7fe062af410f8efb9fc3bca546b2~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=671&h=214&s=75137&e=png&b=fefcfc)


### NCCL 实现

NCCL 实现成 CUDA C++ kernels，包含3种 primitive operations： Copy，Reduce，ReduceAndCopy。

- NCCL 1.0 版本只支持单机多卡，卡之间通过 PCIe、NVlink、GPUDirect P2P 来通信。
- NCCL 2.0 支持多机多卡，多机间通过 Sockets (Ethernet) 或者 InfiniBand with GPUDirect RDMA 通信。 

单机内多卡通过PCIe以及CPU socket通信。


![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/d6a6b3be0c3c4a509bd9d5560fd926e0~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=553&h=183&s=42945&e=png&b=fefcfc)

多机通过InfiniBand通信，在多机多卡内部，也要构成一个通信环。

![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/4fef434d39714c13bf66dfa2dce11d49~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=616&h=176&s=56241&e=png&b=fefcfc)


### 对比 NCCL 在不同硬件架构下网络带宽

下图是 Allreduce 在单机不同架构下的速度比较：

![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/bf622a93a1fb4d5a91974c6916856df9~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=659&h=335&s=77763&e=png&b=fcfbfb)

前面三个是单机多卡典型的三种连接方式：

- 第一种是两个GPU通过CPU然后通过QPI和另一个CPU上的两块卡相连，因此速度最慢，但也能达到>5GB/s。
- 第二种是两个GPU通过PCIe switch相连后再经过CPU连接，速度会稍微低一点。
- 第三种是四张卡都在一个PCIe switch上，所以带宽较高，能达到>10GB/s PCIe的带宽大小。

第四种是DGX-1架构，这是Nvidia推出的深度学习平台，带宽能达到60GB/s。



下图是 Allreduce 多机下的速度表现。其中，左图2机8卡，机内PCIe，机间InfiniBand能达到>10GB/s的速度，InfiniBand基本上能达到机内的通信速度；右图4机32卡，机内NVLink，机间InfiniBand，带宽能达到>40GB/s。

![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/e8abc881b9a749e094c5ed08454fe12e~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=661&h=380&s=102021&e=png&b=fefdfd)


下图是 NCCL 在 CNTK ResNet50 上的可扩展性（scalability），32 卡基本能达到线性加速比。

![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/1cd8d50a4f3b4a79aa3f8a11f7bec3b1~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=665&h=402&s=102462&e=png&b=fefdfd)



### NCCL 常见的环境变量设置


**NCCL_P2P_DISABLE**

该变量禁用 P2P 传输，该传输使用 NVLink 或 PCI 在GPU之间使用CUDA直接访问。

设定为 1 相当于设置 `NCCL_P2P_LEVEL=0`，并且会被 NCCL_P2P_LEVEL 的值所覆盖。


**NCCL_P2P_LEVEL**：

该变量允许用户精细地控制何时在GPU之间使用 P2P 传输。该级别定义了NCCL将使用P2P传输的GPU之间的最大距离。

如果未指定，NCCL 将尝试根据其运行的体系结构和环境来最佳选择一个值。

可选值：

- LOC：从不使用P2P（始终禁用）
- NVL ：当 GPU 通过 NVLink 连接时使用 P2P
- PIX ：当 GPU 位于同一 PCI 交换机上时使用 P2P。
- PXB：当 GPU 通过 PCI 交换机（可能是多跳）连接时使用 P2P。
- PHB ：当 GPU 位于同一 NUMA 节点上时使用 P2P。 流量将通过 CPU。
- SYS ：在 NUMA 节点之间使用 P2P，可能跨越 SMP 互连（例如：QPI/UPI）。



**NCCL_NET_GDR_LEVEL**：

该变量允许用户精细控制何时在NIC和GPU之间使用GPUDirect RDMA。该级别定义NIC和GPU之间的最大距离。

如果未指定，NCCL 将尝试根据其运行的体系结构和环境来最佳选择一个值。

可选值：

- LOC：从不使用 GPU Direct RDMA。（始终禁用）
- PIX：当 GPU 和 NIC 位于同一 PCI 交换机上时，使用 GPU Direct RDMA。
- PXB：当 GPU 和 NIC 通过 PCI 交换机（可能是多跳）连接时，使用 GPU Direct RDMA。
- PHB ：当 GPU 和 NIC 位于同一 NUMA 节点上时，使用 GPU Direct RDMA。 流量将通过 CPU。
- SYS ：即使跨 NUMA 节点之间的 SMP 互连（例如 QPI/UPI）也使用 GPU Direct RDMA。 （始终启用）



**NCCL_NET_GDR_READ**：

只要 GPU-NIC 距离在 NCCL_NET_GDR_LEVEL 指定的距离内，NCCL_NET_GDR_READ 变量就会在发送数据时启用 GPU Direct RDMA。 

- 2.4.2之前，默认情况下禁用GDR读取，即发送数据时，数据先存储在 CPU 内存中，然后再发送到 InfiniBand 卡。 
- 自 2.4.2 起，基于 NVLink 的平台默认启用 GDR 读取。

注意：已知在某些平台（例如：PCI-E）上，发送数据时直接从 GPU 内存读取比从 CPU 内存读取稍慢。

可选值为0或1。定义并设置为1以使用GPU Direct RDMA直接将数据发送到NIC（绕过CPU）。

在 2.4.2 之前，所有平台的默认值都是 0。 自 2.4.2 起，基于 NVLink 的平台的默认值为 1，否则为 0。


**NCCL_IB_DISABLE**：

该变量将禁用 NCCL 要使用的IB传输。NCCL 将使用IP sockets 。

定义并设置为1以强制使用IP sockets 。

**NCCL_SOCKET_IFNAME**：

指定NCCL使用的SOCKET网卡。如：`NCCL_SOCKET_IFNAME=bond0,eth0`。


**NCCL_IB_HCA**：

该变量指定要用于通信的 RDMA 接口。使用IB通信必须要设置的（指定NCCL使用的IB网卡）。 可以通过 ibstat 查看IB网卡名。

用法：

定义一个前缀列表来过滤要由 NCCL 使用的接口。使用 ^ 符号，NCCL 将排除以列表中任何前缀开头的接口。还可以使用 : 符号来指定特定的端口。要匹配（或不匹配）确切的接口名称而不是前缀，在字符串前面加上 = 字符。

示例：

- `mlx5`：使用以 mlx5 开头的所有卡的所有端口。
- `=mlx5_0:1,mlx5_1:1`：使用卡 mlx5_0 和 mlx5_1 的端口 1。
- `^=mlx5_1`：不使用卡 mlx5_1。

比如：
NCCL_IB_HCA=mlx5_2,mlx5_3,mlx5_4,mlx5_5

>注意：
>
> 如果不加前缀 =，使用 mlx5_1 将同时选择 mlx5_1 和 mlx5_10 到 mlx5_19（如果存在）。因此，始终建议添加前缀 = 以确保精确匹配。

使用建议：

通过这个环境变量可以调整NIC（Network Interface Controller）数量，NIC 通常是一块插入计算机主板上的扩展卡，更多NIC，节点带宽更大。通过控制NIC数量可以控制节点间通信带宽。


**NCCL_IB_TIMEOUT**：

该变量用于控制InfiniBand Verbs超时。取值范围1-22。

超时时间的计算公式为4.096微秒 * 2 ^ timeout，正确的值取决于网络的大小。增加该值可以在非常大的网络上提供帮助，例如，如果NCCL在调用ibv_poll_cq时出现错误12。

使用建议：

在大模型训练任务中设置成最大值22，可以减少不少nccl timeout异常。


**NCCL_IB_RETRY_CNT**

该变量控制 InfiniBand 的重试次数。

使用建议：

在大模型训练任务中设置成13，尽可能多重试。


**NCCL_PXN_DISABLE**：

禁止使用非本地 NIC 的进行节点间通信，使用 NVLink 和一个中间 GPU。

使用建议：

设置成1。在PyTorch中进行跨节点all-to-all通信时，如果该环境变量是0会出现异常。

**NCCL_DEBUG_FILE**：

设置一个文件地址，变量用于将NCCL的调试日志输出到文件中，有助于调试NCCL。

**NCCL_IB_PCI_RELAXED_ORDERING**：

启用 IB Verbs 传输的 Relaxed Ordering。Relaxed Ordering可以极大地提高虚拟化环境下 InfiniBand 网络的性能。

传统的顺序执行（Strict Ordering）要求数据在发送和接收之间按照严格的顺序进行传输和处理。这种机制可以确保数据的顺序性，但可能会导致性能瓶颈，特别是在高负载和复杂通信模式下。

而Relaxed Ordering允许数据在发送和接收之间进行乱序传输和处理。这意味着系统可以更灵活地调度和处理数据，提高并行性和吞吐量。Relaxed Ordering 机制在虚拟化环境中尤其有益，因为它可以减少虚拟机之间的争用和延迟，提高整体性能。

接受的取值： 
- 设置为 2，如果可用，自动使用Relaxed Ordering。
- 设置为 1，强制使用Relaxed Ordering，如果不可用，则失败。
- 设置为 0，禁用使用Relaxed Ordering。
 
默认值为 2。建议设置成 1。


**NCCL_SHM_DISABLE**：

该变量禁用共享内存（SHM）传输。

在P2P不能生效的情况下，是否使用CPU的共享内存来传输数据。 当 SHM 禁用时，NCCL 使用网络（ InfiniBand 或 IP sockets）在 CPU sockets 之间进行通信。


## InfiniBand 在 AI 集群中的应用

### GPUDirect 简介

GPUDirect 是 NVIDIA 开发的一项技术，可实现 GPU 与其他设备（例如网络接口卡 (NIC) 和存储设备）之间的直接通信和数据传输，而不涉及 CPU。

传统上，当数据需要在 GPU 和另一个设备之间传输时，数据必须通过 CPU，从而导致潜在的瓶颈并增加延迟。使用 GPUDirect，网络适配器和存储驱动器可以直接读写 GPU 内存，减少不必要的内存消耗，减少 CPU 开销并降低延迟，从而显著提高性能。GPU Direct 技术包括 GPUDirect Storage、GPUDirect RDMA、GPUDirect P2P 和 GPUDirect Video。


### GPUDirect 发展简史

- GPUDirect Shared Memory (2012) ： Nvidia在PCIe上实现了单机上的GPUDirect Shared Memory 技术；  
- GPUDirect P2P (2014)： Nvidia在PCIe上实现了单机上的GPUDirect P2P技术；  
- NVLink（2014） ：解决了单机多卡通信时PCIe瓶颈问题；  
- GPUDirect RDMA（2014）：提升多机多卡通信性能；

### GPUDirect Peer to Peer（P2P）简介

GPUDirect Peer-to-Peer(P2P) 技术主要用于单机GPU间的高速通信，它使得**GPU可以通过PCI Express直接访问目标GPU的显存**，避免了通过拷贝到CPU host memory作为中转，大大降低了数据交换的延迟。

以深度学习应用为例，主流的开源深度学习框架（如：TensorFlow、MXNet）都提供了对GPUDirect P2P的支持，NVIDIA开发的NCCL(NVIDIA Collective Communications Library)也提供了针对GPUDirect P2P的特别优化。

通过使用GPUDirect P2P技术可以大大提升深度学习应用单机多卡的扩展性，使得深度学习框架可以获得接近线性的训练性能加速比。

### GPUDirect RDMA 简介

所谓 GPUDirect RDMA，就是计算机1的GPU可以直接访问计算机2的GPU内存。而在没有这项技术之前，GPU需要先将数据从GPU内存搬移到系统内存，然后再利用RDMA传输到计算机2，计算机2的GPU还要做一次数据从系统内存到GPU内存的搬移动作。GPUDirect RDMA技术使得进一步减少了GPU通信的数据复制次数，通信延迟进一步降低。

![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/02ea5dd1ecf746cbaa6628ae78bf8db6~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=3840&h=2160&s=192623&e=png&a=1&b=2083bd)

使用 GPUDirect RDMA 两个 GPU 设备必须共享相同的上游 PCI Express root complex。

![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/f4c74697fc854927bc45a2dbc23aae6c~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=747&h=506&s=15351&e=png&b=ffffff)



## InfiniBand 在 NVIDIA DGX 集群中应用

**DGX-1 集群中应用 InfiniBand**：
 
下图展示了 DGX-1 配有四个 EDR InfiniBand 卡（每个 100 Gb/s）和两个 10Gb/s 以太网卡（铜质）。 这些网络接口可用于将 DGX-1 连接到网络以进行通信和存储。

每两个 GPU 都连接到系统板上的一个 PCIe 交换机。 该交换机还连接到 InfiniBand (IB) 网卡。 为了减少延迟并提高吞吐量，来自这两个 GPU 的网络流量应流向关联的 IB 卡。 这就是 DGX-1 设备中有四张 IB 卡的原因。

![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/e2af36e854ea4ce8ac99993d861ae893~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1280&h=482&s=1854438&e=png&b=fefefe)

如果您想使用InfiniBand（IB）网络连接DGX设备，理论上，您只需使用其中一张IB卡即可。 然而，这些数据流量将强行通过 CPU 之间的 QPI 链路，这对于 GPU 流量来说是一个非常慢的链路（即，它成为瓶颈）。更好的解决方案是使用两张 IB 卡，一张连接到每个 CPU。这可以是 IB0 和 IB2，或者 IB1 和 IB3，或者 IB0 和 IB3，或者 IB1 和 IB2。 这将大大减少必须穿越 QPI 链路的流量。 最佳性能始终是使用 IB 交换机的所有四个 IB 链路。

使用 IB 链路是将所有四个 IB 卡连接到 IB 结构的最佳方法。 如果您使用多个 DGX 设备进行训练，这将带来最佳性能（完全的平分带宽和最低延迟）。

通常，最小的 IB 交换机配有 36 个端口。 这意味着单个 IB 交换机可容纳使用全部四张 IB 卡的九个 DGX-1 设备。 这允许从 DGX-1 到交换机的带宽为 400 Gb/s。

如果您的应用程序不需要 DGX-1 设备之间的带宽，则可以如前所述为每个 DGX-1 使用两个 IB 连接。 这允许您将最多 18 个 DGX-1 设备连接到单个 36 端口 IB 交换机。

注意：**不建议仅使用单个 IB 卡**，但如果由于某种原因采用这种配置，则您最多可以将 36 个 DGX-1 设备连接到单个交换机。

对于大量 DGX-1 设备，您可能必须使用两级交换网络。 经典的 HPC 配置是在第一级使用 36 端口 IB 交换机（有时称为叶子（Leaf）交换机），并将它们连接到单个大型核心交换机，有时称为导向器级（director class）交换机。 最大的导向器级InfiniBand交换机有648个端口。当然您也可以使用多个核心交换机，但配置会变得相当复杂。


![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/d0a6402a97d14a18ba3ea1f39c4c1337~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=763&h=720&s=1651546&e=png&b=fefefe)

对于两级交换网络，如果每个 DGX-1 设备的全部 4 个 IB 卡都用于连接到 36 端口交换机，并且没有过度订阅，则每个交换机的 DGX-1 设备的最大数量为 4。这时每个 DGX-1 有 4 个端口进入交换机，总共 16 个端口。然后，从叶子交换机到核心交换机（导向器级交换机）有 16 个上行链路。总共 40 台 36 端口叶子交换机可连接到 648 端口核心交换机 (648/16)。 这导致 160（40 * 4） 个 DGX-1 设备（共640卡）以全对分带宽连接。

当然您还可以在设计 IB 网络时使用所谓的过度订阅。过度订阅意味着来自**上行链路的带宽小于进入设备的带宽**（换句话说，带宽性能较差）。如果我们使用从 DGX-1 设备到第一级交换机（36 端口叶交换机）的 2:1 超额订阅，则每个 DGX-1 设备仅使用两个 IB 卡连接到交换机。与使用所有四张卡相比，这会导致带宽更少，并且延迟也会更高。

如果我们保持从叶子交换机到核心交换机的网络带宽为 1:1（换句话说，没有过度订阅，全对分带宽），那么我们可以将九个 DGX-1 设备放入一个单叶子交换机（从 DGX 设备到叶子交换机的总共 18 个端口以及到核心交换机的 18 个上行链路端口）。结果是总共36（648/18）台叶子交换机可以连接到核心交换机。 这使得总共 324（36 * 9） 个 DGX-1 设备可以连接在一起。

您还可以通过使用从叶子交换机到核心交换机的超额订阅来进一步定制 IB 网络。 这可以通过**使用每个 DGX 设备到叶子交换机的四个 IB 连接**，然后对核心交换机进行 2:1 超额订阅，甚至使用到叶子交换机的两个 IB 连接，然后对核心交换机进行 2:1 超额订阅来完成。 

InfiniBand 网络的另一个重要方面是子网管理器 (SM)。 SM仅管理IB网络。 任何时候都有一个 SM 管理 IB 结构，但您可以让其他 SM 运行并准备好在第一个 SM 崩溃时接管。 选择运行多少个 SM 以及在何处运行它们会对集群的设计产生重大影响。

首先要做的决定是**在哪里运行 SM**。 

如果您愿意，它们可以在 IB 交换机上运行。 这称为硬件 SM，因为它在交换机硬件上运行。 这样做的优点是您不需要任何其他也可以运行 SM 的服务器。 

在节点上运行 SM 称为软件 SM。 运行硬件 SM 的一个缺点是，如果 IB 流量很大，SM 可能会遇到困难。 对于大量 IB 流量和较大的网络，最佳实践是在专用服务器上使用软件 SM。

要做的第二个决定是您**想要运行多少个 SM**。 您至少必须运行一个 SM。 最便宜的解决方案是运行单个硬件 SM。 这对于 DGX-1 设备的小集群（可能是 2-4 个）来说效果很好。 随着单元数量的增加，您将需要考虑同时运行两个 SM 以获得 HA（高可用性）功能。 您需要 HA 的原因是集群上有更多用户，并且集群故障比少量设备故障产生的影响更大。

随着设备数量的增长，请考虑在专用服务器（软件 SM）上运行 SM。 您还需要为集群运行至少两个 SM。 理想情况下，这意味着 SM 有两台专用服务器。


**DGX SuperPOD 中广泛应用InfiniBand**：

下图为 DGX A100/H100 256 SuperPOD 网络拓扑图：

![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/fe23e909b133481a9255d546cbd8992d~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1570&h=552&s=627571&e=png&b=fdfbfb)

下图为 DGX A100/H100 1K POD 网络拓扑图：

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/375e5847ec8040f0a426078c216c352d~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1640&h=488&s=388989&e=png&b=fdfcfc)


## InfiniBand 在 AI 框架中的应用

在之前文章（[AI 集群基础设施 NVMe SSD 详解](https://juejin.cn/post/7311604023184162835)）中谈到了 NVMe 在 DeepSpeed 中的应用。DeepSpeed 通过 ZeRO-Infinity 技术尝试**利用 NVMe 的空间进一步打破内存墙的限制训练超大模型**。除此之外，该方法也充分利用了InfiniBand网络进行多机通信，具体如下图所示。

![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/cc71d99bd8a44f8b8f86e60909ee38a7~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1280&h=716&s=3672639&e=png&b=fbfbfb)

另外，像NCCL、Gloo等集合通信库都继承了InfiniBand，Pytorch框架也能够通过通信库轻松应用InfiniBand进行分布式训练。


## 总结

本文讲述了AI集群通信的软硬件；同时，针对NCLL集合通信库以及InfiniBand网络通信技术协议进行了更为详细的介绍；另外，也讲述了AI集群以及AI框架中对于InfiniBand的应用。

码字不易，如果觉得有帮助，欢迎点赞收藏加关注。


## 参考文档

- [带你了解PCIE通信原理](https://zhuanlan.zhihu.com/p/454282470)
- [电脑硬件冷知识：主板北桥芯片为何消失了，南桥也有同样的命运？](https://zhuanlan.zhihu.com/p/662904805)
- [必看: 原来PCIe技术原理这么简单](https://mp.weixin.qq.com/s/FlRc2q8r0fUOzxJFWulGfw)
- [AI网络互联，PCIe还是NVLink？](https://www.sdnlab.com/26316.html)
- [RDMA技术原理分析、主流实现对比和解析](https://www.sohu.com/a/229080366_632967)
- [详谈RDMA技术原理和三种实现方式](https://mp.weixin.qq.com/s/FgKjDjZsPlweVJ03OVr3SA)
- [RDMA技术详解——RDMA的三种实现方式](https://blog.csdn.net/u013253075/article/details/119843611)
- [【英伟达官网】线缆和收发器](https://www.nvidia.cn/networking/interconnect/)
- [DAC高速线缆和AOC有源光缆有什么区别呢？](http://www.rhopto.com/articles/dacgsx.html)
- [你会区分光模块和光纤收发器吗？](https://www.etulink.com/blog/-_b268)
- [都是光电转换作用，光模块和光纤收发器有什么区别？](https://zhuanlan.zhihu.com/p/139294038)
- [态路小课堂丨关于InfiniBand网络相关内容简介！](https://baijiahao.baidu.com/s?id=1760941961023057651&wfr=spider&for=pc)
- [态路小课堂丨InfiniBand AOC有源光缆简介](https://blog.51cto.com/u_14408894/8031135)
- [InfiniBand，到底是个啥？](https://mp.weixin.qq.com/s?__biz=MzI1NTA0MDUyMA==&mid=2456692454&idx=1&sn=031a11b931edee5504b15045cd863d37&chksm=fda68b81cad10297e4dd53bc97f63e0c47c26a27cdbb3c584cce6fc49fc6b4367b1531cbfcb6&scene=0&xtrack=1#rd)
- [NVIDIA MLNX_OFED Documentation v5.8-3.0.7.0.101 for DGX H100 Systems](https://docs.nvidia.com/networking/display/mlnxofedv583070101/introduction)
- [NCCL 环境变量](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [大模型训练场景下NCCL环境变量设置](https://zhuanlan.zhihu.com/p/653001915)
- [【GitHub】CUDA_GPU详细介绍](https://github.com/FelixFu520/README/blob/main/envs/pytorch/cuda_gpu.md)
- [GPU卡的底层通信原理](https://www.jianshu.com/p/e40059d5c832)
- [NVIDIA ConnectX InfiniBand 网卡](https://www.nvidia.cn/networking/infiniband-adapters/)
- [H3C IB网卡常用命令](https://www.h3c.com/cn/d_202007/1317229_30005_0.htm#_Toc46935211)
- [IB常用命令](https://blog.csdn.net/weixin_42319496/article/details/125942763)
- [浅析GPU通信技术（上）-GPUDirect P2P](https://developer.aliyun.com/article/591403)
- [浅析GPU通信技术（下）-GPUDirect RDMA](https://developer.aliyun.com/article/603617)
- [GPUDirect RDMA 12.3 文档](https://docs.nvidia.com/cuda/gpudirect-rdma/index.html)
- [【大模型训练】RDMA高速网络与集合通讯](https://zhuanlan.zhihu.com/p/622853211)
- [百度智能云-NCCL环境搭建](https://cloud.baidu.com/doc/GPU/s/Yl3mr0ren)





