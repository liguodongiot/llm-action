






dist.init_process_group()是PyTorch中用于初始化分布式训练的函数之一。

它用于设置并行训练环境，连接多个进程以进行数据和模型的分布式处理。我们通过init_process_group()函数这个方法来进行初始化，


其参数包括以下内容

backend（必需参数）：指定分布式后端的类型，可以是以下选项之一：

‘tcp’：使用TCP协议进行通信。
‘gloo’：使用Gloo库进行通信。
‘mpi’：使用MPI（Message Passing Interface）进行通信。
‘nccl’：使用NCCL库进行通信（适用于多GPU的分布式训练）。
‘hccl’：使用HCCL库进行通信（适用于华为昇腾AI处理器的分布式训练）。

init_method（可选参数）：指定用于初始化分布式环境的方法。它可以是以下选项之一：
‘env://’：使用环境变量中指定的方法进行初始化。
‘file:// ’：使用本地文件进行初始化。
‘tcp://:’：使用TCP地址和端口进行初始化。
‘gloo://:’：使用Gloo地址和端口进行初始化。
‘mpi://:’：使用MPI地址和端口进行初始化。

rank（可选参数）：指定当前进程的排名（从0开始）。
world_size（可选参数）：指定总共使用的进程数。
timeout（可选参数）：指定初始化的超时时间。
group_name（可选参数）：指定用于连接的进程组名称。



这里由于服务器采用的slurm系统，我们开始计划使用mpi去实现分布式分发，同时torch的初始化也支持mpi，原始想法是通过mpirun来进行分布式计算。但是，如果要使用mpi来实现分布式功能，必须要通过github上的源代码进行编译，通过conda和pip进行下载的pytorch自身是不携带mpi的
通过上面的参数，可以看到backend是有多种通信方式的，常用的有gloo和mpi和nccl，但是这三者是有区别的：


对于分布式 GPU 训练，使用 NCCL 后端。
对于分布式 CPU 训练，使用 Gloo 后端。
如果你的主机是 GPU 主机并且具有 InfiniBand 互连： 使用 NCCL，因为它是目前唯一支持 InfiniBand 和
GPUDirect 的后端。
如果你的主机是 GPU 主机并且具有以太网互连： 使用 NCCL，因为它目前提供了最好的分布式 GPU
训练性能，特别是对于多进程单节点或多节点分布式训练。
如果你遇到 NCCL 的任何问题，使用 Gloo 作为备选选项。（注意，Gloo 目前运行速度比 NCCL 慢）
如果你的主机是 CPU 主机并且具有 InfiniBand 互连： 如果你的 InfiniBand 启用了 IP over IB，使用
Gloo，否则，使用 MPI。我们计划在即将发布的版本中为 Gloo 添加 InfiniBand 支持。
如果你的主机是 CPU 主机并且具有以太网互连： 使用 Gloo，除非你有特定的理由使用 MPI​。




我们可以根据文档的提示，得出，MPI是最不推荐使用的一种方法，对于英伟达的显卡，最优先的还是使用NCCL方法。

和Mpi相匹配的有一种torch官方自带的方法，在torch2.0之前使用的API叫：torch.distributed.launch在使用时显示未来的版本将会弃用这个API，取而代之的是torchrun。因此我们将命令由mpi改为torchrun方法，在dist初始化使用nccl后端通信。




假设我们有三个节点，node02，node03，node04，每个节点上有四张GPU。现在我们将官方测试文档中的代码写为test_mpi.py。最终通过torchrun实现的命令如下：


torchrun --nproc_per_node=4 --nnodes=3 --node_rank=0 --master_addr=192.168.0.101 --master_port=29500 test_mpi.py


我们没有必要和torchrun的官方文档一样去设置**–rdzv-backend** 和**–rdzv-id**，因为这不是必须的，用默认的即可。我们只需要设置的参数只有上面这几个。具体参数介绍如下：
–nproc_per_node=4：指定每个节点（机器）上的进程数，这里是4个。意味着每个机器将启动4个进程来参与分布式训练。

–nnodes=3：指定总共的节点数，这里是3个。意味着总共有3个机器参与分布式训练。
–node_rank=0：指定当前节点（机器）的排名，这里是0。排名从0开始，用于在分布式环境中区分不同的节点。
–master_addr=192.168.0.101：指定主节点的IP地址，这里是192.168.0.101。主节点用于协调分布式训练过程。
–master_port=29500：指定主节点的端口号，这里是29500。主节点使用指定的端口来与其他节点进行通信。
通过设置这些参数，该命令将在3个节点的分布式环境中启动4个进程，并指定192.168.0.101作为主节点进行协调和通信。
这里的主节点地址我随便写的，可以根据实际情况进行修改。主节点的地址的- --node_rank必须设置为0，也就是上述这行命令，必须要先在主节点上线运行。

举个例子，假如我的主节点是node02，那么我就要先在node02节点的终端上运行上述torchrun命令，同时–master_addr要为node02的ip地址（查看IP地址可以通过：ip addr），然后node03，node04的顺序就不重要了，在其节点的终端上将–node_rank=0改为–node_rank=1和–node_rank=2运行即可。





