



NVIDIA性能分析工具Nsight Systems/Compute 的使用介绍: https://www.bilibili.com/video/BV15P4y1R7VG （简介及分析案例）



---

checklist：可以帮助快速判断模型存在的问题：

数据加载阶段：

小文件是否太多，导致文件 io 耗时太长，读取会浪费很多时间在寻道上。

存储介质是否已达到瓶颈，可以监控存储介质的繁忙度，如果达到瓶颈可以增加存储介质缓解读取性能；

是否启用多进程并行读取数据，另外可以注意线程争用问题，监控线程等待时候是否过长，可以采用私有线程池进行环境；

是否启用提前加载机制来实现 CPU 和 GPU 的并行；

数据预处理阶段：

是否设置开启共享内存 pin_memory，可以直接将数据放置在pin_memory中；

优化 I/O 和网络操作，确保数据以与其计算相匹配的速率馈送到 GPU；

如果是A100或以上的机器，可以考虑开启numa绑定，缓解争用，提升性能；

模型训练阶段：

是否存在大量的CPU运算，可以通过实现GPU实现或去除指定CPU设备，尽可能的让模型运行在GPU上；

模型是否存在GPU利用率不均的情况，尽可能得不在代码里指定GPU运行的卡；

对于比较复杂运行效率较低的模块，可以通过实现融合的大GPU算子提升训练速度；

避免指标和日志打印太频繁，CPU 和 GPU 频繁切换导致 GPU 利用率低；

是否开启AMP来提升模型的训练性能；

使用最新的高性能库和 GPU 驱动程序，cuda是否升级到最新版本；

---



优化策略

1. 减少不必要的同步:

尽量减少显式的同步调用，如 cudaDeviceSynchronize。

使用 cudaStreamWaitEvent 等事件机制来实现更细粒度的同步控制。

2. 使用多个流:

将独立的CUDA操作分配到不同的流中，以实现并行执行。

确保内核启动和内存拷贝操作尽可能在不同流中并行执行。

3. 优化内存拷贝:

使用异步内存拷贝函数（如 cudaMemcpyAsync ）并将其分配到不同的流中。尽量减少Host与Device之间的内存拷贝次数，使用统一内存（Unified Memory）或零拷贝（Zero Copy）技术。


---





export CUDA_VISIBLE_DEVICES=2

nsys profile -w true \
-t cuda,nvtx,osrt,cudnn,cublas \
-s cpu \
--capture-range=cudaProfilerApi \
--capture-range-end=stop \
--cudabacktrace=true -x true --force-overwrite true -o nsys-sglang-046-14 python3  qwen25_sys.py
