



修改共享内存：https://www.alibabacloud.com/help/zh/eci/user-guide/mount-an-emptydir-volume-to-modify-the-shm-size-of-a-pod?spm=a2c63.p38356.0.0.1c055267llapbW

swap（交换内存）和shm（共享内存）的区别：https://blog.csdn.net/songyu0120/article/details/89169987

- tmpfs使用内存空间而swap使用物理存储空间


训练过程内存碎片化问题

- export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'
- 一文读懂 PyTorch 显存管理机制：https://zhuanlan.zhihu.com/p/486360176


RuntimeError: CUDA out of memory.一些调bug路程：https://zhuanlan.zhihu.com/p/581606031



max_split_size_mb 阻止原生（native）分配器分割大于此大小（MB）的块。这可以减少碎片，并允许某些边缘工作负载在内存不耗尽的情况下完成。性能代价从 "零 "到 "大量 "不等，取决于分配模式。
默认值没有限制，即所有块都可以分割。memory_stats()和 memory_summary()方法可用于调整。如果工作负载因 "内存不足 "而终止，并显示大量未活动的分割块，则应在万不得已时使用该选项。 max_split_size_mb 只对 backend:native 有意义。在使用 backend:cudaMallocAsync 时，max_split_size_mb 将被忽略。

- 官方：https://pytorch.org/docs/stable/notes/cuda.html






