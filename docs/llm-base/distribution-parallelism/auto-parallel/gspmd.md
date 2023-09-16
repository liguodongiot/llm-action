



- GSPMD

- GSPMD:General and Scalable Parallelization for ML Computation Graphs: https://zhuanlan.zhihu.com/p/506026413
- GSPMD: ML计算图的通用可扩展并行化: https://zhuanlan.zhihu.com/p/504670919




## 原paper




GSPMD 是一个用于机器学习计算的高度自动化的并行化系统。 

它提供了一个简单但功能强大的 API，该 API 足够通用，可以组合不同的典型并行模式。 GSPMD 提供直观的自动完成功能，使用户只需注解几个张量即可有效地划分整个模型。 

我们已经证明，GSPMD 能够在多达数千个 Cloud TPUv3 核心上对多个图像、语音和语言模型进行分区，并具有良好且可预测的性能和内存扩展。