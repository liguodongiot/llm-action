



DP 将批次（global batch size）拆分为小批次（mini-batch）。PP 将一个小批次切分为多个块 (chunks)，因此，PP 引入了微批次(micro-batch，MBS) 的概念。

计算 DP + PP 设置的全局批量大小的公式为: `mbs*chunks*dp_degree` ， 比如：DP并行度为4，微批次大小为8，块为32，则全局批次大小为：`8*32*4=1024`。



