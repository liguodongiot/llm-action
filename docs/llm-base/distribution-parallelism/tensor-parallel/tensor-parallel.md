



我们以一个线性层为例，它包括一个通用矩阵乘法(GEMM)：$Y=XA$。 给定2个处理器，我们把列 A 划分为 $[A1 A2]$, 并在每个处理器上计算 $Y_i=XA_i$ ， 然后，形成 $[Y_1 Y_2]=[XA_1 XA_2]$。 这被称为列并行方式。



当第二个线性层 $Z=YB$ 跟随上述列并行层的时候，我们把 B 划分为
$
  \begin{bmatrix}
   B1 \\\
   B2 
  \end{bmatrix}
$，这就是所谓的行并行方式。


为了计算 $
 Z = \begin{bmatrix}
   Y1 & Y2
  \end{bmatrix} 
  \begin{bmatrix}
   B1 \\\
   B2 
  \end{bmatrix}
$，我们首先在每个处理器上计算$Y_iB_i$，然后使用一个all-reduce操作将结果汇总为 $Z=Y_1B_1+Y_2B_2$。


![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/15955f8b9f7c4c139f6f82c98eeb357d~tplv-k3u1fbpfcp-watermark.image?)

需要注意，在后向计算中，列并行线性层需要聚合输入张量 X, 因为在每个处理器 i 上，我们只有 $\dot{X_i}=\dot{Y_i} A_i^T$，其中，$\dot{X_i}和\dot{Y_i}$为一阶导数，因此，我们在各处理器之间进行all-reduce，得到 $\dot{X}=\dot{Y}A^T=\dot{Y_1}A_1^T+\dot{Y_2}A_2^T$。





参考：
- 图解大模型训练之：张量模型并行Megatron-LM ：https://zhuanlan.zhihu.com/p/622212228
- Megatron论文和代码详细分析：https://zhuanlan.zhihu.com/p/366906920
- [源码解析]模型并行分布式训练Megatron ： https://juejin.cn/post/7057837676430360584
- 张量模型并行详解 | 深度学习分布式训练专题 ：https://www.paddlepaddle.org.cn/support/news?action=detail&id=2913



