





https://epochai.org/blog/backward-forward-FLOP-ratio


如何计算FLOPs
有两种方式：

根据计算公式和模型结构手动推算
借助第三方工具：calflops、ptflops、thop、torchstat、torchsumary、fvcore
手动推导FLOPs原则：
手动推导模型的FLOPs时只推导前向传播，大部分情况默认模型后向传播的计算量是前向传播的2倍， 总共FLOPs是前向的3倍。(结论出自——https://epochai.org/blog/backward-forward-FLOP-ratio)
由于LLM模型参数过大，占用显存过多，有时候为了降低显存在训练采用将中间参数保留在内存里——激活重计算。因此推导LLM训练时FLOPs如果考虑到中间参数的激活重计算的过程，需要计算整体FLOPs需要再加一份前向计算量，即1(前向） + 2(反向）+ 1(激活重计算）= 4 倍 计算量。 （结论出自——https://arxiv.org/pdf/2205.05198.pdf）
手动推导模型的FLOPs时，优先推导整个过程计算量占大头部分，通常忽略激活函数、layer normalize，softmax等等部分计算量。


参考最简单的计算模型(LLM)FLOPs的方法: https://zhuanlan.zhihu.com/p/652697200