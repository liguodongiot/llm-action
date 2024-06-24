

> Transformers 结构中 Encoder-only，Decoder-only，Encoder-Decoder 划分的具体标准是什么？典型代表模型有哪些？




> BatchNorm与LayerNorm的异同


https://zhuanlan.zhihu.com/p/428620330

代码：

https://zhuanlan.zhihu.com/p/656647661



https://www.zhihu.com/question/395811291/answer/1251829041

选择什么样的归一化方式，取决于你关注数据的哪部分信息。如果某个维度信息的差异性很重要，需要被拟合，那就别在那个维度进行归一化。


关联性  --- 差异性

CV 里面 不同样本的同一channel有关联性， 同一样本的不同channel是有差异性的（没有关联性）
CV 里面 不同样本的同一特性没有关联性（有差异性），同一样本的不同特征有关联性


https://zhuanlan.zhihu.com/p/643560888


> 为什么像 baichuan2 和 llama 使用 RMSNorm 归一化?

RMS是LayerNorm的平替，发表在“Root Mean Square Layer Normalization ”，其提出的动机是LayerNorm运算量比较大，所提出的RMSNorm性能和LayerNorm相当，但是可以节省7%到64%的运算。RMSNorm和LayerNorm的主要区别在于RMSNorm不需要同时计算均值和方差两个统计量，而只需要计算均方根Root Mean Square这一个统计量。

- https://zhuanlan.zhihu.com/p/694909672



> Post-Norm vs. Pre-Norm

同一设置之下，Pre Norm结构往往更容易训练，但最终效果通常不如Post Norm。Pre Norm更容易训练好理解，因为它的恒等路径更突出.

post-norm和pre-norm其实各有优势，post-norm在残差之后做归一化，对参数正则化的效果更强，进而模型的鲁棒性也会更好；

pre-norm相对于post-norm，因为有一部分参数直接加在了后面，不需要对这部分参数进行正则化，正好可以防止模型的梯度爆炸或者梯度消失，因此，这里笔者可以得出的一个结论是如果层数少post-norm的效果其实要好一些，如果要把层数加大，为了保证模型的训练，pre-norm显然更好一些。






