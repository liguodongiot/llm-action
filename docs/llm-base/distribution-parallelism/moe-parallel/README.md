

- https://github.com/laekov/fastmoe
- SmartMoE: https://github.com/zms1999/SmartMoE





- 飞浆-MOE：https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/06_distributed_training/moe_cn.html
- 
- https://blog.csdn.net/qq_41185868/article/details/103219988

- [GShard-MoE](https://arxiv.org/abs/2006.16668)





GShard，Switch-Transformer， GLaM



- Mixture-of-Experts (MoE) 经典论文一览：https://zhuanlan.zhihu.com/p/542465517


```
GShard，按照文章的说法，是第一个将MoE的思想拓展到Transformer上的工作。
具体的做法是，把Transformer的encoder和decoder中，每隔一个（every other）的FFN层，替换成position-wise 的 MoE层，使用的都是 Top-2 gating network。




跟其他MoE模型的一个显著不同就是，Switch Transformer 的 gating network 每次只 route 到 1 个 expert，而其他的模型都是至少2个。
这样就是最稀疏的MoE了，因此单单从MoE layer的计算效率上讲是最高的了。
```





- Google的 Pathways（理想）与 PaLM（现实）：https://zhuanlan.zhihu.com/p/541281939

```
当前模型的主要问题：

基本都是一个模型做一个任务；
在一个通用的模型上继续fine-tune，会遗忘很多其他知识；
基本都是单模态；
基本都是 dense 模型，在完成一个任务时（不管难易程度），网络的所有参数都被激活和使用；



Pathways 的愿景 —— 一个跟接近人脑的框架：

一个模型，可以做多任务，多模态
sparse model，在做任务时，只是 sparsely activated，只使用一部分的参数
```




- GShard论文笔记（1）-MoE结构：https://zhuanlan.zhihu.com/p/344344373

```
Mixture-of-Experts结构的模型更像是一个智囊团，里面有多个专家，你的问题会分配给最相关的一个或多个专家，综合他们的意见得到最终结果。

为了实现这个结构，显而易见需要两部分：

1）分发器：根据你的问题决定应该问哪些专家
2）一群各有所长的专家：根据分发器分过来的问题做解答
3）（可选）综合器：很多专家如果同事给出了意见，决定如何整合这些意见，这个东西某种程度上和分发器是一样的，其实就是根据问题，给各个专家分配一个权重



左边部分展示了普通的Transformer模型，右边展示了引入MoE结构的Transformer模型：其实就是把原来的FFN（两层全连接）替换成了红框里的MoE结构。不过MoE里面的“专家”依旧是FFN，只是从单个FFN换成了一群FFN，又加了一个分发器（图中的Gating）。分发器的任务是把不同的token分发给不同的专家。




看完了专家和分发器的作用，我们再进一步看看GShard里面他们的具体实现：

对于分发器来说，在训练过程中，最好把token平均分配给各个专家：不然有些专家闲着，有些专家一堆事，会影响训练速度，而且那些整天无所事事的专家肯定最后训练的效果不好。。。因此分发器有一个很重要的任务，就是尽可能把token均分给各个专家。

为了完成这个目标，有一些繁琐的设定：

1）引入了一个loss，专门用来控制我分发器分发的怎么样：如果我把token都分给一个人，loss就很高，分的越均匀（最好是彻底均分），loss越小
2）每个token最多分配给两个专家。如果我每个token哐叽一下发给了所有人，那我多专家有什么意义？（专家之间的差别主要就是训练数据的不同引起的）
3）每个专家每次最多接手C个token。和2类似，如果一个专家成天：“教练，我想打篮球”，“教练，我想唱”，“教练，我想rapper”。。。那估计最后学出来也是四不像


```








