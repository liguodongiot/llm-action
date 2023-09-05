


Transformer中FFN的记忆功能: https://zhuanlan.zhihu.com/p/604739354



大型语言模型的强大能力离不开其对知识的记忆：比如模型想要回答“中国的首都是哪座城市？”，就必须在某种意义上记住“中国的首都是北京”。

Transformer并没有外接显式的数据库，记忆只能隐式地表达在参数当中。


与attention相比不那么引人注意的FFN承担了transformer中记忆的功能。


---
Transformer Feed-Forward Layers Are Key-Value Memories





---

Knowledge Neurons in Pretrained Transformers


梯度大意味着该神经元对输出答案的影响大


定位知识神经元以后就可以对相关神经元进行操作。如下图所示，将知识神经元的激活置0或翻倍可以有效抑制或增强相关知识的表达。在具体操作上，应避开同种关系共用的神经元，以减小对其他事实的影响。

---

Transformer Feed-Forward Layers Are Key-Value Memories一文指出了FFN的记忆作用，

Knowledge Neurons in Pretrained Transformers一文给出了操作知识神经元的应用方式。这些工作对于去除现有语言模型的错误知识，或将新知识注入现有语言模型可能带来帮助。





---

每一层经过attention之后，还会有一个FFN，这个FFN的作用就是空间变换。FFN包含了2层linear transformation层，中间的激活函数是ReLu。



FFN的加入引入了非线性(ReLu激活函数)，变换了attention output的空间, 从而增加了模型的表现能力。把FFN去掉模型也是可以用的，但是效果差了很多。





