







- [预训练中文语料汇总（附数据）](https://zhuanlan.zhihu.com/p/163616279)





好的数据： 书籍 、 维基百科、 代码











## 业界大模型训练数据


### OPT-175B




Meta AI 团队希望在尽可能大的语料库上训练这个模型。 它由以下 5 个经过过滤的文本文档数据集的并集组成：

BookCorpus，由超过 10K 未出版的书籍组成，

CC-Stories，其中包含 CommonCrawl 数据的子集，经过过滤以匹配 Winograd 模式的故事风格，

The Pile，其中包括 Pile-CC、OpenWebText2、USPTO、Project Gutenberg、OpenSubtitles、Wikipedia、DM Mathematics 和 HackerNews。

Baumgartner 等人开发的 Pushshift.io Reddit 数据集。 并由 Roller 等人处理。

CCNewsV2 包含 RoBERTa 中使用的 CommonCrawl 新闻数据集英文部分的更新版本


最终的训练数据包含180B个token，对应800GB的数据。 验证分割由 200MB 的预训练数据组成，根据预训练语料库中每个数据集的大小按比例进行采样。

该数据集可能包含令人反感的内容，因为数据集的一部分是公共 Common Crawl 数据的子集以及公共 Reddit 数据的子集，其中可能包含如果直接查看可能具有侮辱性、威胁性或可能导致焦虑的句子 。






### Bloom-176B



41.5TB 经过大量去重和清洗的文本，包含 46 种语言，最终转换为 350B 个词元


46种自然语言，13种编程语言。


模型的词汇表含 250,680 个词元



























