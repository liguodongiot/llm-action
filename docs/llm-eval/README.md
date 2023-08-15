
- How to Evaluate a Large Language Model (LLM)?：https://www.analyticsvidhya.com/blog/2023/05/how-to-evaluate-a-large-language-model-llm/



## 评估指标

### 困惑度 perplexity

语言模型的效果好坏的常用评价指标是困惑度(perplexity),在一个测试集上得到的perplexity 越低，说明建模的效果越好。

PPL是用在自然语言处理领域（NLP）中，衡量语言模型好坏的指标。它主要是根据每个词来估计一句话出现的概率，并用句子长度作normalize。

PPL越小越好，PPL越小，p(wi)则越大，也就是说这句话中每个词的概率较高，说明这句话契合的表较好。

- https://blog.csdn.net/hxxjxw/article/details/107722646

