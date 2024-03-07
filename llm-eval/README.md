
- How to Evaluate a Large Language Model (LLM)?：https://www.analyticsvidhya.com/blog/2023/05/how-to-evaluate-a-large-language-model-llm/



## 评估指标

### 困惑度 perplexity

语言模型的效果好坏的常用评价指标是困惑度(perplexity),在一个测试集上得到的perplexity 越低，说明建模的效果越好。

PPL是用在自然语言处理领域（NLP）中，衡量语言模型好坏的指标。它主要是根据每个词来估计一句话出现的概率，并用句子长度作normalize。

PPL越小越好，PPL越小，p(wi)则越大，也就是说这句话中每个词的概率较高，说明这句话契合的表较好。

- https://blog.csdn.net/hxxjxw/article/details/107722646



### Rouge


https://github.com/Isaac-JL-Chen/rouge_chinese




## helm


- https://crfm.stanford.edu/helm/latest/

### 59 metrics

```

# Accuracy
none
Quasi-exact match
F1
Exact match
RR@10
NDCG@10
ROUGE-2
Bits/byte
Exact match (up to specified indicator)
Absolute difference
F1 (set match)
Equivalent
Equivalent (chain of thought)
pass@1


# Calibration
Max prob
1-bin expected calibration error
10-bin expected calibration error
Selective coverage-accuracy area
Accuracy at 10% coverage
1-bin expected calibration error (after Platt scaling)
10-bin Expected Calibration Error (after Platt scaling)
Platt Scaling Coefficient
Platt Scaling Intercept


# Robustness
Quasi-exact match (perturbation: typos)
F1 (perturbation: typos)
Exact match (perturbation: typos)
RR@10 (perturbation: typos)
NDCG@10 (perturbation: typos)
Quasi-exact match (perturbation: synonyms)
F1 (perturbation: synonyms)
Exact match (perturbation: synonyms)
RR@10 (perturbation: synonyms)
NDCG@10 (perturbation: synonyms)


# Fairness
Quasi-exact match (perturbation: dialect)
F1 (perturbation: dialect)
Exact match (perturbation: dialect)
RR@10 (perturbation: dialect)
NDCG@10 (perturbation: dialect)
Quasi-exact match (perturbation: race)
F1 (perturbation: race)
Exact match (perturbation: race)
RR@10 (perturbation: race)
NDCG@10 (perturbation: race)
Quasi-exact match (perturbation: gender)
F1 (perturbation: gender)
Exact match (perturbation: gender)
RR@10 (perturbation: gender)
NDCG@10 (perturbation: gender)
Bias
Stereotypical associations (race, profession)
Stereotypical associations (gender, profession)
Demographic representation (race)
Demographic representation (gender)
Toxicity
Toxic fraction
Efficiency
Observed inference runtime (s)
Idealized inference runtime (s)
Denoised inference runtime (s)
Estimated training emissions (kg CO2)
Estimated training energy cost (MWh)

---
# General information

# eval
# train
truncated
# prompt tokens
# output tokens
# trials

---

# Summarization metrics
SummaC
QAFactEval
BERTScore (F1)
Coverage
Density
Compression
HumanEval-faithfulness
HumanEval-relevance
HumanEval-coherence

# APPS metrics
Avg. # tests passed
Strict correctness


# BBQ metrics
BBQ (ambiguous)
BBQ (unambiguous)


# Copyright metrics
Longest common prefix length
Edit distance (Levenshtein)
Edit similarity (Levenshtein)


# Disinformation metrics
Self-BLEU
Entropy (Monte Carlo)

# Classification metrics
Macro-F1
Micro-F1
```


## lm-evaluation-harness

- https://github.com/EleutherAI/lm-evaluation-harness






## Chatbot Arena

- https://chat.lmsys.org/



### 现有的 LLM 基准框架

尽管存在 HELM 和 lm-evaluation-harness 等基准测试，但由于缺乏成对比较兼容性，它们在评估自由形式问题时存在不足。 

这就是 Chatbot Arena 等众包基准测试平台发挥作用的地方。





## CLEVA

中文语言模型评估平台

- https://github.com/LaVi-Lab/CLEVA
- http://www.lavicleva.com/#/homepage/overview










