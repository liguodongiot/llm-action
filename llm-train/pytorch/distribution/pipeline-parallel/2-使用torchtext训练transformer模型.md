





## 定义模型



```
import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # 单个EncoderLayer
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)

        # TransformerEncoder 包含多个 EncoderLayer
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        # 嵌入层
        self.embedding = nn.Embedding(ntoken, d_model)

        self.d_model = d_model
        
        self.linear = nn.Linear(d_model, ntoken)

        # 初始化权重
        self.init_weights()



    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)



    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        return output

```


```python
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

```



## 构造数据


```
import torch
from torch import nn, Tensor
from torch.utils.data import dataset

from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

train_iter = WikiText2(split='train')

tokenizer = get_tokenizer('basic_english')

# 构建词表
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])

# 设置默认索引值，当单词OOV时，使用默认索引
vocab.set_default_index(vocab['<unk>'])

# 0
print(vocab['<unk>'])
print(vocab.get_default_index())

item = "You can now install TorchText using pip!"
# [178, 112, 199, 19230, 0, 438, 0, 385]
print(vocab(tokenizer(item)))

def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
    """Converts raw text into a flat Tensor."""
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


# ``train_iter`` was "consumed" by the process of building the vocab,
# so we have to create it again
train_iter, val_iter, test_iter = WikiText2()

train_data = data_process(train_iter)
val_data = data_process(val_iter)
test_data = data_process(test_iter)

print(train_data.shape, val_data.shape, test_data.shape)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def batchify(data: Tensor, bsz: int) -> Tensor:
    """Divides the data into ``bsz`` separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Arguments:
        data: Tensor, shape ``[N]``
        bsz: int, batch size

    Returns:
        Tensor of shape ``[N // bsz, bsz]``
    """

    seq_len = data.size(0) // bsz
    # 移除无法整除的余数
    data = data[:seq_len * bsz]
    # torch.Size([2049980])
    print(data.shape)
    # t() 转置
    # contiguous()方法首先拷贝了一份张量在内存中的地址，然后将地址按照形状改变后的张量的语义进行排列。
    data = data.view(bsz, seq_len).t().contiguous()
    # torch.Size([102499, 20])
    print(data.shape)
    return data.to(device)


batch_size = 20
eval_batch_size = 10
# torch.Size([102499, 20])
train_data = batchify(train_data, batch_size)  # shape ``[seq_len, batch_size]``
val_data = batchify(val_data, eval_batch_size)
test_data = batchify(test_data, eval_batch_size)
```






### 生成输入-目标序列









