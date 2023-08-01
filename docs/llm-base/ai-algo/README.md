





## 模型对比


| 模型    | GPT2 Medium（345M）          | Bloom-7b1 | LLaMA-7B      | LLaMA2-7B       |
| ---- | ---------- | ----- | ------------------- | ----------- |
| 词表大小（vocab_size）  | 50257     |  250880 |  32000 |   32000    |
| Transformer层（n_layer, num_layers, num_hidden_layers）  | 24     | 30  | 32  |    32   |
| 注意力头数（num_attention_heads, n_head） | 16      | 32  | 32 |   32       |
| 隐藏层大小（hidden_size）  | 1024(n_embd)      |  4096 | 4096  |     4096      |
| 前馈神经网络的隐藏层大小（ffn_hidden_size, intermediate_size）      | N/A   | 4 * hidden_size    | 11008 |   11008       |
| seq_length, n_ctx      | 1024      | 2048  | N/A    |   N/A          |
| n_positions,max_position_embeddings,n_embed      | 1024      |  4096  | N/A |   4096       |




