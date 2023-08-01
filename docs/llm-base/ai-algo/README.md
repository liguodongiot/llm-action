





## 模型对比


| 模型    | GPT2 Medium（345M）          | Bloom-7b1 | LLaMA-7B      | LLaMA2-7B       |
| ---- | ---------- | ----- | ------------------- | ----------- |
| 词表大小（vocab_size）  | 50257     |  250880 |  32000 |   32000    |
| Transformer层（n_layer, num_layers, num_hidden_layers）  | 24     | 30  | 32  |    32   |
| 注意力头数（num_attention_heads, n_head） | 16      | 32  | 32 |   32       |
| key_value头数（num_key_value_heads） | N/A       | N/A   | N/A  |   N/A        |
| 隐藏层大小（hidden_size）  | 1024(n_embd)      |  4096(n_embed) | 4096  |     4096      |
| 前馈神经网络的隐藏层大小（ffn_hidden_size, intermediate_size,n_inner） | 4*n_embd  | 4 * hidden_size    | 11008 |   11008       |
| seq_length, n_ctx      | 1024      | 2048  | 2048(max_position_embeddings)    |   2048(max_position_embeddings)         |
| n_positions,max_position_embeddings,n_embed      | 1024(default)      |  2048(4096,bloomz-7b1-hf)  | 2048 |   2048(4096,llama2-chat-hf)       |



- https://huggingface.co/gpt2-medium/resolve/main/config.json
- https://huggingface.co/bigscience/bloom-7b1/blob/main/config.json
- https://huggingface.co/bigscience/bloomz-7b1-mt/blob/main/config.json
- https://huggingface.co/yahma/llama-7b-hf/blob/main/config.json
- https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/blob/main/config.json

说明：
- 通常 seq_length 与 max_position_embeddings 相等。
- key_value头数：This is the number of key_value heads that should be used to implement Grouped Query Attention. If
`num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
`num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
by meanpooling all the original heads within that group.






