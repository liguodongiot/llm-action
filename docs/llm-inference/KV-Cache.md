


最后需要注意当sequence特别长的时候，KV Cache其实还是个Memory刺客。

比如batch_size=32, head=32, layer=32, dim_size=4096, seq_length=2048, float32类型，则需要占用的显存为（感谢网友指正） 2 * 32 * 4096 * 2048 * 32 * 4 / 1024/1024/1024 /1024 = 64G。


