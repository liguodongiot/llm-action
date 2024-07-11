

引擎启动参数：

https://docs.vllm.ai/en/stable/models/engine_args.html



max-num-seqs：默认 256，

当 max-num-seqs 比较小时，较迟接收到的 request 会进入 waiting_list，直到前面有request 结束后再被添加进生成队列。

当 max-num-seqs 太大时，会出现一部分 request 在生成了 3-4 个 tokens 之后，被加入到 waiting_list（有些用户出现生成到一半卡住的情况）。过大或过小的 max-num-seqs 都会影响用户体验。


max-num-batched-tokens：很重要的配置，比如你配置了 max-num-batched-tokens=1000 那么你大概能在一个 batch 里面处理 10 条平均长度约为 100 tokens 的 inputs。max-num-batched-tokens 应尽可能大，来充分发挥 continuous batching 的优势。不过似乎（对于 TGI 是这样，vllm 不太确定），在提供 HF 模型时，该 max-num-batched-tokens 能够被自动推导出来。





