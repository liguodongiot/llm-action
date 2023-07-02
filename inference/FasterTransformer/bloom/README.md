



## 统计性能指标

推理耗时，平均每个token生成时长等。

单卡：
```
CUDA_VISIBLE_DEVICES=1 python examples/pytorch/gpt/firefly_lambada_dianxiao_1w_stat_token.py \
--checkpoint-path /workspace/model/firefly-2b6-dx-1tp/belle7b/1/1-gpu \
--tokenizer-path /workspace/model/firefly-2b6-dx \
--dataset-path /workspace/data/lambada_test.jsonl \
--lib-path  /workspace/lib/libth_transformer.so \
--inference-data-type fp16 --show-progress --input-token-len 64 --output-token-len 256 \
--dianxiao-path-stat /workspace/output/firefly_random_sample_1w_256_stat_ft.json
```

双卡张量并行：
```
CUDA_VISIBLE_DEVICES=2,3  mpirun -n 2 python examples/pytorch/gpt/firefly_lambada_dianxiao_1w_stat_token.py \
--checkpoint-path /workspace/model/firefly-2b6-dx-2tp/belle7b/1/2-gpu \
--tokenizer-path /workspace/model/firefly-2b6-dx \
--dataset-path /workspace/data/lambada_test.jsonl \
--lib-path  /workspace/lib/libth_transformer.so \
--inference-data-type fp16 \
--tensor-para-size 2 \
--pipeline-para-size 1 \
--show-progress \
--input-token-len 64 \
--output-token-len 256 \
--dianxiao-path-stat  /workspace/output/firefly_random_sample_1w_256_stat_ft_tp2.json
```

