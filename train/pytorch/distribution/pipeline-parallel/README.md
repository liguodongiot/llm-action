

- Pipeline Parallelism：https://pytorch.org/docs/stable/pipeline.html
- Sequence-to-Sequence Modeling with nn.Transformer and TorchText：https://pytorch.org/tutorials/intermediate/pipeline_tutorial.html
- TRAINING TRANSFORMER MODELS USING PIPELINE PARALLELISM：https://pytorch.org/tutorials/beginner/transformer_tutorial.html
- TRAINING TRANSFORMER MODELS USING DISTRIBUTED DATA PARALLEL AND PIPELINE PARALLELISM
：https://pytorch.org/tutorials/advanced/ddp_pipeline.html









---


- SINGLE-MACHINE MODEL PARALLEL BEST PRACTICES：https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html





## 使用DDP与PP训练Transformer模型


```
python ddp_pipeline.py
---------
---------
partition_len: 4 rank: 0
partition_len: 4 rank: 1
[RANK 1]: Total parameters in model: 1,061,924,974
[RANK 0]: Total parameters in model: 1,061,924,974
[W logger.cpp:317] Warning: Cuda time stats are not collected for multi-device modules. (function operator())
[W logger.cpp:317] Warning: Cuda time stats are not collected for multi-device modules. (function operator())
[RANK 1]: | epoch   1 |    10/   50 batches | lr 5.00 | ms/batch 517.85 | loss 45.48 | ppl 56432121366884458496.00
[RANK 0]: | epoch   1 |    10/   50 batches | lr 5.00 | ms/batch 517.64 | loss 44.56 | ppl 22504870874679681024.00
[RANK 1]: | epoch   1 |    20/   50 batches | lr 5.00 | ms/batch 385.36 | loss 46.63 | ppl 178457758192923508736.00
[RANK 0]: | epoch   1 |    20/   50 batches | lr 5.00 | ms/batch 385.36 | loss 45.77 | ppl 75770339191959027712.00
[RANK 0]: | epoch   1 |    30/   50 batches | lr 5.00 | ms/batch 385.49 | loss 41.88 | ppl 1546590899039937792.00
[RANK 1]: | epoch   1 |    30/   50 batches | lr 5.00 | ms/batch 385.49 | loss 42.50 | ppl 2856464247237597696.00
[RANK 1]: | epoch   1 |    40/   50 batches | lr 5.00 | ms/batch 386.09 | loss 38.95 | ppl 82371305929986336.00
[RANK 0]: | epoch   1 |    40/   50 batches | lr 5.00 | ms/batch 386.09 | loss 39.99 | ppl 233730045251711744.00
[RANK 0]: -----------------------------------------------------------------------------------------
[RANK 0]: | end of epoch   1 | time: 22.57s | valid loss  0.91 | valid ppl     2.48
[RANK 0]: -----------------------------------------------------------------------------------------
[RANK 1]: -----------------------------------------------------------------------------------------
[RANK 1]: | end of epoch   1 | time: 22.59s | valid loss  0.91 | valid ppl     2.48
[RANK 1]: -----------------------------------------------------------------------------------------
[RANK 1]: | epoch   2 |    10/   50 batches | lr 4.75 | ms/batch 425.29 | loss 39.88 | ppl 209605830679931392.00
[RANK 0]: | epoch   2 |    10/   50 batches | lr 4.75 | ms/batch 427.78 | loss 39.88 | ppl 208211260703358496.00
[RANK 0]: | epoch   2 |    20/   50 batches | lr 4.75 | ms/batch 386.57 | loss 28.43 | ppl 2212304159830.81
[RANK 1]: | epoch   2 |    20/   50 batches | lr 4.75 | ms/batch 386.57 | loss 28.97 | ppl 3807579905494.86
[RANK 0]: | epoch   2 |    30/   50 batches | lr 4.75 | ms/batch 387.16 | loss 26.35 | ppl 278391001398.60
[RANK 1]: | epoch   2 |    30/   50 batches | lr 4.75 | ms/batch 387.17 | loss 25.44 | ppl 111852774843.71
[RANK 1]: | epoch   2 |    40/   50 batches | lr 4.75 | ms/batch 387.81 | loss 19.52 | ppl 298987362.70
[RANK 0]: | epoch   2 |    40/   50 batches | lr 4.75 | ms/batch 387.82 | loss 19.41 | ppl 269058605.29
[RANK 0]: -----------------------------------------------------------------------------------------
[RANK 0]: | end of epoch   2 | time: 21.62s | valid loss  0.23 | valid ppl     1.26
[RANK 0]: -----------------------------------------------------------------------------------------
[RANK 1]: -----------------------------------------------------------------------------------------
[RANK 1]: | end of epoch   2 | time: 21.67s | valid loss  0.23 | valid ppl     1.26
[RANK 1]: -----------------------------------------------------------------------------------------
[RANK 0]: | epoch   3 |    10/   50 batches | lr 4.51 | ms/batch 433.81 | loss 11.59 | ppl 108360.86
[RANK 1]: | epoch   3 |    10/   50 batches | lr 4.51 | ms/batch 426.98 | loss 11.59 | ppl 107512.77
[RANK 0]: | epoch   3 |    20/   50 batches | lr 4.51 | ms/batch 386.89 | loss 10.04 | ppl 22813.06
[RANK 1]: | epoch   3 |    20/   50 batches | lr 4.51 | ms/batch 386.89 | loss  9.99 | ppl 21883.85
[RANK 0]: | epoch   3 |    30/   50 batches | lr 4.51 | ms/batch 387.63 | loss 10.69 | ppl 43941.38
[RANK 1]: | epoch   3 |    30/   50 batches | lr 4.51 | ms/batch 387.63 | loss 10.55 | ppl 38258.17
[RANK 1]: | epoch   3 |    40/   50 batches | lr 4.51 | ms/batch 387.79 | loss  9.80 | ppl 18089.39
[RANK 0]: | epoch   3 |    40/   50 batches | lr 4.51 | ms/batch 387.79 | loss  9.80 | ppl 18088.98
[RANK 0]: -----------------------------------------------------------------------------------------
[RANK 0]: | end of epoch   3 | time: 21.76s | valid loss  0.31 | valid ppl     1.37
[RANK 0]: -----------------------------------------------------------------------------------------
[RANK 1]: -----------------------------------------------------------------------------------------
[RANK 1]: | end of epoch   3 | time: 21.83s | valid loss  0.31 | valid ppl     1.37
[RANK 1]: -----------------------------------------------------------------------------------------
[RANK 0]: =========================================================================================
[RANK 0]: | End of training | test loss  0.27 | test ppl     1.31
[RANK 0]: =========================================================================================
[RANK 1]: =========================================================================================
[RANK 1]: | End of training | test loss  0.27 | test ppl     1.31
[RANK 1]: =========================================================================================

```


显存占用：
```
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A     29461      C   ...nv-py310-cu117/bin/python    11612MiB |
|    0   N/A  N/A     29462      C   ...nv-py310-cu117/bin/python      556MiB |
|    1   N/A  N/A     29461      C   ...nv-py310-cu117/bin/python    11748MiB |
|    2   N/A  N/A     29462      C   ...nv-py310-cu117/bin/python    11354MiB |
|    3   N/A  N/A     29462      C   ...nv-py310-cu117/bin/python    11748MiB |
+-----------------------------------------------------------------------------+
```

