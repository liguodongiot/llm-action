
# HelloDeepSpeed


- 源码：https://github.com/microsoft/DeepSpeedExamples/tree/master/training/HelloDeepSpeed


## 

```
python train_bert.py --checkpoint_dir ./experiments --local_rank 0
```


```
tree experiments/
experiments/
└── bert_pretrain.2023.6.13.5.34.39.addjtvxg
    ├── checkpoint.iter_1000.pt
    ├── checkpoint.iter_2000.pt
    ├── checkpoint.iter_3000.pt
    ├── checkpoint.iter_4000.pt
    ├── checkpoint.iter_5000.pt
    ├── checkpoint.iter_6000.pt
    ├── checkpoint.iter_7000.pt
    ├── checkpoint.iter_8000.pt
    ├── checkpoint.iter_9000.pt
    ├── gitdiff.log
    ├── githash.log
    ├── hparams.json
    └── tb_dir
        └── events.out.tfevents.1686659679.ai-app-2-46-msxf.54673.0

```


## Deepspeed

```
# 默认使用当前服务器所有GPU卡
deepspeed train_bert_ds.py --checkpoint_dir ./experiments_ds
```

```

```











