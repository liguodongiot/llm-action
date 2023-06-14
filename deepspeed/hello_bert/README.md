
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

tree experiments_ds/
experiments_ds/
└── bert_pretrain.2023.6.13.18.58.44.addjtvxg
    ├── gitdiff.log
    ├── githash.log
    ├── global_step1000
    │   ├── mp_rank_00_model_states.pt
    │   ├── zero_pp_rank_0_mp_rank_00_optim_states.pt
    ...
    │   └── zero_pp_rank_7_mp_rank_00_optim_states.pt
    ├── global_step9000
    │   ├── mp_rank_00_model_states.pt
    │   ├── zero_pp_rank_0_mp_rank_00_optim_states.pt
    ...
    │   └── zero_pp_rank_7_mp_rank_00_optim_states.pt
    ├── hparams.json
    ├── latest
    ├── tb_dir
    │   └── events.out.tfevents.1686707924.ai-app-2-46-msxf.599.0
    └── zero_to_fp32.py



deepspeed --include localhost:2,3,4,5 train_bert_ds.py --checkpoint_dir ./experiments_multigpu --num_iterations=500 --checkpoint_every=250

tree -h ./experiments_multigpu
./experiments_multigpu
├── [  36]  bert_pretrain.2023.6.13.19.37.59.addjtvxg
│   └── [  63]  global_step250
│       └── [ 47M]  zero_pp_rank_3_mp_rank_00_optim_states.pt
└── [ 169]  bert_pretrain.2023.6.13.19.38.0.addjtvxg
    ├── [ 45K]  gitdiff.log
    ├── [  41]  githash.log
    ├── [ 207]  global_step250
    │   ├── [ 31M]  mp_rank_00_model_states.pt
    │   ├── [ 47M]  zero_pp_rank_0_mp_rank_00_optim_states.pt
    │   ├── [ 47M]  zero_pp_rank_1_mp_rank_00_optim_states.pt
    │   └── [ 47M]  zero_pp_rank_2_mp_rank_00_optim_states.pt
    ├── [ 298]  hparams.json
    ├── [  14]  latest
    ├── [  77]  tb_dir
    │   └── [2.4K]  events.out.tfevents.1686710280.ai-app-2-46-msxf.14672.0
    └── [ 18K]  zero_to_fp32.py
```











