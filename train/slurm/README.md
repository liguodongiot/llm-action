


## pytorch



### 单机多卡
```

```


### 多机多卡


```

```


## deepspeed


### 单机多卡

```
deepspeed --include localhost:0,1,2,3 train.py --deepspeed_config=ds_config.json -p 2 --steps=200
```


### 多机多卡

```
python -m torch.distributed.run --nproc_per_node=2 --nnode=2 --node_rank=0 --master_addr=10.99.2.xx \
--master_port=9901 train.py --deepspeed_config=ds_config.json -p 2 --steps=200


python -m torch.distributed.run --nproc_per_node=2 --nnode=2 --node_rank=1 --master_addr=10.99.2.xx \
--master_port=9901 train.py --deepspeed_config=ds_config.json -p 2 --steps=200
```
