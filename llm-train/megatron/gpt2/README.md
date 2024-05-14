
## GPT2 模型训练 



## 环境
```
docker run -dt --name nvidia_pytorch_2304_temp --restart=always --gpus all \
--network=host \
--shm-size 4G \
-v /home/gdong/workspace:/workspace \
-w /workspace \
nvcr.io/nvidia/pytorch:23.04-py3 \
/bin/bash

docker exec -it nvidia_pytorch_2304_temp bash
```

---

```
pip freeze > requirements.txt
```



## 代码

```
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout 992da75
```

修改代码：
- megatron/tokenizer/file_utils.py
- tools/openwebtext/merge_data.py

脚本：
- pretrain_gpt.sh：单机
- pretrain_gpt_distributed.sh：数据并行
- pretrain_gpt_distributed_with_mp.sh：模型并行+数据并行



## 权重

```
> tree -h megatron
megatron
├── [   8]  latest_checkpointed_iteration.txt
└── [4.0K]  release
    └── [4.0K]  mp_rank_00
        └── [677M]  model_optim_rng.pt

2 directories, 2 files
> cat megatron/latest_checkpointed_iteration.txt 
release
```

## [数据预处理](https://github.com/liguodongiot/llm-action/blob/main/train/megatron/gpt2/gpt-data-preprocess.md)


## [模型训练](https://github.com/liguodongiot/llm-action/blob/main/train/megatron/gpt2/model_train.md)


## [模型评估及推理](https://github.com/liguodongiot/llm-action/blob/main/train/megatron/gpt2/model_merge_eval_inference.md)


