



- 代码：https://github.com/jiaweizzhao/GaLore
- 论文：https://arxiv.org/abs/2403.03507

- GaLore：通过梯度低秩投影进行内存高效的 LLM 训练


```
git clone git@github.com:jiaweizzhao/GaLore.git
cd GaLore
git checkout a6bc1650
```



https://download.pytorch.org/whl/cu118/torch-2.1.0%2Bcu118-cp310-cp310-linux_x86_64.whl#sha256=a81b554184492005543ddc32e96469f9369d778dedd195d73bda9bed407d6589



使用单个 GPU（例如：NVIDIA RTX 4090）训练 7B 模型，仅需指定 --optimizer=galore_adamw8bit_per_layer 即可，这会启用 GaLoreAdamW8bit 并进行每层权重更新。通过激活检查点，在 NVIDIA RTX 4090 上批量大小可以到达 16。


```
# LLaMA-7B, 8-bit GaLore-Adam, single GPU, activation checkpointing
# bsz=16, 22.8G, 
torchrun --standalone --nproc_per_node 1 torchrun_main.py \
    --model_config configs/llama_7b.json \
    --lr 0.005 \
    --galore_scale 0.25 \
    --rank 1024 \
    --update_proj_gap 500 \
    --batch_size 16 \
    --total_batch_size 512 \
    --activation_checkpointing \
    --num_training_steps 150000 \
    --warmup_steps 15000 \
    --weight_decay 0 \
    --grad_clipping 1.0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --single_gpu \
    --optimizer galore_adamw8bit_per_layer
```

目前，每层权重更新技术仅支持单 GPU 训练 ( --single_gpu )，而不能使用 nn.parallel.DistributedDataParallel 。

