



```
python tools/checkpoint_util.py \
        --model-type GPT \
        --load-dir /workspace/model/megatron-models/345m-init-mp\
        --save-dir /workspace/model/megatron-models/345m-init-mp-out \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 1 \
        --vocab-file /workspace/model/gpt2-vocab/gpt2-vocab.json 
```
