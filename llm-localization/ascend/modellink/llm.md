
```
git clone https://gitee.com/ascend/ModelLink.git
cd ModelLink
mkdir logs
mkdir dataset
mkdir ckpt


git clone https://gitee.com/ascend/AscendSpeed.git
cd AscendSpeed
pip install -r requirements.txt 
pip3 install -e .
cd ..

# 安装其余依赖库
pip install -r requirements.txt 
```



```

# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
   
python tools/checkpoint/convert_ckpt.py \
    --model-type GPT \
    --loader llama2_hf \
    --saver megatron \
    --target-tensor-parallel-size 2 \
    --load-dir /home/ma-user/work/aicc/model_from_hf/Baichuan2-7B-Chat/ \
    --save-dir /home/ma-user/work/aicc/model_weights/Baichuan2-7B-v0.1-tp8-pp1/ \
    --tokenizer-model /home/ma-user/work/aicc/model_from_hf/Baichuan2-7B/tokenizer.model \
    --params-dtype bf16 \
    --w-pack True   

```




