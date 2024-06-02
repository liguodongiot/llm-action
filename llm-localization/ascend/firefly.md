

Xformers






```
conda activate llm-dev 
source /usr/local/Ascend/ascend-toolkit/set_env.sh


cd /workspace/llm-train
```


```
docker exec -it pytorch_ubuntu_dev bash
conda activate llm-dev 
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /workspace/llm-train


sh run_all_npu.sh
```



```

sh run_lora_npu.sh
```