

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


```
docker start pytorch_ubuntu_upgrade

docker exec -it pytorch_ubuntu_upgrade bash
. /usr/local/Ascend/ascend-toolkit/set_env.sh
conda activate llm-dev 

cd /workspace/llm-train

sh run_all_npu.sh
```

