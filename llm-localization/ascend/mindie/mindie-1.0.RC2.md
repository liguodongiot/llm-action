


文档：
- https://www.hiascend.com/document/detail/zh/mindie/10RC2/whatismindie/mindie_what_0001.html

docker： 
- https://www.hiascend.com/developer/ascendhub/detail/af85b724a7e5469ebd7ea13c3439d48f



rsync -P --rsh=ssh -r root@192.168.16.211:/root/mindie-1.0.rc2.tar .



swr.cn-south-1.myhuaweicloud.com/ascendhub/mindie:1.0.RC2-800I-A2-aarch64 


```
docker run  -it -d --name mindie-rc2-45 --net=host  \
-e ASCEND_VISIBLE_DEVICES=4,5 \
-p 1925:1025 \
--shm-size=32g \
-w /workspace \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /data/model_from_hf:/workspace/model \
swr.cn-south-1.myhuaweicloud.com/ascendhub/mindie:1.0.RC2-800I-A2-aarch64 \
/bin/bash


docker exec -it mindie-rc2-45 bash



cd /opt/package
# 安装CANN包
source ./install_and_enable_cann.sh



source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/mindie/set_env.sh
source /usr/local/Ascend/llm_model/set_env.sh



vim /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json

/workspace/model/Qwen1.5-7B-Chat/


export MIES_PYTHON_LOG_TO_FILE=1
export MIES_PYTHON_LOG_TO_STDOUT=1
export PYTHONPATH=/usr/local/Ascend/llm_model:$PYTHONPATH
cd /usr/local/Ascend/mindie/latest/mindie-service/bin
./mindieservice_daemon

```


## 新镜像

```
docker commit -a "guodong" -m "mindie-1.0.RC2" 365815a95f16 harbor/ascend/mindie-base:1.0.RC2

# -p 192.168.16.xx:1025:1025

docker run  -it --rm  \
-e ASCEND_VISIBLE_DEVICES=2,3 \
-p 1025:1025 \
--shm-size=32g \
-w /workspace \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /data/model_from_hf:/workspace/model \
harbor/ascend/mindie-base:1.0.RC2 \
/bin/bash

```


```
llm-server3.sh



docker run -it --rm \
-e ASCEND_VISIBLE_DEVICES=6,7 \
-p 1825:1025 \
--env AIE_LLM_CONTINUOUS_BATCHING=1 \
--shm-size=32g \
-w /workspace \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /data/model_from_hf/Qwen1.5-7B-Chat:/workspace/model \
-v /home/workspace/llm-server3.sh:/workspace/llm-server.sh \
-v /home/workspace/mindservice.log:/usr/local/Ascend/mindie/latest/mindie-service/logs/mindservice.log \
harbor/ascend/mindie-base:1.0.RC2 \
/bin/bash





docker run -it --rm \
-e ASCEND_VISIBLE_DEVICES=4,5 \
-p 1525:1025 \
--env AIE_LLM_CONTINUOUS_BATCHING=1 \
--shm-size=32g \
-w /workspace \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /data/model_from_hf/Qwen1.5-7B-Chat:/workspace/model \
-v /home/workspace/llm-server3.sh:/workspace/llm-server.sh \
-v /home/workspace/mindservice.log:/usr/local/Ascend/mindie/latest/mindie-service/logs/mindservice.log \
harbor/ascend/mindie-base:1.0.RC2 \
/workspace/llm-server.sh \
--model_name=qwen-chat \
--model_weight_path=/workspace/model \
--world_size=2 \
--npu_mem_size=15





```



