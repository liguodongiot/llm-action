


## 进入容器内

```
ascendhub.huawei.com/public-ascendhub/mindie:1.0.RC1-800I-A2-aarch64


# commit
docker commit -a "guodong" -m "mindie-service" b7fe01c81fcc ascendhub.huawei.com/public-ascendhub/mindie-service-env:1.0.RC1-800I-A2-aarch64


docker run  -it --rm --net=host --ipc=host \
--shm-size=50g \
--privileged=true \
-w /home \
--device=/dev/davinci4 \
--device=/dev/davinci5 \
--device=/dev/davinci6 \
--device=/dev/davinci7 \
--device=/dev/davinci_manager \
--device=/dev/hisi_hdc \
--device=/dev/devmm_svm \
--entrypoint=bash \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/sbin:/usr/local/sbin \
-v /home:/home \
-v /tmp:/tmp \
-v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
ascendhub.huawei.com/public-ascendhub/mindie-service-env:1.0.RC1-800I-A2-aarch64



docker commit -a "guodong" -m "mindie-service" 45bafed49c5b ascendhub.huawei.com/public-ascendhub/mindie-service-env:v2


docker save -o  mindie-service-env.tar ascendhub.huawei.com/public-ascendhub/mindie-service-env:v2

```


## 启动服务

```

docker run  -it --rm --net=host --ipc=host \
--shm-size=50g \
--privileged=true \
-w /home \
--device=/dev/davinci_manager \
--device=/dev/hisi_hdc \
--device=/dev/devmm_svm \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/sbin:/usr/local/sbin \
-v /home:/home \
-v /tmp:/tmp \
-v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
-v /home/aicc/docker/qwen1.5-14b.json:/usr/local/Ascend/mindie/latest/mindie-service/conf/config.json \
ascendhub.huawei.com/public-ascendhub/mindie-service-online:v1.1

pkill -9 mindieservice_d


docker save -o  mindie.tar ascendhub.huawei.com/public-ascendhub/mindie:1.0.RC1-800I-A2-aarch64

docker save -o  mindie-service-online.tar ascendhub.huawei.com/public-ascendhub/mindie-service-online:v1.0
docker save -o  mindie-service-online-v1.1.tar ascendhub.huawei.com/public-ascendhub/mindie-service-online:v1.1






docker run  -it --rm --net=host --ipc=host \
--shm-size=50g \
--privileged=true \
-w /home \
--device=/dev/davinci_manager \
--device=/dev/hisi_hdc \
--device=/dev/devmm_svm \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/sbin:/usr/local/sbin \
-v /home:/home \
-v /tmp:/tmp \
-v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
-v /home/aicc/docker/qwen-72b.json:/usr/local/Ascend/mindie/latest/mindie-service/conf/config.json \
ascendhub.huawei.com/public-ascendhub/mindie-service-online:v1.1



```

### qwen1.5


```
docker run  -it --rm --net=host --ipc=host \
--shm-size=50g \
--privileged=true \
-w /home \
--device=/dev/davinci_manager \
--device=/dev/hisi_hdc \
--device=/dev/devmm_svm \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/sbin:/usr/local/sbin \
-v /home:/home \
-v /tmp:/tmp \
-v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
-v /home/aicc/docker/qwen1.5-14b.json:/usr/local/Ascend/mindie/latest/mindie-service/conf/config.json \
ascendhub.huawei.com/public-ascendhub/mindie-service-online:v1.1




docker run  -it --rm --net=host --ipc=host \
--shm-size=50g \
--privileged=true \
-w /home \
--device=/dev/davinci_manager \
--device=/dev/hisi_hdc \
--device=/dev/devmm_svm \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/sbin:/usr/local/sbin \
-v /home:/home \
-v /tmp:/tmp \
-v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
-v /home/aicc/docker/qwen1.5-72b.json:/usr/local/Ascend/mindie/latest/mindie-service/conf/config.json \
ascendhub.huawei.com/public-ascendhub/mindie-service-online:v1.1

```



```

docker run  -it --rm --net=host --ipc=host \
-e ASCEND_VISIBLE_DEVICES=0,1 \
--shm-size=50g \
--privileged=true \
-w /home \
--device=/dev/davinci_manager \
--device=/dev/hisi_hdc \
--device=/dev/devmm_svm \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/sbin:/usr/local/sbin \
-v /home:/home \
-v /tmp:/tmp \
-v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
-v /home/aicc/docker/qwen1.5-7b-4tp.json:/usr/local/Ascend/mindie/latest/mindie-service/conf/config.json \
ascendhub.huawei.com/public-ascendhub/mindie-service-online:v1.1

```


```
docker run  -it --rm --net=host --ipc=host \
-e ASCEND_VISIBLE_DEVICES=0 \
--shm-size=50g \
--privileged=true \
-w /home \
--device=/dev/davinci_manager \
--device=/dev/hisi_hdc \
--device=/dev/devmm_svm \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/sbin:/usr/local/sbin \
-v /home:/home \
-v /tmp:/tmp \
-v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
-v /home/aicc/docker/qwen1.5-7b-1tp.json:/usr/local/Ascend/mindie/latest/mindie-service/conf/config.json \
ascendhub.huawei.com/public-ascendhub/mindie-service-online:v1.1

```

```
nohup python performance.py  > qwen1.5-7b-2tp.log 2>&1 &
nohup python performance.py  > qwen1.5-7b-4tp.log 2>&1 &
nohup python performance.py  > qwen1.5-7b-1tp.log 2>&1 &

```







### qwen1


```
docker run  -it --rm --net=host --ipc=host \
--shm-size=50g \
--privileged=true \
-w /home \
--device=/dev/davinci_manager \
--device=/dev/hisi_hdc \
--device=/dev/devmm_svm \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/sbin:/usr/local/sbin \
-v /home:/home \
-v /tmp:/tmp \
-v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
-v /home/aicc/docker/qwen-72b.json:/usr/local/Ascend/mindie/latest/mindie-service/conf/config.json \
ascendhub.huawei.com/public-ascendhub/mindie-service-online:v1.1
```


### baichuan2

```
docker run  -it --rm \
--net=host --ipc=host \
--shm-size=50g \
--privileged=true \
-w /home \
--device=/dev/davinci_manager \
--device=/dev/hisi_hdc \
--device=/dev/devmm_svm \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/sbin:/usr/local/sbin \
-v /home:/home \
-v /tmp:/tmp \
-v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
-v /home/aicc/docker/baichuan2-7b.json:/usr/local/Ascend/mindie/latest/mindie-service/conf/config.json \
ascendhub.huawei.com/public-ascendhub/mindie-service-online:v1.1
```

```
docker run  -it --rm \
--net=host --ipc=host \
--shm-size=50g \
--privileged=true \
-w /home \
--device=/dev/davinci_manager \
--device=/dev/hisi_hdc \
--device=/dev/devmm_svm \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/sbin:/usr/local/sbin \
-v /home:/home \
-v /tmp:/tmp \
-v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
-v /home/aicc/docker/baichuan2-7b-4tp.json:/usr/local/Ascend/mindie/latest/mindie-service/conf/config.json \
ascendhub.huawei.com/public-ascendhub/mindie-service-online:v1.1
```

### baichuan2-13b


```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/mindie/set_env.sh 
source /usr/local/Ascend/llm_model/set_env.sh


export PYTHONPATH=/usr/local/Ascend/llm_model:$PYTHONPATH
cd /usr/local/Ascend/mindie/latest/mindie-service/bin

python convert_weights.py --model_path /home/aicc/model_from_hf/Baichuan2-13B-Chat
```


将congfig.json中的bfloat16改为float16。



```
docker run  -it --rm \
-e ASCEND_VISIBLE_DEVICES=0,1,2,3 \
--net=host --ipc=host \
--shm-size=50g \
--privileged=true \
-w /home \
--device=/dev/davinci_manager \
--device=/dev/hisi_hdc \
--device=/dev/devmm_svm \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/sbin:/usr/local/sbin \
-v /home:/home \
-v /tmp:/tmp \
-v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
-v /home/aicc/docker/baichuan2-13b.json:/usr/local/Ascend/mindie/latest/mindie-service/conf/config.json \
ascendhub.huawei.com/public-ascendhub/mindie-service-online:v1.1
```




```
docker run  -it --rm \
-e ASCEND_VISIBLE_DEVICES=0,1 \
--net=host --ipc=host \
--shm-size=50g \
--privileged=true \
-w /home \
--device=/dev/davinci_manager \
--device=/dev/hisi_hdc \
--device=/dev/devmm_svm \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/sbin:/usr/local/sbin \
-v /home:/home \
-v /tmp:/tmp \
-v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
-v /home/aicc/docker/baichuan2-13b-2tp.json:/usr/local/Ascend/mindie/latest/mindie-service/conf/config.json \
ascendhub.huawei.com/public-ascendhub/mindie-service-online:v1.1



nohup python performance-stream-baichuan2.py  > baichuan2-2tp.log 2>&1 &

```


### chatglm3


```
docker run  -it --rm \
--net=host --ipc=host \
--shm-size=50g \
--privileged=true \
-w /home \
--device=/dev/davinci_manager \
--device=/dev/hisi_hdc \
--device=/dev/devmm_svm \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/sbin:/usr/local/sbin \
-v /home:/home \
-v /tmp:/tmp \
-v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
-v /home/aicc/docker/baichuan2-13b-2tp.json:/usr/local/Ascend/mindie/latest/mindie-service/conf/config.json \
ascendhub.huawei.com/public-ascendhub/mindie-service-online:v1.1
```




```
docker run  -it --rm \
--net=host --ipc=host \
--shm-size=50g \
--privileged=true \
-w /home \
--device=/dev/davinci_manager \
--device=/dev/hisi_hdc \
--device=/dev/devmm_svm \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/sbin:/usr/local/sbin \
-v /home:/home \
-v /tmp:/tmp \
-v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
-v /home/aicc/docker/chatglm3-6b-4tp.json:/usr/local/Ascend/mindie/latest/mindie-service/conf/config.json \
ascendhub.huawei.com/public-ascendhub/mindie-service-online:v1.1
```


```
docker run  -it --rm \
--net=host --ipc=host \
--shm-size=50g \
--privileged=true \
-w /home \
--device=/dev/davinci_manager \
--device=/dev/hisi_hdc \
--device=/dev/devmm_svm \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/sbin:/usr/local/sbin \
-v /home:/home \
-v /tmp:/tmp \
-v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
-v /home/aicc/docker/chatglm3-6b-1tp.json:/usr/local/Ascend/mindie/latest/mindie-service/conf/config.json \
ascendhub.huawei.com/public-ascendhub/mindie-service-online:v1.1
```








