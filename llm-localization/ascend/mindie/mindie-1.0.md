

- https://ascendhub.huawei.com/#/detail/mindie
- ascendhub.huawei.com/public-ascendhub/mindie:1.0.RC1-800I-A2-aarch64


一键使能 CANN 软件栈的 shell 脚本（install_and_enable_cann.sh）


- /usr/local/Ascend/llm_model/pytorch/examples/chatglm2/6b/README.md


建议将权重存放于 /home/chatglm2_6b/weight 目录下，并设置 CHECKPOINT=/home/chatglm2_6b/weight



## 编写 docker 启动脚本

编写 docker 启动脚本 start-docker.sh 如下所示，存放于 /home/chatglm2_6b 目录下


```
IMAGES_ID=$1
NAME=$2
if [ $# -ne 2 ]; then
    echo "error: need one argument describing your container name."
    exit 1
fi
docker run --name ${NAME} -it -d --net=host --shm-size=500g \
    --privileged=true \
    -w /home \
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
    -e http_proxy=$http_proxy \
    -e https_proxy=$https_proxy \
    ${IMAGES_ID}
```


参数说明：

- IMAGES_ID 为镜像版本号。（docker images 命令回显中的 IMAGES ID）
- NAME 为启动容器名，可自定义设置。



## 启动并进入容器

依次执行如下命令启动并进入容器：

```
cd /home/chatglm2_6b
# 用户可以设置 docker images 命令回显中的 IMAGES ID
image_id=001b7368f6e0
# 用户可以自定义设置镜像名
custom_image_name=chatGLM2_6B
# 启动容器(确保启动容器前，本机可访问外网)
bash start-docker.sh ${image_id} ${custom_image_name}
# 进入容器
docker exec -itu root ${custom_image_name} bash
```


## 使能昇腾CANN软件栈

```
cd /opt/package
# 安装CANN包
source install_and_enable_cann.sh
# 若退出后重新进入容器，则需要重新加载 CANN 环境变量，执行以下三行命令
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/mindie/set_env.sh
source /usr/local/Ascend/llm_model/set_env.sh

```


## 推理 Chatglm2_6b 模型

```
cd /usr/local/Ascend/llm_model

# 权重转 safetensor
python examples/convert/convert_weights.py --model_path ${CHECKPOINT}

# 执行推理脚本
python examples/run_pa.py --model_path ${CHECKPOINT}

启动后会执行推理，显示默认问题Question和推理结果Answer，若用户想要自定义输入问题，可使用--input_texts参数设置，如：
python examples/run_pa.py --model_path ${CHECKPOINT} --input_texts "What is deep learning?"
```




## Qwen1.5-14B



```

# docker rm -f mindie-dev

docker run --name mindie-dev2 -it -d --net=host --ipc=host \
--shm-size=50g \
--privileged=true \
-w /home \
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
ascendhub.huawei.com/public-ascendhub/mindie:1.0.RC1-800I-A2-aarch64


docker exec -itu root mindie-dev2 bash



cd /opt/package
# 安装CANN包
source ./install_and_enable_cann.sh


# 若退出后重新进入容器，则需要重新加载 CANN 环境变量，执行以下三行命令
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/mindie/set_env.sh
source /usr/local/Ascend/llm_model/set_env.sh

rm /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json

vim /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json

- /home/aicc/model_from_hf/Qwen1.5-14B-Chat




export PYTHONPATH=/usr/local/Ascend/llm_model:$PYTHONPATH
cd /usr/local/Ascend/mindie/latest/mindie-service/bin
./mindieservice_daemon



```

```
transformers==4.30.2


pip install transformers==4.37.2 -i https://pypi.tuna.tsinghua.edu.cn/simple



"torch_dtype": "bfloat16"  改为 "float16" 
```











