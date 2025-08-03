

# 入口

- https://www.hiascend.com/document/detail/zh/mindie/20RC2/index/index.html

# 模型支持列表

- https://www.hiascend.com/software/mindie/modellist



- https://www.hiascend.com/document/detail/zh/mindie/10RC1/description/whatismindie/mindie_what_0000.html



```
docker run -it -u root --name=mindie_server_t35 --net=host --ipc=host \
--device=/dev/davinci0 \
--device=/dev/davinci1 \
--device=/dev/davinci2 \
--device=/dev/davinci3 \
--device=/dev/davinci4 \
--device=/dev/davinci5 \
--device=/dev/davinci6 \
--device=/dev/davinci7 \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
-v /usr/local/sbin/:/usr/local/sbin/ \
-v /var/log/npu/slog/:/var/log/npu/slog \
-v /var/log/npu/profiling/:/var/log/npu/profiling \
-v /var/log/npu/dump/:/var/log/npu/dump \
-v /var/log/npu/:/usr/slog \
-v /etc/hccn.conf:/etc/hccn.conf \
-v /home:/workspace \
mindie_server:1.0.T35 \
/bin/bash


docker exec -it mindie_server_t35 bash


/home/HwHiAiUser/mindie-service_1.0.RC1_linux-aarch64/bin

cp -r /workspace/token_input_gsm.csv .



vim conf/config.json 


cd /workspace/aicc/model_from_hf/chatglm3-6b-chat


/workspace/aicc/model_from_hf/Baichuan2-7B-Chat


---

/home/HwHiAiUser/atb-models/examples/convert


convert_weights.py


使用${llm_path}/examples/convert/convert_weights.py将bin转成safetensor格式


示例
python ${llm_path}/examples/convert/convert_weights.py --model_path ${weight_path}
输出结果会保存在bin权重同目录下


/home/HwHiAiUser

source  set_env.sh 


python examples/convert/convert_weights.py --model_path /workspace/aicc/model_from_hf/Baichuan2-7B-Chat --from_pretrained False

python examples/convert/convert_weights.py --model_path /workspace/aicc/model_from_hf/Baichuan2-7B-Chat 





---




启动脚本

Flash Attention的启动脚本路径为${llm_path}/examples/run_fa.py

Page Attention的启动脚本路径为${llm_path}/examples/run_pa.py
```



## 镜像



- https://ascendhub.huawei.com/#/detail/mindie

```
# 获取登录访问权限，输入已设置的“镜像下载凭证”,如果未设置或凭证超过24小时过期,请点击登录用户名下拉设置镜像下载凭证
docker login -u 157xxxx4031 ascendhub.huawei.com

# 下载镜像 
docker pull ascendhub.huawei.com/public-ascendhub/mindie:1.0.RC1-800I-A2-aarch64
```


## 迁移

```
docker save -o mindie-1.0.tar ascendhub.huawei.com/public-ascendhub/mindie:1.0.RC1-800I-A2-aarch64

scp root@192.xxx.16.211:/root/mindie-1.0.tar .


# 断点续传
rsync -P --rsh=ssh -r root@192.xxx.16.211:/root/mindie-1.0.tar .
```


## 性能测试

```
nohup python performance-stream-baichuan2.py  > baichuan2.log 2>&1 &
```






