




```
docker login -u 157xxx4031 ascendhub.huawei.com
docker pull ascendhub.huawei.com/public-ascendhub/ascend-mindspore:23.0.0-A2-ubuntu18.04
```


```
https://www.hiascend.com/zh/software/cann/community-history


wget -c https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/Milan-ASL/Milan-ASL%20V100R001C18B800TP015/Ascend-cann-kernels-910b_8.0.RC2.alpha001_linux.run

wget -c https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/Milan-ASL/Milan-ASL%20V100R001C18B800TP015/Ascend-cann-toolkit_8.0.RC2.alpha001_linux-aarch64.run
```


```
docker stop pytorch_ubuntu_dev

docker rm -f pytorch_ubuntu_upgrade

docker run -it -u root \
--name pytorch_ubuntu_upgrade \
--network host \
--shm-size 4G \
-e ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
-v /etc/localtime:/etc/localtime \
-v /var/log/npu/:/usr/slog \
-v /usr/bin/hccn_tool:/usr/bin/hccn_tool \
-v /data/containerd/workspace/:/workspace \
ascendhub.huawei.com/public-ascendhub/ascend-mindspore:23.0.0-A2-ubuntu18.04 \
/bin/bash

docker start pytorch_ubuntu_upgrade
docker exec -it pytorch_ubuntu_upgrade bash


```



```

cd /usr/local/Ascend/ascend-toolkit/7.0.0/aarch64-linux/script/
./uninstall.sh 


chmod +x Ascend-cann-toolkit_8.0.RC2.alpha001_linux-aarch64.run
./Ascend-cann-toolkit_8.0.RC2.alpha001_linux-aarch64.run --check
./Ascend-cann-toolkit_8.0.RC2.alpha001_linux-aarch64.run --install

. /usr/local/Ascend/ascend-toolkit/set_env.sh
```


```
./Ascend-cann-kernels-910b_8.0.RC2.alpha001_linux.run --install --feature=aclnn_ops_train

```




```
. /usr/local/Ascend/ascend-toolkit/set_env.sh
```





```
bash Miniconda3-py39_24.4.0-0-Linux-aarch64.sh -p /workspace/installs/conda

conda init 

source ~/.bashrc



export PATH=/root/miniconda3/bin:$PATH

conda list



conda create -n llm-dev python=3.9
conda activate llm-dev 
```








