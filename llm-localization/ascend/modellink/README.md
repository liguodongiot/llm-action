


```
docker pull --platform=arm64 swr.cn-central-221.ovaijisuan.com/dxy/pytorch2_1_0_kernels:PyTorch2.1.0-cann7.0.0.alpha003_py_3.9-euler_2.8.3-64GB
```


```

docker run -it  -u root  \
--device=/dev/davinci0  \
--device=/dev/davinci1  \
--device=/dev/davinci2  \
--device=/dev/davinci3  \
--device=/dev/davinci4  \
--device=/dev/davinci5  \
--device=/dev/davinci6  \
--device=/dev/davinci7  \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm   \
--device=/dev/hisi_hdc    \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver    \
-v /usr/local/dcmi:/usr/local/dcmi   \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi   \
-v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware  \
-v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi   \
-v /home/aicc:/home/ma-user/work/aicc    \
--name pytorch_ma   \
--entrypoint=/bin/bash   \
swr.cn-central-221.ovaijisuan.com/dxy/pytorch2_1_0_kernels:PyTorch2.1.0-cann7.0.0.alpha003_py_3.9-euler_2.8.3-64GB

```



```
docker start pytorch_ma

docker exec -it pytorch_ma-0-3 /bin/bash
```



--network=host

```
docker rm -f pytorch_ma-0-3

docker run -it  -u root --network=host \
--device=/dev/davinci0  \
--device=/dev/davinci1  \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm   \
--device=/dev/hisi_hdc    \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver    \
-v /usr/local/dcmi:/usr/local/dcmi   \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi   \
-v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware  \
-v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi   \
-v /home/aicc:/home/ma-user/work/aicc    \
--name pytorch_ma-0-3   \
--entrypoint=/bin/bash   \
swr.cn-central-221.ovaijisuan.com/dxy/pytorch2_1_0_kernels:PyTorch2.1.0-cann7.0.0.alpha003_py_3.9-euler_2.8.3-64GB

docker exec -it pytorch_ma-0-3 /bin/bash

```



```
# https://mirror.ghproxy.com
pip3 install --no-use-pep517 -e git+https://mirror.ghproxy.com/https://github.com/NVIDIA/Megatron-LM.git@23.05#egg=megatron-core
```




## torch_npu

torch_npu(Ascend Adapter for PyTorch插件)使昇腾NPU可以适配PyTorch框架，为使用PyTorch框架的开发者提供昇腾AI处理器的超强算力。

https://gitee.com/ascend/pytorch






```
import torch
import torch_npu

x = torch.randn(2, 2).npu()
y = torch.randn(2, 2).npu()
z = x.mm(y)

print(z)
```
