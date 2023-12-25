
| Fermi **†** | Kepler **†** | Maxwell **‡** | Pascal | Volta | Turing | Ampere | Ada (Lovelace) | [Hopper](https://www.nvidia.com/en-us/data-center/hopper-architecture/) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sm_20 | sm_30 | sm_50 | sm_60 | sm_70 | sm_75 | sm_80 | sm_89 | sm_90 |
|     | sm_35 | sm_52 | sm_61 | sm_72<br>(Xavier) |     | sm_86 |     | sm_90a (Thor) |
|     | sm_37 | sm_53 | sm_62 |     |     | sm_87 (Orin) |     |     |

**†** Fermi and Kepler are deprecated from CUDA 9 and 11 onwards

**‡** Maxwell is deprecated from CUDA 11.6 onwards

参考：https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/


如果使用H800，CUDA版本要在11.8及以上，同时，PyTorch版本要在2.0.0以上。下面是我使用CUDA为11.7，同时PyTorch为1.13.1的报错信息。

```
NVIDIA H800 with CUDA capability sm_90 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_37 sm_50 sm_60 sm_70 sm_75 sm_80 sm_86.
```

## CUDA

```
mkdir -p /home/local/cuda-11.8
sudo ln -s /home/local/cuda-11.7 /usr/local/cuda-11.8

wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
chmod +x cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
```





## NCCL 安装

```
tar xvf nccl_2.16.5-1+cuda11.8_x86_64.txz
cd nccl_2.16.5-1+cuda11.8_x86_64/
sudo cp -r include/* /usr/local/cuda-11.8/include/
sudo cp -r lib/* /usr/local/cuda-11.8/lib64/

# export LD_LIBRARY_PATH="/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH"
```

## cuDNN 安装

```
tar -xvf cudnn-linux-x86_64-8.9.2.26_cuda11-archive.tar.xz

cd cudnn-linux-x86_64-8.9.2.26_cuda11-archive
sudo cp include/cudnn*.h /usr/local/cuda-11.8/include 
sudo cp -P lib/libcudnn*  /usr/local/cuda-11.8/lib64/
sudo chmod a+r /usr/local/cuda-11.8/include/cudnn*.h /usr/local/cuda-11.8/lib64/libcudnn*
```



## 虚拟环境

```
# mkdir -p /home/guodong.li/virtual-venv

cd /home/guodong.li/virtual-venv
virtualenv -p /usr/bin/python3.10 llama-venv-py310-cu118
source /home/guodong.li/virtual-venv/llama-venv-py310-cu118/bin/activate
```

## Python 库安装


### Pytorch

下载地址: https://download.pytorch.org/whl/torch_stable.html



```
wget -c https://download.pytorch.org/whl/cu118/torch-2.0.0%2Bcu118-cp310-cp310-linux_x86_64.whl
wget -c https://download.pytorch.org/whl/cu118/torchvision-0.15.0%2Bcu118-cp310-cp310-linux_x86_64.whl

pip install torch-2.0.0+cu118-cp310-cp310-linux_x86_64.whl
pip install torchvision-0.15.0+cu118-cp310-cp310-linux_x86_64.whl
```






### Deepspeed...

```
pip install deepspeed==0.9.5
pip install accelerate
pip install tensorboardX
```

### 安装Apex

```
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout 30a7ad3
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
```





## 配套


```
wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run
sudo sh cuda_12.1.1_530.30.02_linux.run
```


```
NCCL：

https://developer.nvidia.com/downloads/compute/machine-learning/nccl/secure/2.18.3/agnostic/x64/nccl_2.18.3-1+cuda12.1_x86_64.txz/


CUDNN：




pytorch:

https://download.pytorch.org/whl/cu121/torch-2.1.2%2Bcu121-cp310-cp310-linux_x86_64.whl
https://download.pytorch.org/whl/cu121/torchaudio-2.1.2%2Bcu121-cp310-cp310-linux_x86_64.whl
https://download.pytorch.org/whl/cu121/torchvision-0.16.2%2Bcu121-cp310-cp310-linux_x86_64.whl
```


