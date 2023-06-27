
## GCC 升级

```
yum update -y
yum install -y centos-release-scl
yum install -y devtoolset-9


source /opt/rh/devtoolset-9/enable

gcc -v
```


## GPU 驱动安装


```
chmod +x NVIDIA-Linux-x86_64-525.105.17.run
sh NVIDIA-Linux-x86_64-525.105.17.run -no-x-check

nvidia-smi
```


## CUDA 安装


```
mkdir -p /home/local/cuda-11.7
sudo ln -s /home/local/cuda-11.7 /usr/local/cuda-11.7

wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run
sudo sh cuda_11.7.0_515.43.04_linux.run

# 配置环境变量
export PATH="/usr/local/cuda-11.7/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH"

nvcc -V
```





## openssl 升级

```
yum install -y zlib-devel.x86_64
```

```
mkdir -p /home/local/openssl
ln -s /home/local/openssl /usr/local/openssl


tar -zxvf openssl-1.1.1t.tar.gz
cd openssl-1.1.1t/
./config --prefix=/usr/local/openssl shared zlib
make depend
sudo make 
sudo make install
sudo ln -s /usr/local/openssl/bin/openssl /usr/bin/openssl

echo "/usr/local/openssl/lib" >> /etc/ld.so.conf
ldconfig -v

ln -s /usr/local/openssl/lib/libssl.so.1.1 /usr/lib/libssl.so.1.1
ln -s /usr/local/openssl/lib/libcrypto.so.1.1 /usr/lib/libcrypto.so.1.1
```



## Docker 安装

```
wget https://download.docker.com/linux/centos/docker-ce.repo -O /etc/yum.repos.d/docker-ce.repo
wget https://nvidia.github.io/nvidia-docker/centos7/x86_64/nvidia-docker.repo -O /etc/yum.repos.d/nvidia-docker.repo
yum install -y epel-release
# 安装docker
yum install -y docker-ce nvidia-docker2
systemctl enable docker

# 重启docker
sudo systemctl restart docker

docker info
```



## Python3.10 安装

```
yum install libffi-devel
```


```
mkdir -p /home/local/py310
ln -s /home/local/py310 /usr/local/py310 

tar -xvf Python-3.10.10.tgz

./configure --prefix=/usr/local/py310 --with-openssl=/usr/local/openssl --with-openssl-rpath=no
make
sudo make install

sudo ln -s /usr/local/py310/bin/python3.10 /usr/bin/python3.10
sudo ln -s /usr/local/py310/bin/pip3.10 /usr/bin/pip3.10
# sudo ln -s /usr/local/py310/bin/pip3.10 /usr/bin/pip


pip3.10  install virtualenv -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn 
sudo ln -s /usr/local/py310/bin/virtualenv /usr/bin/virtualenv


mkdir -p /home/guodong.li/virtual-venv
cd /home/guodong.li/virtual-venv
virtualenv -p /usr/bin/python3.10 llama-venv-py310-cu117
source /home/guodong.li/virtual-venv/llama-venv-py310-cu117/bin/activate
```

配置pip源：

```
mkdir ~/.pip
cd ~/.pip
vim ~/.pip/pip.conf

[global] 
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
[install]
trusted-host = https://pypi.tuna.tsinghua.edu.cn
```



## NCCL 安装

```
tar xvf nccl_2.14.3-1+cuda11.7_x86_64.txz
cd nccl_2.14.3-1+cuda11.7_x86_64/

sudo cp -r include/* /usr/local/cuda-11.7/include/
sudo cp -r lib/* /usr/local/cuda-11.7/lib64/

export LD_LIBRARY_PATH="/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH"
```


## cuDNN 安装

```

cd cudnn-linux-x86_64-8.8.1.3_cuda11-archive
sudo cp include/cudnn*.h /usr/local/cuda-11.7/include 
sudo cp -P lib/libcudnn*  /usr/local/cuda-11.7/lib64/
sudo chmod a+r /usr/local/cuda-11.7/include/cudnn*.h /usr/local/cuda-11.7/lib64/libcudnn*
```


## Python


### Pytorch

```
pip install torch-1.13.1+cu117-cp310-cp310-linux_x86_64.whl
pip install torchvision-0.14.1+cu117-cp310-cp310-linux_x86_64.whl
```

### Transformers

```
cd transformers-20230327
git checkout 0041be5
pip install .
```

### Deepspeed...

```
pip install deepspeed==0.8.0
pip install accelerate
pip install tensorboardX
```

