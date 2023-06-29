
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

GPUDirect 通信矩阵：
```
> nvidia-smi topo --matrix
        GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7    NIC0    NIC1    NIC2    NIC3    CPU Affinity    NUMA Affinity
GPU0     X      NV8     NV8     NV8     NV8     NV8     NV8     NV8     NODE    NODE    SYS     SYS     0-31,64-95      0
GPU1    NV8      X      NV8     NV8     NV8     NV8     NV8     NV8     PIX     NODE    SYS     SYS     0-31,64-95      0
GPU2    NV8     NV8      X      NV8     NV8     NV8     NV8     NV8     NODE    NODE    SYS     SYS     0-31,64-95      0
GPU3    NV8     NV8     NV8      X      NV8     NV8     NV8     NV8     NODE    PIX     SYS     SYS     0-31,64-95      0
GPU4    NV8     NV8     NV8     NV8      X      NV8     NV8     NV8     SYS     SYS     NODE    NODE    32-63,96-127    1
GPU5    NV8     NV8     NV8     NV8     NV8      X      NV8     NV8     SYS     SYS     NODE    NODE    32-63,96-127    1
GPU6    NV8     NV8     NV8     NV8     NV8     NV8      X      NV8     SYS     SYS     NODE    NODE    32-63,96-127    1
GPU7    NV8     NV8     NV8     NV8     NV8     NV8     NV8      X      SYS     SYS     PIX     PIX     32-63,96-127    1
NIC0    NODE    PIX     NODE    NODE    SYS     SYS     SYS     SYS      X      NODE    SYS     SYS
NIC1    NODE    NODE    NODE    PIX     SYS     SYS     SYS     SYS     NODE     X      SYS     SYS
NIC2    SYS     SYS     SYS     SYS     NODE    NODE    NODE    PIX     SYS     SYS      X      PIX
NIC3    SYS     SYS     SYS     SYS     NODE    NODE    NODE    PIX     SYS     SYS     PIX      X

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks

NIC Legend:

  NIC0: mlx5_0
  NIC1: mlx5_1
  NIC2: mlx5_2
  NIC3: mlx5_3
```


开启持久模式：

如果未开启持久模式（Persistence Mode），每次用nvidia-smi查询显卡资源的时候，会等到较长时间才有结果。
```
nvidia-smi -pm ENABLED

# 查询 Persistence Mode 是否开启
nvidia-smi -a
nvidia-smi
```


### 异常解决方案

- **问题**：Centos服务器重启后出现显卡驱动无法使用的情况：NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. 
- **解决方案**：

```
#  安装dkms修护驱动
> sudo yum install dkms

# 查看显卡驱动版本
> ls /usr/src
debug
kernels
nvidia-525.105.17

# 重新安装对应nvidia的驱动模块
> dkms install -m nvidia -v 525.105.17
```


## NVIDIA-Fabric Manager 安装 

NVIDIA-Fabric Manager服务可以使多A100显卡间通过NVSwitch互联。

要通过NVSwitch互联必须安装与GPU驱动版本对应的NVIDIA-Fabric Manager软件包，否则将无法正常使用实例。

参考：https://www.volcengine.com/docs/6419/73634

```
wget -c https://developer.download.nvidia.cn/compute/cuda/repos/rhel7/x86_64/nvidia-fabric-manager-525.105.17-1.x86_64.rpm
rpm -ivh nvidia-fabric-manager-525.105.17-1.x86_64.rpm

```

```
wget -c https://developer.download.nvidia.com/compute/cuda/repos/fedora37/x86_64/nvidia-fabric-manager-devel-525.105.17-1.x86_64.rpm
rpm -ivh nvidia-fabric-manager-devel-525.105.17-1.x86_64.rpm
```


启动NVIDIA-Fabric Manager：
```
# 启动Fabric Manager服务，实现NVSwitch互联
sudo systemctl restart nvidia-fabricmanager

# 查看Fabric Manager服务是否正常启动，回显active（running）表示启动成功。
sudo systemctl status nvidia-fabricmanager

# 配置Fabric Manager服务随实例开机自启动。
sudo systemctl enable nvidia-fabricmanager
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







## Python3.10 安装

```
yum install -y libffi-devel bzip2-devel
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
tar -xvf cudnn-linux-x86_64-8.8.1.3_cuda11-archive.tar.xz

cd cudnn-linux-x86_64-8.8.1.3_cuda11-archive
sudo cp include/cudnn*.h /usr/local/cuda-11.7/include 
sudo cp -P lib/libcudnn*  /usr/local/cuda-11.7/lib64/
sudo chmod a+r /usr/local/cuda-11.7/include/cudnn*.h /usr/local/cuda-11.7/lib64/libcudnn*
```


## Python 库安装


### Pytorch

```
pip install torch-1.13.1+cu117-cp310-cp310-linux_x86_64.whl
pip install torchvision-0.14.1+cu117-cp310-cp310-linux_x86_64.whl

# pip install torch-scatter torch-sparse torch-cluster torch-geometric
```

```
python -c "import torch; print(torch.cuda.is_available())"
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

### 安装Apex

```
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout 22.04-dev
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

### 安装mpi4py

```
yum -y install openmpi-devel
export CC=/usr/lib64/openmpi/bin/mpicc

pip install mpi4py
```





## pdsh安装

```
tar xf pdsh-2.31.tar.gz 
cd /data/nfs/llm/pkg/pdsh-pdsh-2.31

./configure \
--prefix=/home/local/pdsh \
--with-ssh \
--with-machines=/home/local/pdsh/machines \
--with-dshgroups=/home/local/pdsh/group \
--with-rcmd-rank-list=ssh \
--with-exec && \
make && \
make install


ln -s /home/local/pdsh /usr/local/pdsh

ll /usr/local/pdsh/bin/

# 将pdsh的所有命令追加到环境变量中
echo "export PATH=/home/local/pdsh/bin:$PATH" >> /etc/profile
source /etc/profile

pdsh -V
```


## nvidia-docker 安装

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

---

修改Docker配置（`/etc/docker/daemon.json`）：

```
{
    "data-root": "/home/docker",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
```
重启docker服务:

```
systemctl daemon-reload
systemctl restart docker
```

---

关闭 selinux 安全系统：

```
1. 临时关闭（setenforce 0），系统重启后，恢复启动。
setenforce 0

查看：
getenforce


2. 永久关闭，修改文件 /etc/selinux/config

SELINUX=disabled

保存后，重启 reboot
```



### 镜像导入及归档

```
docker save -o tritonserver.tar runoob/ubuntu:v3

docker load --input tritonserver.tar
```

常见docker命令参考该[文档](https://juejin.cn/post/7016238524286861325)。








