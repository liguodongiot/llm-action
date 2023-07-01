

# NVIDIA DCGM


## Remove Older Installations

To remove the previous installation (if any), perform the following steps (e.g. on an RPM-based system).

Make sure that the nv-hostengine is not running. You can stop it using the following command:

```
# 启动nv-hostengine
# nv-hostengine
# 停止 nv-hostengine
sudo nv-hostengine -t
```

Remove the previous installation:

```
sudo yum remove datacenter-gpu-manager
```

## Install

### Ubuntu
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID | sed -e 's/\.//g')
wget https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/cuda-keyring_1.0-1_all.deb
dpkg -i cuda-keyring_1.0-1_all.deb
apt-get update
apt-get install -y datacenter-gpu-manager
```


### CentOS

centos8以上使用dnf，centos7建议还是使用yum.

centos7:
```
wget https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
#或者使用wget https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
yum install datacenter-gpu-manager
systemctl restart nvidia-dcgm.service && systemctl enable nvidia-dcgm.service

systemctl status nvidia-dcgm.service
```

centos8:
```
yum install epel-release
yum install dnf
```

```
# Determine the distribution name
distribution=$(. /etc/os-release;echo $ID`rpm -E "%{?rhel}%{?fedora}"`)
# Install the repository meta-data and the CUDA GPG key
sudo dnf config-manager \
    --add-repo http://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/cuda-rhel8.repo
# Update the repository metadata
sudo dnf clean expire-cache
# install DCGM
sudo dnf install -y datacenter-gpu-manager
```


### docker(dcgm)

url: https://hub.docker.com/r/nvidia/dcgm

```
docker pull nvidia/dcgm:3.1.7-1-ubuntu20.04
```

### docker(dcgm-exporter)

url: https://hub.docker.com/r/nvidia/dcgm-exporter
```
docker pull nvidia/dcgm-exporter:3.1.8-3.1.5-ubuntu20.04
```






## 访问 GPU Telemetry

在此场景中，DCGM 独立容器已使用以下命令启动，其中端口 5555 映射到主机，以便其他客户端可以访问容器中运行的 nv-hostengine 服务。 

请注意，要收集分析指标，需要向容器提供 SYS_ADMIN 功能：

```
docker run --gpus all \
   --cap-add SYS_ADMIN \
   -p 5555:5555 \
   nvidia/dcgm:3.1.7-1-ubuntu20.04
```

现在，诸如 dcgmi dmon 之类的客户端可以在控制台上传输 GPU Telemetry/指标。

## GPU 健康状况
在这种情况下，DCGM 不需要任何额外的caps，并且可以以非特权方式运行：

```
docker run --gpus all \
   -p 5555:5555 \
   nvidia/dcgm:3.1.7-1-ubuntu20.04
```

现在用于报告运行状况的 DCGM API 可以通过连接到 DCGM 容器的客户端访问。


