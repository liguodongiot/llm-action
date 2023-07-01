

# NVIDIA DCGM




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
```
yum install epel-release
yum install dnf
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
