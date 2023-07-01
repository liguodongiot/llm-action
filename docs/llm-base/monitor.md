

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
