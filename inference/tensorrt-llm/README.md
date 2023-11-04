







----


基于镜像：nvcr.io/nvidia/pytorch:23.10-py3


卸载镜像中原来的tensorrt

```
pip uninstall -y tensorrt
pip install mpi4py

```

安装CMake

```
ARCH=$(uname -m)
CMAKE_VERSION="3.24.4"

PARSED_CMAKE_VERSION=$(echo $CMAKE_VERSION | sed 's/\.[0-9]*$//')
CMAKE_FILE_NAME="cmake-${CMAKE_VERSION}-linux-${ARCH}"
RELEASE_URL_CMAKE=https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/${CMAKE_FILE_NAME}.tar.gz
wget --no-verbose ${RELEASE_URL_CMAKE} -P /tmp
tar -xf /tmp/${CMAKE_FILE_NAME}.tar.gz -C /usr/local/
ln -s /usr/local/${CMAKE_FILE_NAME} /usr/local/cmake

echo 'export PATH=/usr/local/cmake/bin:$PATH' >> "${BASH_ENV}"
```




----





```
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git submodule update --init --recursive
git lfs install
git lfs pull




docker pull nvcr.io/nvidia/pytorch:23.10-py3

```





```
https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/9.1.0/tars/tensorrt-9.1.0.4.ubuntu-22.04.x86_64-gnu.cuda-12.1.tar.gz;



https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/9.1.0/tars/tensorrt-9.1.0.4.linux.x86_64-gnu.cuda-12.1.tar.gz;


https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/9.1.0/tars/tensorrt-9.1.0.4.linux.x86_64-gnu.cuda-12.1.tar.gz


```





```
docker rm -f tensorrt_llm

docker run -dt --name tensorrt_llm \
--restart=always \
--gpus all \
--network=host \
--shm-size=4g \
-m 64G \
-v /home/gdong:/workspace \
-w /workspace \
nvcr.io/nvidia/pytorch:23.05-py3 \
/bin/bash


docker exec -it tensorrt_llm bash




pip install transformers==4.31.0 -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn


pip install transformers==4.31.0 --progress-bar off -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn


```




```
apt-get update && apt-get -y install git git-lfs




mkdir -p /workspace/local/tensorrt

python3 ./scripts/build_wheel.py --trt_root /workspace/local/tensorrt


```





## 错误信息


```
RuntimeError: can't start new thread
```

解决：

更换docker镜像







```
sed -i -e 's/^APT/# APT/' -e 's/^DPkg/# DPkg/' /etc/apt/apt.conf.d/docker-clean
```

