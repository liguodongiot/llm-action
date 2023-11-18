


## 安装

```
pip install --upgrade mxnet gluonnlp

pip install  mxnet==1.9.1 gluonnlp==0.10.0
```

## docker 

```
# GPU Instance
docker pull gluonai/gluon-nlp:gpu-latest
docker run --gpus all --rm -it -p 8888:8888 -p 8787:8787 -p 8786:8786 --shm-size=2g gluonai/gluon-nlp:gpu-latest

# CPU Instance
docker pull gluonai/gluon-nlp:cpu-latest
docker run --rm -it -p 8888:8888 -p 8787:8787 -p 8786:8786 --shm-size=2g gluonai/gluon-nlp:cpu-latest
```



```
docker run --gpus all  -itd \
--ipc=host \
--network host \
--shm-size=4g \
-v /home/guodong.li/workspace/:/workspace/ \
--name mxnet_dev \
gluonai/gluon-nlp:gpu-latest \
/bin/bash


docker exec -it mxnet_dev bash

pip uninstall mxnet-cu102
pip install  mxnet==1.9.1 gluonnlp==0.10.0 -i http://nexus3.xxx.com/repository/pypi/simple --trusted-host nexus3.xxx.com
```

