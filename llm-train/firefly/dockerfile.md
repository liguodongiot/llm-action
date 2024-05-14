
```
FROM aiharbor.xxxx.local/base/python:py310-11.7-cudnn8-devel-centos7

MAINTAINER guodong.li liguodongiot@163.com

RUN yum -y install devtoolset-9 which && yum clean all && rm -rf /var/cache/yum/* && rm -rf /tmp/* \
&& echo ""  >> /etc/profile && echo "source /opt/rh/devtoolset-9/enable" >> /etc/profile

RUN conda init \
&& conda create -n torch1131-venv python=3.10 -y && source ~/.bashrc && conda env list && conda activate torch1131-venv  && pip install --no-cache-dir http://10.xx.2.25:81/pypi/pytorch/torch-1.13.1%2Bcu117-cp310-cp310-linux_x86_64.whl \
&& conda create -n torch201-venv python=3.10 -y && source ~/.bashrc && conda env list &&  conda activate torch201-venv && pip install --no-cache-dir http://10.cc.2.46:8000/base-env/torch-2.0.1+cu117-cp310-cp310-linux_x86_64.whl \
&& rm -rf ~/.cache/pip/* && conda clean -all && rm -rf /tmp/*


```

```

sudo docker build --network=host -f base-env.Dockerfile -t harbor.xxxx.io/base/tianqiong-base-env:v1-20240131 .


sudo docker run -it --gpus '"device=4,5"' --network=host \
--shm-size 4G \
harbor.xxx.io/base/tianqiong-base-env:v1-20240131  \
/bin/bash 

source ~/.bashrc 
conda activate torch1131-venv && pip list | grep torch
conda activate torch201-venv && pip list | grep torch
```


------------------------------------

```
FROM harbor.xxxx.io/base/tianqiong-base-env:v1-20240131

MAINTAINER guodong.li liguodongiot@163.com

ENV APP_DIR=/workspace
RUN mkdir -p -m 777 $APP_DIR

COPY train-env ${APP_DIR}/train-env

# llm-安装依赖
RUN source ~/.bashrc && conda env list && conda create -n llm-venv --clone torch1131-venv \
&& conda activate llm-venv && pip install --no-cache-dir -r ${APP_DIR}/train-env/requirements-llm.txt \
-i http://nexus3.xxx.com/repository/pypi/simple --trusted-host nexus3.xxx.com && rm -rf ~/.cache/pip/* && conda clean -all

#RUN source ~/.bashrc && conda env list && conda activate llm-venv && pip install  --no-cache-dir  storageutils==0.1.2 -i http://nexus3.xxx.com/repository/xxx_py_release/simple --trusted-host nexus3.xxx.com && rm -rf ~/.cache/pip/* && conda clean -all

RUN source ~/.bashrc && conda env list && conda activate llm-venv && cd ${APP_DIR}/train-env/bitsandbytes && source /opt/rh/devtoolset-9/enable && CUDA_VERSION=117 make cuda11x && python setup.py install && python setup.py clean --all && rm -rf ~/.cache/pip/* && conda clean -all


# baichuan2-t5-安装依赖
RUN source ~/.bashrc && conda env list && conda create -n llm-baichuan2-venv --clone torch201-venv && conda activate llm-baichuan2-venv && pip install  --no-cache-dir -r \
${APP_DIR}/train-env/requirements-llm-baichuan2.txt \
-i http://nexus3.xxx.com/repository/pypi/simple --trusted-host nexus3.xxx.com && rm -rf ~/.cache/pip/* && conda clean -all

#RUN source ~/.bashrc && conda env list && conda activate llm-baichuan2-venv && pip install  --no-cache-dir storageutils==0.1.2 -i http://nexus3.xxx.com/repository/xxx_py_release/simple --trusted-host nexus3.xxx.com && rm -rf ~/.cache/pip/* && conda clean -all

RUN source ~/.bashrc && conda env list && conda activate llm-baichuan2-venv && cd ${APP_DIR}/train-env/bitsandbytes && source /opt/rh/devtoolset-9/enable && CUDA_VERSION=117 make cuda11x && python setup.py install  && python setup.py clean --all && rm -rf ~/.cache/pip/* && conda clean -all

# t5-安装依赖
RUN source ~/.bashrc && conda env list && conda create -n t5-venv --clone torch201-venv \
&& conda activate t5-venv &&  pip install --no-cache-dir -r ${APP_DIR}/train-env/requirements-t5.txt \
-i http://nexus3.xxd.com/repository/pypi/simple --trusted-host nexus3.xxx.com && rm -rf ~/.cache/pip/* && conda clean -all

#RUN source ~/.bashrc && conda env list && conda activate t5-venv && pip install --no-cache-dir storageutils==0.1.2 -i http://nexus3.mxxx.com/repository/xxx_py_release/simple --trusted-host nexus3.fss.com && rm -rf ~/.cache/pip/* && conda clean -all

RUN rm -rf ${APP_DIR}/train-env

#设置工作目录
WORKDIR $APP_DIR


```

```

sudo docker build --network=host -f train-env.Dockerfile -t harbor.xxx.io/base/tianqiong-train-env:v1-20240131 .



sudo docker run -it --gpus '"device=4,5"' --network=host \
--shm-size 4G \
harbor.xxx.io/base/tianqiong-train-env:v1-20240131  \
/bin/bash 


source ~/.bashrc 
conda activate llm-venv && pip list | grep torch
conda activate llm-baichuan2-venv && pip list | grep torch
conda activate t5-venv && pip list | grep torch
```