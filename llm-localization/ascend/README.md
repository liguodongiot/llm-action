


## pytorch

- https://gitee.com/ascend/pytorch
- https://gitee.com/ascend/transformers#fqa
- https://gitee.com/ascend/AscendSpeed
- https://gitee.com/ascend/AscendSpeed2



- https://gitee.com/ascend/DeepSpeed
- https://gitee.com/ascend/Megatron-LM




## mindspore

https://gitee.com/mindspore/mindspore

https://gitee.com/mindspore/mindformers


https://www.mindspore.cn/versions#2.2.10




```
cd /tmp
curl -O https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py37_4.10.3-Linux-$(arch).sh
bash Miniconda3-py37_4.10.3-Linux-$(arch).sh -b
cd -
. ~/miniconda3/etc/profile.d/conda.sh
conda init bash
```


```
conda create -n mindspore_py37 python=3.7.5 -y
conda activate mindspore_py37

```



```
CentOS 7可以使用以下命令安装。

sudo yum install centos-release-scl
sudo yum install devtoolset-7

安装完成后，需要使用如下命令切换到GCC 7。

scl enable devtoolset-7 bash


```



### 推理

https://www.mindspore.cn/lite/docs/zh-CN/r2.2/quick_start/one_hour_introduction_cloud.html






export LITE_HOME=$some_path/mindpsore-lite-2.0.0-linux-x64


export LD_LIBRARY_PATH=$LITE_HOME/runtime/lib:$LITE_HOME/runtime/third_party/dnnl:$LITE_HOME/tools/converter/lib:$LD_LIBRARY_PATH


export PATH=$LITE_HOME/tools/converter/converter:$LITE_HOME/tools/benchmark:$PATH






