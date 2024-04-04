

```

 docker pull swr.cn-central-221.ovaijisuan.com/dxy/mindspore_kernels:MindSpore2.2.10-cann7.0.0beta1_py_3.9-euler_2.8.3-32GB


```

```

docker run -it  -u root  \
--device=/dev/davinci0   \
--device=/dev/davinci1   \
--device=/dev/davinci2   \
--device=/dev/davinci3   \
--device=/dev/davinci4   \
--device=/dev/davinci5   \
--device=/dev/davinci6   \
--device=/dev/davinci7   \
--device=/dev/davinci_manager   \
--device=/dev/devmm_svm   \
--device=/dev/hisi_hdc   \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver    \
-v /usr/local/dcmi:/usr/local/dcmi   \
-v /usr/local/Ascend/toolbox:/usr/local/Ascend/toolbox    \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi   \
-v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware   \
-v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi   \
-v /home/aicc:/home/ma-user/work/aicc    \
--name mindspore_ma   \
--entrypoint=/bin/bash  \
swr.cn-central-221.ovaijisuan.com/dxy/mindspore_kernels:MindSpore2.2.10-cann7.0.0beta1_py_3.9-euler_2.8.3-32GB \
/bin/bash
```

