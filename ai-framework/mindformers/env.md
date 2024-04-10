


```
docker pull --platform=arm64 swr.cn-central-221.ovaijisuan.com/mindformers/mindformers1.0_mindspore2.2.11:aarch_20240125
```


```
docker run -it -u root \
--ipc=host \
--network=host \
--device=/dev/davinci0 \
--device=/dev/davinci1 \
--device=/dev/davinci2 \
--device=/dev/davinci3 \
--device=/dev/davinci4 \
--device=/dev/davinci5 \
--device=/dev/davinci6 \
--device=/dev/davinci7 \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-v /var/log/npu/:/usr/slog \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
--name mindformers_dev \
swr.cn-central-221.ovaijisuan.com/mindformers/mindformers1.0_mindspore2.2.11:aarch_20240125 \
/bin/bash
```

```
git clone -b dev https://gitee.com/mindspore/mindformers.git
cd mindformers
bash build.sh
```
