


```
# dockerfile

docker build --network=host -f mindie-env-1.0.Dockerfile -t ascendhub.huawei.com/public-ascendhub/mindie-env:1.0.RC1-800I-A2-aarch64 .


docker run  -it --rm --net=host --ipc=host \
--shm-size=50g \
--privileged=true \
-w /home \
--device=/dev/davinci4 \
--device=/dev/davinci5 \
--device=/dev/davinci6 \
--device=/dev/davinci7 \
--device=/dev/davinci_manager \
--device=/dev/hisi_hdc \
--device=/dev/devmm_svm \
--entrypoint=bash \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/sbin:/usr/local/sbin \
-v /home:/home \
-v /tmp:/tmp \
-v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
ascendhub.huawei.com/public-ascendhub/mindie-env:1.0.RC1-800I-A2-aarch64




docker build --network=host -f mindie-all-1.0.Dockerfile -t ascendhub.huawei.com/public-ascendhub/mindie-all:1.0.RC1-800I-A2-aarch64 .



docker run  -it --rm --net=host --ipc=host \
--shm-size=50g \
--privileged=true \
-w /home \
--device=/dev/davinci4 \
--device=/dev/davinci5 \
--device=/dev/davinci6 \
--device=/dev/davinci7 \
--device=/dev/davinci_manager \
--device=/dev/hisi_hdc \
--device=/dev/devmm_svm \
--entrypoint=bash \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/sbin:/usr/local/sbin \
-v /home:/home \
-v /tmp:/tmp \
-v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
ascendhub.huawei.com/public-ascendhub/mindie-all:1.0.RC1-800I-A2-aarch64

```