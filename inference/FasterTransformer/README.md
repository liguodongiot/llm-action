
镜像下载地址：https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch

```
docker pull nvcr.io/nvidia/pytorch:23.05-py3
```

```
nvidia-docker run -dti --name faster_transformer \
--restart=always --gpus all --network=host \
--shm-size 5g \
-v /home/h800/h800-work/h800-workspace:/workspace \
-w /workspace \
nvcr.io/nvidia/pytorch:23.05-py3 \
bash

sudo docker exec -it faster_transformer bash
```
