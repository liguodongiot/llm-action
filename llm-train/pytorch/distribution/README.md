


```
docker run -dt --name pytorch_env_cu117 --restart=always --gpus all \
--network=host \
--shm-size 4G \
-v /home/gdong/workspace/code:/workspace/code \
-v /home/gdong/workspace/data:/workspace/data \
-v /home/gdong/workspace/model:/workspace/model \
-v /home/gdong/workspace/output:/workspace/output \
-v /home/gdong/workspace/package:/workspace/package \
-w /workspace \
pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel \
/bin/bash


docker exec -it pytorch_env_cu117 bash

```



