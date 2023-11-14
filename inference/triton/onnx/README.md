





```
model_repository
|
+-- densenet_onnx
    |
    +-- config.pbtxt
    +-- 1
        |
        +-- model.onnx
```


```
mkdir -p model_repository/densenet_onnx/1
wget -O model_repository/densenet_onnx/1/model.onnx \
     https://contentmamluswest001.blob.core.windows.net/content/14b2744cf8d6418c87ffddc3f3127242/9502630827244d60a1214f250e3bbca7/08aed7327d694b8dbaee2c97b8d0fcba/densenet121-1.2.onnx


docker pull nvcr.io/nvidia/tritonserver:23.10-py3

# /home/gdong/lgd

docker run --gpus all --rm \
-p 8000:8000 \
-p 8001:8001 \
-p 8002:8002 \
-v ${PWD}/model_repository:/models \
nvcr.io/nvidia/tritonserver:23.05-py3 \
tritonserver --model-repository=/models



```



```
docker pull nvcr.io/nvidia/tritonserver:23.05-py3-sdk

docker run -it --rm --net=host \
-v ${PWD}:/workspace/ \
nvcr.io/nvidia/tritonserver:23.05-py3-sdk \
bash

pip install torchvision

wget  -O img1.jpg "https://www.hakaimagazine.com/wp-content/uploads/header-gulf-birds.jpg"
```










