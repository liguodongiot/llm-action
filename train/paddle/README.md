



```
nvidia-docker pull registry.baidubce.com/paddlepaddle/paddle:2.5.1-gpu-cuda11.7-cudnn8.4-trt8.4

nvidia-docker run --name paddle -it -v $PWD:/paddle registry.baidubce.com/paddlepaddle/paddle:2.5.1-gpu-cuda11.7-cudnn8.4-trt8.4 /bin/bash
```


```
python -m pip install paddlepaddle-gpu==2.5.1.post117 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```


```
import paddle
paddle.utils.run_check()
```



```
python3 -m pip uninstall paddlepaddle-gpu
```










