



## 驱动


```
chmod +x NVIDIA-Linux-x86_64-525.105.17.run
sh NVIDIA-Linux-x86_64-525.105.17.run -no-x-check

nvidia-smi
```


## CUDA


```
sudo ln -s /home/local/cuda-11.7 /usr/local/cuda-11.7
sudo sh cuda_11.7.1_515.65.01_linux.run
```

## 

```

```



## Docker 

```


# 安装docker
sudo yum install -y nvidia-docker2

# 重启docker
sudo systemctl restart docker
```






