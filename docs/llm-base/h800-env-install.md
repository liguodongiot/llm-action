


## CUDA

| Fermi **†** | Kepler **†** | Maxwell **‡** | Pascal | Volta | Turing | Ampere | Ada (Lovelace) | [Hopper](https://www.nvidia.com/en-us/data-center/hopper-architecture/) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sm_20 | sm_30 | sm_50 | sm_60 | sm_70 | sm_75 | sm_80 | sm_89 | sm_90 |
|     | sm_35 | sm_52 | sm_61 | sm_72<br>(Xavier) |     | sm_86 |     | sm_90a (Thor) |
|     | sm_37 | sm_53 | sm_62 |     |     | sm_87 (Orin) |     |     |

**†** Fermi and Kepler are deprecated from CUDA 9 and 11 onwards

**‡** Maxwell is deprecated from CUDA 11.6 onwards

参考：https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/

```
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
```



## NCCL 安装

```

```

## cuDNN 安装

```

```

## Pytorch

```
wget -c https://download.pytorch.org/whl/cu118/torch-2.0.0%2Bcu118-cp310-cp310-linux_x86_64.whl
```



