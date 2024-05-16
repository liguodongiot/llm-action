




```
git clone https://gitee.com/ascend/ModelLink.git
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout -f bcce6f
cp -r megatron ../ModelLink/
cd ..
cd ModelLink
mkdir logs
mkdir model_from_hf
mkdir dataset
mkdir ckpt
```


```
# python3.8
conda create -n test python=3.8
conda activate test

# 安装 torch 和 torch_npu
pip install torch-2.1.0-cp38-cp38m-manylinux2014_aarch64.whl
pip install torch_npu-2.1.0*-cp38-cp38m-linux_aarch64.whl
pip install apex-0.1_ascend*-cp38-cp38m-linux_aarch64.whl

# 安装加速库
git clone https://gitee.com/ascend/AscendSpeed.git
cd AscendSpeed
git checkout 224ae35e8fc96778f957029d1371ddb623452a50
pip install -r requirements.txt
pip install -e .
cd ..

# 安装其余依赖库
pip install -r requirements.txt
```


