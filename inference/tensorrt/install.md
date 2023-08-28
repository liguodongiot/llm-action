



```
cd /home/guodong.li/virtual-venv
virtualenv -p /usr/bin/python3.10 model-inference-venv-py310-cu117
source /home/guodong.li/virtual-venv/model-inference-venv-py310-cu117/bin/activate



pip install torch


version="8.6.1.6"
arch=$(uname -m)
cuda="cuda-11.8"
tar -xzvf TensorRT-${version}.Linux.${arch}-gnu.${cuda}.tar.gz




export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<TensorRT-${version}/lib>


cd TensorRT-${version}/python

python3 -m pip install tensorrt-*-cp310-none-linux_x86_64.whl

python3 -m pip install tensorrt_lean-*-cp310-none-linux_x86_64.whl
python3 -m pip install tensorrt_dispatch-*-cp310-none-linux_x86_64.whl



python3 -m pip install uff-0.6.9-py2.py3-none-any.whl


which convert-to-uff

---

cd TensorRT-${version}/graphsurgeon

python3 -m pip install graphsurgeon-0.4.6-py2.py3-none-any.whl

---

cd TensorRT-${version}/onnx_graphsurgeon
	
python3 -m pip install onnx_graphsurgeon-0.3.12-py2.py3-none-any.whl


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/guodong.li/TensorRT-8.6.1.6/lib




import tensorrt 
print(tensorrt.version) 
assert tensorrt.Builder(tensorrt.Logger())


import tensorrt_lean as trt
print(trt.version) 
assert trt.Builder(trt.Logger())


----exit

python3
>>> import tensorrt
>>> print(tensorrt.__version__)
>>> assert tensorrt.Builder(tensorrt.Logger())


Use a similar procedure to verify that the lean and dispatch modules work as expected:
python3
>>> import tensorrt_lean as trt
>>> print(trt.__version__)
>>> assert trt.Builder(trt.Logger())



python3
>>> import tensorrt_dispatch as trt
>>> print(trt.__version__)
>>> assert trt.Builder(trt.Logger())





pip install torch_tensorrt

```
