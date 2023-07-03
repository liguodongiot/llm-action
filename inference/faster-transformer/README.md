
镜像下载地址：https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch

## FP16
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


```
cd code
git clone https://github.com/NVIDIA/FasterTransformer.git
cd FasterTransformer/
mkdir -p build
cd build
git submodule init && git submodule update
```

```
cmake -DSM=90 -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON ..
```


```
make -j12
```



## FP8

拉取镜像：
```
docker pull nvcr.io/nvidia/pytorch:22.10-py3
```
创建并启动容器：
```
nvidia-docker run -dti --name faster_transformer_fp8 \
--restart=always --gpus all --network=host \
--shm-size 5g \
-v /home/h800/h800-work/h800-workspace:/workspace \
-w /workspace \
nvcr.io/nvidia/pytorch:22.10-py3 \
bash

sudo docker exec -it faster_transformer_fp8 bash
```

拉取代码：
```
cd code
git clone https://github.com/NVIDIA/FasterTransformer.git
cd FasterTransformer/
git checkout eb9b81b65909cb14f582581c1ed4ee8e1e299be9
mkdir -p build
cd build
git submodule init && git submodule update
```



```
# cmake -DSM=90 -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON ..
cmake -DSM=90 -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON -DENABLE_FP8=ON ..
```

运行过程：
```
> cmake -DSM=90 -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON -DENABLE_FP8=ON ..
-- The CXX compiler identification is GNU 9.4.0
\-- The CUDA compiler identification is NVIDIA 11.8.89
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Detecting CUDA compiler ABI info
-- Detecting CUDA compiler ABI info - done
-- Check for working CUDA compiler: /usr/local/cuda/bin/nvcc - skipped
-- Detecting CUDA compile features
-- Detecting CUDA compile features - done
-- Looking for C++ include pthread.h
-- Looking for C++ include pthread.h - found
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Failed
-- Looking for pthread_create in pthreads
-- Looking for pthread_create in pthreads - not found
-- Looking for pthread_create in pthread
-- Looking for pthread_create in pthread - found
-- Found Threads: TRUE
-- Found CUDA: /usr/local/cuda (found suitable version "11.8", minimum required is "10.2")
CUDA_VERSION 11.8 is greater or equal than 11.0, enable -DENABLE_BF16 flag
CUDA_VERSION 11.8 is greater or equal than 11.8, enable -DENABLE_FP8 flag
-- Found CUDNN: /usr/lib/x86_64-linux-gnu/libcudnn.so
-- Add DBUILD_CUTLASS_MOE, requires CUTLASS. Increases compilation time
-- Add DBUILD_CUTLASS_MIXED_GEMM, requires CUTLASS. Increases compilation time
-- Running submodule update to fetch cutlass
-- Add DBUILD_MULTI_GPU, requires MPI and NCCL
-- Found MPI_CXX: /opt/hpcx/ompi/lib/libmpi.so (found version "3.1")
-- Found MPI: TRUE (found version "3.1")
-- Found NCCL: /usr/include
-- Determining NCCL version from /usr/include/nccl.h...
-- Looking for NCCL_VERSION_CODE
-- Looking for NCCL_VERSION_CODE - not found
-- Found NCCL (include: /usr/include, library: /usr/lib/x86_64-linux-gnu/libnccl.so.2.15.5)
-- NVTX is enabled.
-- Assign GPU architecture (sm=90)
-- Use WMMA
CMAKE_CUDA_FLAGS_RELEASE: -O3 -DNDEBUG -Xcompiler -O3 -DCUDA_PTX_FP8_F2FP_ENABLED --use_fast_math
-- COMMON_HEADER_DIRS: /workspace/FasterTransformer;/usr/local/cuda/include;/workspace/FasterTransformer/3rdparty/cutlass/include;/workspace/FasterTransformer/src/fastertransformer/cutlass_extensions/include;/workspace/FasterTransformer/3rdparty/trt_fp8_fmha/src;/workspace/FasterTransformer/3rdparty/trt_fp8_fmha/generated
-- Found CUDA: /usr/local/cuda (found version "11.8")
-- Caffe2: CUDA detected: 11.8
-- Caffe2: CUDA nvcc is: /usr/local/cuda/bin/nvcc
-- Caffe2: CUDA toolkit directory: /usr/local/cuda
-- Caffe2: Header version is: 11.8
CMake Warning (dev) at /opt/conda/lib/python3.8/site-packages/torch/share/cmake/Caffe2/public/cuda.cmake:117 (find_package):
  Policy CMP0074 is not set: find_package uses <PackageName>_ROOT variables.
  Run "cmake --help-policy CMP0074" for policy details.  Use the cmake_policy
  command to set the policy and suppress this warning.

  CMake variable CUDNN_ROOT is set to:

    /usr/local/cuda

  For compatibility, CMake is ignoring the variable.
Call Stack (most recent call first):
  /opt/conda/lib/python3.8/site-packages/torch/share/cmake/Caffe2/Caffe2Config.cmake:92 (include)
  /opt/conda/lib/python3.8/site-packages/torch/share/cmake/Torch/TorchConfig.cmake:68 (find_package)
  CMakeLists.txt:257 (find_package)
This warning is for project developers.  Use -Wno-dev to suppress it.

-- Found cuDNN: v8.6.0  (include: /usr/include, library: /usr/lib/x86_64-linux-gnu/libcudnn.so)
-- /usr/local/cuda/lib64/libnvrtc.so shorthash is 672ee683
CMake Warning at /opt/conda/lib/python3.8/site-packages/torch/share/cmake/Caffe2/public/utils.cmake:385 (message):
  In the future we will require one to explicitly pass TORCH_CUDA_ARCH_LIST
  to cmake instead of implicitly setting it as an env variable.  This will
  become a FATAL_ERROR in future version of pytorch.
Call Stack (most recent call first):
  /opt/conda/lib/python3.8/site-packages/torch/share/cmake/Caffe2/public/cuda.cmake:437 (torch_cuda_get_nvcc_gencode_flag)
  /opt/conda/lib/python3.8/site-packages/torch/share/cmake/Caffe2/Caffe2Config.cmake:92 (include)
  /opt/conda/lib/python3.8/site-packages/torch/share/cmake/Torch/TorchConfig.cmake:68 (find_package)
  CMakeLists.txt:257 (find_package)


-- Added CUDA NVCC flags for: -gencode;arch=compute_90,code=sm_90
CMake Warning at /opt/conda/lib/python3.8/site-packages/torch/share/cmake/Torch/TorchConfig.cmake:22 (message):
  static library kineto_LIBRARY-NOTFOUND not found.
Call Stack (most recent call first):
  /opt/conda/lib/python3.8/site-packages/torch/share/cmake/Torch/TorchConfig.cmake:127 (append_torchlib_if_found)
  CMakeLists.txt:257 (find_package)


-- Found Torch: /opt/conda/lib/python3.8/site-packages/torch/lib/libtorch.so
-- USE_CXX11_ABI=True
-- The C compiler identification is GNU 9.4.0
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /usr/bin/cc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Found Python: /opt/conda/bin/python3.8 (found version "3.8.13") found components: Interpreter
-- Configuring done
-- Generating done
-- Build files have been written to: /workspace/FasterTransformer/build
```


```
make -j12
```

运行过程：
```
> make -j12
[  0%] Building CXX object src/fastertransformer/utils/CMakeFiles/logger.dir/logger.cc.o
[  1%] Building CUDA object src/fastertransformer/kernels/CMakeFiles/transpose_int8_kernels.dir/transpose_int8_kernels.cu.o
[  1%] Building CUDA object src/fastertransformer/kernels/CMakeFiles/add_residual_kernels.dir/add_residual_kernels.cu.o
[  1%] Building CXX object 3rdparty/common/CMakeFiles/cuda_driver_wrapper.dir/cudaDriverWrapper.cpp.o
[  1%] Building CXX object src/fastertransformer/kernels/cutlass_kernels/CMakeFiles/cutlass_preprocessors.dir/cutlass_preprocessors.cc.o
[  2%] Building CXX object src/fastertransformer/utils/CMakeFiles/cuda_utils.dir/cuda_utils.cc.o
[  2%] Building CXX object src/fastertransformer/utils/CMakeFiles/nvtx_utils.dir/nvtx_utils.cc.o
[  2%] Building CUDA object src/fastertransformer/kernels/CMakeFiles/custom_ar_kernels.dir/custom_ar_kernels.cu.o
[  3%] Building CUDA object src/fastertransformer/kernels/CMakeFiles/activation_kernels.dir/activation_kernels.cu.o
[  3%] Building CUDA object src/fastertransformer/kernels/CMakeFiles/bert_preprocess_kernels.dir/bert_preprocess_kernels.cu.o
[  3%] Building CUDA object src/fastertransformer/kernels/CMakeFiles/unfused_attention_kernels.dir/unfused_attention_kernels.cu.o
[  3%] Building CUDA object src/fastertransformer/kernels/CMakeFiles/layernorm_kernels.dir/layernorm_kernels.cu.o
[  3%] Linking CUDA device code CMakeFiles/cuda_driver_wrapper.dir/cmake_device_link.o
[  3%] Linking CUDA device code CMakeFiles/nvtx_utils.dir/cmake_device_link.o
[  3%] Linking CXX static library ../../lib/libcuda_driver_wrapper.a
[  3%] Built target cuda_driver_wrapper
[  4%] Linking CXX static library ../../../lib/libnvtx_utils.a
[  4%] Building CUDA object src/fastertransformer/kernels/CMakeFiles/matrix_vector_multiplication.dir/matrix_vector_multiplication.cu.o
[  4%] Linking CUDA device code CMakeFiles/logger.dir/cmake_device_link.o
[  4%] Built target nvtx_utils
[  4%] Building CXX object src/fastertransformer/kernels/cutlass_kernels/CMakeFiles/cutlass_heuristic.dir/cutlass_heuristic.cc.o
[  4%] Linking CXX static library ../../../lib/liblogger.a
...
[ 99%] Built target multi_gpu_gpt_example
[ 99%] Building CXX object examples/cpp/multi_gpu_gpt/CMakeFiles/multi_gpu_gpt_async_example.dir/multi_gpu_gpt_async_example.cc.o
[ 99%] Built target gptneox_example
[ 99%] Building CXX object examples/cpp/multi_gpu_gpt/CMakeFiles/multi_gpu_gpt_triton_example.dir/multi_gpu_gpt_triton_example.cc.o
[ 99%] Linking CXX executable ../../../bin/gptneox_triton_example
[ 99%] Built target gptneox_triton_example
[ 99%] Building CXX object examples/cpp/multi_gpu_gpt/CMakeFiles/multi_gpu_gpt_interactive_example.dir/multi_gpu_gpt_interactive_example.cc.o
[100%] Linking CXX executable ../../../bin/multi_gpu_gpt_triton_example
[100%] Linking CXX static library ../../../../lib/libth_gpt_fp8.a
[100%] Building CXX object examples/cpp/gpt_fp8/CMakeFiles/gpt_fp8_triton_example.dir/gpt_fp8_triton_example.cc.o
[100%] Built target th_gpt_fp8
[100%] Linking CUDA device code CMakeFiles/transformer-shared.dir/cmake_device_link.o
[100%] Linking CXX shared library lib/libtransformer-shared.so
[100%] Built target multi_gpu_gpt_triton_example
[100%] Linking CXX executable ../../../bin/multi_gpu_gpt_async_example
[100%] Built target transformer-shared
[100%] Built target multi_gpu_gpt_async_example
[100%] Linking CXX executable ../../../bin/multi_gpu_gpt_interactive_example
[100%] Linking CXX static library ../../../../lib/libth_parallel_gpt.a
[100%] Built target th_parallel_gpt
[100%] Built target multi_gpu_gpt_interactive_example
[100%] Linking CXX executable ../../../bin/gpt_fp8_triton_example
[100%] Built target gpt_fp8_triton_example
[100%] Linking CXX static library ../../../../lib/libth_decoding.a
[100%] Built target th_decoding
[100%] Linking CXX static library ../../../../lib/libth_bart.a
[100%] Built target th_bart
[100%] Linking CXX shared library ../../../lib/libth_transformer.so
[100%] Built target th_transformer
```

```
pip install -r examples/pytorch/gpt/requirement.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
```


```
CUDA_VISIBLE_DEVICES=1 python examples/pytorch/gpt/lambada_task_example_gpt.py \
   --batch-size 1 \
   --checkpoint-path /workspace/model/megatron-models/c-model/345m/1-gpu \
   --lib-path /workspace/FasterTransformer/build/lib/libth_transformer.so \
   --lambada-path /workspace/data/lambada_test.jsonl
```




