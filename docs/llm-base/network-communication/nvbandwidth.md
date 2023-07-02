

用于测量 NVIDIA GPU 带宽的工具。

使用copy engine或kernel copy方法测量不同链路上各种 memcpy 模式的带宽。 

nvbandwidth 报告系统上当前测量的带宽。 可能需要额外的系统特定调整才能实现最大峰值带宽。




```
> sh ./debian_install.sh
Reading package lists... Done
Building dependency tree
Reading state information... Done
build-essential is already the newest version (12.8ubuntu1.1).
0 upgraded, 0 newly installed, 0 to remove and 94 not upgraded.
Reading package lists... Done
Building dependency tree
Reading state information... Done
libboost-program-options-dev is already the newest version (1.71.0.0ubuntu2).
0 upgraded, 0 newly installed, 0 to remove and 94 not upgraded.
Reading package lists... Done
Building dependency tree
Reading state information... Done
The following additional packages will be installed:
  cmake-data libarchive13 libicu66 libjsoncpp1 librhash0 libuv1 libxml2 tzdata
Suggested packages:
  cmake-doc ninja-build lrzip
The following NEW packages will be installed:
  cmake cmake-data libarchive13 libicu66 libjsoncpp1 librhash0 libuv1 libxml2
  tzdata
0 upgraded, 9 newly installed, 0 to remove and 94 not upgraded.
Need to get 15.3 MB of archives.
...
Setting up libuv1:amd64 (1.34.2-1ubuntu1.3) ...
Setting up librhash0:amd64 (1.3.9-1) ...
Setting up cmake-data (3.16.3-1ubuntu1.20.04.1) ...
Setting up libjsoncpp1:amd64 (1.7.4-3.1ubuntu2) ...
Setting up libicu66:amd64 (66.1-2ubuntu2.1) ...
Setting up libxml2:amd64 (2.9.10+dfsg-5ubuntu0.20.04.6) ...
Setting up libarchive13:amd64 (3.4.0-2ubuntu1.2) ...
Setting up cmake (3.16.3-1ubuntu1.20.04.1) ...
Processing triggers for libc-bin (2.31-0ubuntu9.9) ...
-- The CUDA compiler identification is NVIDIA 11.8.89
-- The CXX compiler identification is GNU 9.4.0
-- Detecting CUDA compiler ABI info
-- Detecting CUDA compiler ABI info - done
-- Check for working CUDA compiler: /usr/local/cuda/bin/nvcc - skipped
-- Detecting CUDA compile features
-- Detecting CUDA compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Found Boost: /usr/lib/x86_64-linux-gnu/cmake/Boost-1.71.0/BoostConfig.cmake (found version "1.71.0") found components: program_options
-- Configuring done
-- Generating done
-- Build files have been written to: /root/nvbandwidth
[ 14%] Building CXX object CMakeFiles/nvbandwidth.dir/testcase.cpp.o
[ 28%] Building CXX object CMakeFiles/nvbandwidth.dir/testcases_ce.cpp.o
[ 42%] Building CXX object CMakeFiles/nvbandwidth.dir/testcases_sm.cpp.o
[ 57%] Building CUDA object CMakeFiles/nvbandwidth.dir/kernels.cu.o
[ 71%] Building CXX object CMakeFiles/nvbandwidth.dir/memcpy.cpp.o
[ 85%] Building CXX object CMakeFiles/nvbandwidth.dir/nvbandwidth.cpp.o
[100%] Linking CXX executable nvbandwidth
[100%] Built target nvbandwidth
```


```
./nvbandwidth -h
nvbandwidth Version: v0.2
Built from Git version: 42e94d2

nvbandwidth CLI:
  -h [ --help ]             Produce help message
  --bufferSize arg (=64)    Memcpy buffer size in MiB
  --loopCount arg (=16)     Iterations of memcpy to be performed
  -l [ --list ]             List available testcases
  -t [ --testcase ] arg     Testcase(s) to run (by name or index)
  -v [ --verbose ]          Verbose output
  -d [ --disableAffinity ]  Disable automatic CPU affinity control

```


官方文档：https://github.com/NVIDIA/nvbandwidth
