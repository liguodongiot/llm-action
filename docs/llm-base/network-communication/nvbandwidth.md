# nvbandwidth

用于测量 NVIDIA GPU 带宽的工具。

使用copy engine或kernel copy方法测量不同链路上各种 memcpy 模式的带宽。 

nvbandwidth 报告系统上当前测量的带宽。 可能需要额外的系统特定调整才能实现最大峰值带宽。


## 安装
<details><summary>Example output</summary><p>


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

</p></details>


## help

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
  -v [ --verbose ]          Verbose output(详细输出)
  -d [ --disableAffinity ]  Disable automatic CPU affinity control

```

---

## 列出可用的测试用例

```
> ./nvbandwidth -l
nvbandwidth Version: v0.2
Built from Git version: 42e94d2

Index, Name:
        Description
=======================
0, host_to_device_memcpy_ce:
        Host to device CE memcpy using cuMemcpyAsync

1, device_to_host_memcpy_ce:
        Device to host CE memcpy using cuMemcpyAsync

2, host_to_device_bidirectional_memcpy_ce:
        A host to device copy is measured while a device to host copy is run simultaneously.
        Only the host to device copy bandwidth is reported.

3, device_to_host_bidirectional_memcpy_ce:
        A device to host copy is measured while a host to device copy is run simultaneously.
        Only the device to host copy bandwidth is reported.

4, device_to_device_memcpy_read_ce:
        Measures bandwidth of cuMemcpyAsync between each pair of accessible peers.
        Read tests launch a copy from the peer device to the target using the target's context.

5, device_to_device_memcpy_write_ce:
        Measures bandwidth of cuMemcpyAsync between each pair of accessible peers.
        Write tests launch a copy from the target device to the peer using the target's context.

6, device_to_device_bidirectional_memcpy_read_ce:
        Measures bandwidth of cuMemcpyAsync between each pair of accessible peers.
        A copy in the opposite direction of the measured copy is run simultaneously but not measured.
        Read tests launch a copy from the peer device to the target using the target's context.

7, device_to_device_bidirectional_memcpy_write_ce:
        Measures bandwidth of cuMemcpyAsync between each pair of accessible peers.
        A copy in the opposite direction of the measured copy is run simultaneously but not measured.
        Write tests launch a copy from the target device to the peer using the target's context.

8, all_to_host_memcpy_ce:
        Measures bandwidth of cuMemcpyAsync between a single device and the host while simultaneously
        running copies from all other devices to the host.

9, all_to_host_bidirectional_memcpy_ce:
        A device to host copy is measured while a host to device copy is run simultaneously.
        Only the device to host copy bandwidth is reported.
        All other devices generate simultaneous host to device and device to host interferring traffic.

10, host_to_all_memcpy_ce:
        Measures bandwidth of cuMemcpyAsync between the host to a single device while simultaneously
        running copies from the host to all other devices.

11, host_to_all_bidirectional_memcpy_ce:
        A host to device copy is measured while a device to host copy is run simultaneously.
        Only the host to device copy bandwidth is reported.
        All other devices generate simultaneous host to device and device to host interferring traffic.

12, all_to_one_write_ce:
        Measures the total bandwidth of copies from all accessible peers to a single device, for each
        device. Bandwidth is reported as the total inbound bandwidth for each device.
        Write tests launch a copy from the target device to the peer using the target's context.

13, all_to_one_read_ce:
        Measures the total bandwidth of copies from all accessible peers to a single device, for each
        device. Bandwidth is reported as the total outbound bandwidth for each device.
        Read tests launch a copy from the peer device to the target using the target's context.

14, one_to_all_write_ce:
        Measures the total bandwidth of copies from a single device to all accessible peers, for each
        device. Bandwidth is reported as the total outbound bandwidth for each device.
        Write tests launch a copy from the target device to the peer using the target's context.

15, one_to_all_read_ce:
        Measures the total bandwidth of copies from a single device to all accessible peers, for each
        device. Bandwidth is reported as the total inbound bandwidth for each device.
        Read tests launch a copy from the peer device to the target using the target's context.

16, host_to_device_memcpy_sm:
        Host to device SM memcpy using a copy kernel

17, device_to_host_memcpy_sm:
        Device to host SM memcpy using a copy kernel

18, device_to_device_memcpy_read_sm:
        Measures bandwidth of a copy kernel between each pair of accessible peers.
        Read tests launch a copy from the peer device to the target using the target's context.

19, device_to_device_memcpy_write_sm:
        Measures bandwidth of a copy kernel between each pair of accessible peers.
        Write tests launch a copy from the target device to the peer using the target's context.

20, device_to_device_bidirectional_memcpy_read_sm:
        Measures bandwidth of a copy kernel between each pair of accessible peers. Copies are run
        in both directions between each pair, and the sum is reported.
        Read tests launch a copy from the peer device to the target using the target's context.

21, device_to_device_bidirectional_memcpy_write_sm:
        Measures bandwidth of a copy kernel between each pair of accessible peers. Copies are run
        in both directions between each pair, and the sum is reported.
        Write tests launch a copy from the target device to the peer using the target's context.

22, all_to_host_memcpy_sm:
        Measures bandwidth of a copy kernel between a single device and the host while simultaneously
        running copies from all other devices to the host.

23, all_to_host_bidirectional_memcpy_sm:
        A device to host bandwidth of a copy kernel is measured while a host to device copy is run simultaneously.
        Only the device to host copy bandwidth is reported.
        All other devices generate simultaneous host to device and device to host interferring traffic using copy kernels.

24, host_to_all_memcpy_sm:
        Measures bandwidth of a copy kernel between the host to a single device while simultaneously
        running copies from the host to all other devices.

25, host_to_all_bidirectional_memcpy_sm:
        A host to device bandwidth of a copy kernel is measured while a device to host copy is run simultaneously.
        Only the host to device copy bandwidth is reported.
        All other devices generate simultaneous host to device and device to host interferring traffic using copy kernels.

26, all_to_one_write_sm:
        Measures the total bandwidth of copies from all accessible peers to a single device, for each
        device. Bandwidth is reported as the total inbound bandwidth for each device.
        Write tests launch a copy from the target device to the peer using the target's context.

27, all_to_one_read_sm:
        Measures the total bandwidth of copies from all accessible peers to a single device, for each
        device. Bandwidth is reported as the total outbound bandwidth for each device.
        Read tests launch a copy from the peer device to the target using the target's context.

28, one_to_all_write_sm:
        Measures the total bandwidth of copies from a single device to all accessible peers, for each
        device. Bandwidth is reported as the total outbound bandwidth for each device.
        Write tests launch a copy from the target device to the peer using the target's context.

29, one_to_all_read_sm:
        Measures the total bandwidth of copies from a single device to all accessible peers, for each
        device. Bandwidth is reported as the total inbound bandwidth for each device.
        Read tests launch a copy from the peer device to the target using the target's context.

```


## 运行所有测试用例

```
> ./nvbandwidth

nvbandwidth Version: v0.2
Built from Git version: 42e94d2

NOTE: This tool reports current measured bandwidth on your system.
Additional system-specific tuning may be required to achieve maximal peak bandwidth.

CUDA Runtime Version: 11080
CUDA Driver Version: 12000
Driver Version: 525.105.17

Device 0: NVIDIA H800
Device 1: NVIDIA H800
Device 2: NVIDIA H800
Device 3: NVIDIA H800
Device 4: NVIDIA H800
Device 5: NVIDIA H800
Device 6: NVIDIA H800
Device 7: NVIDIA H800

Running host_to_device_memcpy_ce.
memcpy CE CPU(row) -> GPU(column) bandwidth (GB/s)
          0         1         2         3         4         5         6         7
0     55.28     55.30     55.32     55.30     55.31     55.34     55.32     55.34

SUM host_to_device_memcpy_ce 442.52

Running device_to_host_memcpy_ce.
memcpy CE CPU(row) <- GPU(column) bandwidth (GB/s)
          0         1         2         3         4         5         6         7
0     55.15     55.15     55.15     55.15     55.09     55.13     55.10     55.13

SUM device_to_host_memcpy_ce 441.05

Running host_to_device_bidirectional_memcpy_ce.
memcpy CE CPU(row) <-> GPU(column) bandwidth (GB/s)
          0         1         2         3         4         5         6         7
0     52.75     52.33     52.96     52.86     52.70     52.90     52.96     52.85

SUM host_to_device_bidirectional_memcpy_ce 422.30

Running device_to_host_bidirectional_memcpy_ce.
memcpy CE CPU(row) <-> GPU(column) bandwidth (GB/s)
          0         1         2         3         4         5         6         7
0     34.47     50.50     34.41     34.72     34.29     34.38     34.41     34.68

SUM device_to_host_bidirectional_memcpy_ce 291.85

Running device_to_device_memcpy_read_ce.
memcpy CE GPU(row) -> GPU(column) bandwidth (GB/s)
          0         1         2         3         4         5         6         7
0       N/A    172.16    172.24    171.99    171.88    172.16    172.16    172.16
1    171.99       N/A    172.05    172.05    172.16    171.88    172.16    172.18
2    171.96    172.05       N/A    172.05    172.16    172.16    172.16    171.88
3    172.24    171.99    172.05       N/A    172.13    172.41    172.38    172.24
4    172.07    172.24    172.27    172.16       N/A    172.32    172.32    172.32
5    172.18    172.21    172.29    172.35    172.32       N/A    172.32    172.32
6    172.27    172.24    172.32    172.29    172.32    172.54       N/A    172.32
7    172.07    172.32    172.21    172.32    172.32    172.57    172.32       N/A

SUM device_to_device_memcpy_read_ce 9643.22

Running device_to_device_memcpy_write_ce.
memcpy CE GPU(row) <- GPU(column) bandwidth (GB/s)
          0         1         2         3         4         5         6         7
0       N/A    176.08    176.08    176.08    176.11    176.05    176.08    176.08
1    176.14       N/A    176.14    176.14    176.08    176.08    176.20    176.05
2    176.14    176.11       N/A    176.08    176.05    176.05    176.05    176.11
3    176.14    176.14    176.11       N/A    176.08    176.17    176.17    176.20
4    176.08    176.11    176.02    176.11       N/A    176.14    176.14    176.08
5    176.08    176.17    175.99    176.08    176.08       N/A    176.08    176.11
6    176.11    176.17    176.14    176.08    176.14    176.05       N/A    176.08
7    176.08    176.14    176.11    176.08    176.05    176.11    176.02       N/A

SUM device_to_device_memcpy_write_ce 9861.63

Running device_to_device_bidirectional_memcpy_read_ce.
memcpy CE GPU(row) <-> GPU(column) bandwidth (GB/s)
          0         1         2         3         4         5         6         7
0       N/A    170.54    170.54    170.54    170.54    170.76    170.54    170.76
1    170.54       N/A    170.52    170.54    170.52    170.79    170.54    170.54
2    170.81    170.54       N/A    170.65    170.57    170.62    170.38    170.46
3    170.76    170.54    170.54       N/A    170.62    170.60    170.41    170.44
4    170.52    170.54    170.73    170.76       N/A    170.54    170.54    170.79
5    170.54    170.54    170.79    170.73    170.73       N/A    170.76    170.54
6    170.54    170.76    170.79    170.49    170.54    170.79       N/A    170.54
7    170.54    170.52    170.76    170.54    170.76    170.79    170.79       N/A

SUM device_to_device_bidirectional_memcpy_read_ce 9554.42

Running device_to_device_bidirectional_memcpy_write_ce.
memcpy CE GPU(row) <-> GPU(column) bandwidth (GB/s)
          0         1         2         3         4         5         6         7
0       N/A    174.34    174.25    174.28    174.28    174.31    174.34    174.31
1    174.42       N/A    174.45    174.37    174.45    174.39    174.39    174.34
2    174.39    174.42       N/A    174.31    174.37    174.28    174.37    174.37
3    174.39    174.45    174.45       N/A    174.37    174.42    174.39    174.37
4    174.42    174.39    174.34    174.39       N/A    174.37    174.34    174.45
5    174.37    174.31    174.37    174.37    174.39       N/A    174.34    174.31
6    174.37    174.34    174.34    174.34    174.37    174.34       N/A    174.37
7    174.31    174.34    174.39    174.37    174.34    174.34    174.34       N/A

SUM device_to_device_bidirectional_memcpy_write_ce 9764.26

Running all_to_host_memcpy_ce.
memcpy CE CPU(row) <- GPU(column) bandwidth (GB/s)
          0         1         2         3         4         5         6         7
0     44.95     44.45     45.12     44.41     44.61     45.06     45.06     45.00

SUM all_to_host_memcpy_ce 358.66

Running all_to_host_bidirectional_memcpy_ce.
memcpy CE CPU(row) <- GPU(column) bandwidth (GB/s)
          0         1         2         3         4         5         6         7
0     22.22     35.09     22.43     22.34     22.59     22.75     22.84     22.78

SUM all_to_host_bidirectional_memcpy_ce 193.04

Running host_to_all_memcpy_ce.
memcpy CE CPU(row) -> GPU(column) bandwidth (GB/s)
          0         1         2         3         4         5         6         7
0     55.21     55.26     55.15     55.12     55.12     55.19     55.13     55.11

SUM host_to_all_memcpy_ce 441.30

Running host_to_all_bidirectional_memcpy_ce.
memcpy CE CPU(row) <- GPU(column) bandwidth (GB/s)
          0         1         2         3         4         5         6         7
0     33.49     24.64     33.51     33.56     34.15     34.34     34.36     34.42

SUM host_to_all_bidirectional_memcpy_ce 262.47

Running all_to_one_write_ce.
memcpy CE All Gpus -> GPU(column) total bandwidth (GB/s)
          0         1         2         3         4         5         6         7
0    177.69    177.63    177.71    177.70    177.67    177.68    177.68    177.68

SUM all_to_one_write_ce 1421.43

Running all_to_one_read_ce.
memcpy CE All Gpus <- GPU(column) total bandwidth (GB/s)
          0         1         2         3         4         5         6         7
0    142.09    142.13    142.14    142.13    142.15    142.08    142.09    142.12

SUM all_to_one_read_ce 1136.93

Running one_to_all_write_ce.
memcpy CE GPU(column) -> All GPUs total bandwidth (GB/s)
          0         1         2         3         4         5         6         7
0    175.95    176.03    176.00    176.05    175.99    175.98    175.96    175.99

SUM one_to_all_write_ce 1407.95

Running one_to_all_read_ce.
memcpy CE GPU(column) <- All GPUs total bandwidth (GB/s)
          0         1         2         3         4         5         6         7
0    177.62    177.70    177.68    177.67    177.70    177.70    177.69    177.70

SUM one_to_all_read_ce 1421.47

Running host_to_device_memcpy_sm.
memcpy SM CPU(row) -> GPU(column) bandwidth (GB/s)
          0         1         2         3         4         5         6         7
0     38.20     51.49     38.26     38.16     37.17     37.33     37.45     37.39

SUM host_to_device_memcpy_sm 315.45

Running device_to_host_memcpy_sm.
memcpy SM CPU(row) <- GPU(column) bandwidth (GB/s)
          0         1         2         3         4         5         6         7
0     52.81     52.81     52.82     52.81     52.78     52.79     52.78     52.79

SUM device_to_host_memcpy_sm 422.39

Running device_to_device_memcpy_read_sm.
memcpy CE GPU(row) -> GPU(column) bandwidth (GB/s)
          0         1         2         3         4         5         6         7
0       N/A    167.56    167.58    167.53    167.58    167.63    167.63    167.58
1    167.58       N/A    167.61    167.63    167.66    167.63    167.63    167.56
2    167.58    167.58       N/A    167.56    167.50    167.58    167.61    167.61
3    167.61    167.63    167.61       N/A    167.63    167.58    167.58    167.61
4    167.61    167.58    167.58    167.61       N/A    167.58    167.61    167.56
5    167.56    167.56    167.61    167.50    167.53       N/A    167.61    167.66
6    167.61    167.56    167.61    167.56    167.53    167.66       N/A    167.63
7    167.61    167.56    167.61    167.53    167.56    167.61    167.63       N/A

SUM device_to_device_memcpy_read_sm 9385.10

Running device_to_device_memcpy_write_sm.
memcpy SM GPU(row) <- GPU(column) bandwidth (GB/s)
          0         1         2         3         4         5         6         7
0       N/A    165.44    165.39    165.54    165.31    165.44    165.72    165.49
1    165.82       N/A    165.69    165.54    165.67    165.74    165.62    165.77
2    165.72    165.51       N/A    165.36    165.23    165.33    165.49    165.28
3    165.95    165.82    165.77       N/A    165.62    165.77    165.74    165.87
4    165.90    165.82    165.74    165.51       N/A    165.62    165.72    165.79
5    165.67    165.41    165.26    165.44    165.13       N/A    165.21    165.44
6    165.64    165.49    165.49    165.41    165.49    165.39       N/A    165.51
7    165.74    165.51    165.46    165.41    165.44    165.23    165.54       N/A

SUM device_to_device_memcpy_write_sm 9271.04

Running device_to_device_bidirectional_memcpy_read_sm.
memcpy SM GPU(row) -> GPU(column) bandwidth (GB/s)
          0         1         2         3         4         5         6         7
0       N/A    301.70    301.81    301.70    301.64    301.78    301.76    301.78
1    301.74       N/A    301.76    301.81    301.78    301.74    301.72    301.74
2    301.83    301.68       N/A    301.59    301.66    301.78    301.76    301.81
3    301.81    301.81    301.81       N/A    301.81    301.76    301.78    301.76
4    301.72    301.81    301.76    301.83       N/A    301.72    301.76    301.74
5    301.81    301.68    301.85    301.70    301.64       N/A    301.74    301.76
6    301.78    301.74    301.78    301.70    301.68    301.81       N/A    301.78
7    301.76    301.72    301.76    301.70    301.66    301.74    301.74       N/A

SUM device_to_device_bidirectional_memcpy_read_sm 16898.01

Running device_to_device_bidirectional_memcpy_write_sm.
memcpy SM GPU(row) <- GPU(column) bandwidth (GB/s)
          0         1         2         3         4         5         6         7
0       N/A    327.84    327.47    327.27    327.49    327.56    327.22    327.32
1    328.04       N/A    327.52    327.72    327.77    327.72    327.56    327.54
2    327.49    327.44       N/A    327.97    327.09    327.39    327.14    327.39
3    327.74    327.89    327.77       N/A    327.77    327.77    327.49    327.64
4    327.72    327.92    327.74    327.69       N/A    327.56    327.29    327.74
5    327.37    327.49    327.29    327.59    327.57       N/A    327.39    327.37
6    327.74    327.69    327.32    327.14    327.44    326.82       N/A    327.54
7    327.72    327.46    327.49    327.27    327.62    327.29    327.39       N/A

SUM device_to_device_bidirectional_memcpy_write_sm 18341.61

Running all_to_host_memcpy_sm.
memcpy SM CPU(row) <- GPU(column) bandwidth (GB/s)
          0         1         2         3         4         5         6         7
0     43.66     43.25     44.16     43.36     43.26     43.67     43.65     43.69

SUM all_to_host_memcpy_sm 348.69

Running all_to_host_bidirectional_memcpy_sm.
memcpy SM CPU(row) <- GPU(column) bandwidth (GB/s)
          0         1         2         3         4         5         6         7
0     30.13     41.15     20.38     38.47     35.57     31.79     23.19     32.49

SUM all_to_host_bidirectional_memcpy_sm 253.18

Running host_to_all_memcpy_sm.
memcpy SM CPU(row) -> GPU(column) bandwidth (GB/s)
          0         1         2         3         4         5         6         7
0     34.98     51.55     35.16     34.98     35.21     35.28     35.33     35.33

SUM host_to_all_memcpy_sm 297.82

Running host_to_all_bidirectional_memcpy_sm.
memcpy SM CPU(row) -> GPU(column) bandwidth (GB/s)
          0         1         2         3         4         5         6         7
0     20.87     34.09     21.54     23.16     22.16     26.02     17.97     20.36

SUM host_to_all_bidirectional_memcpy_sm 186.16

Running all_to_one_write_sm.
memcpy SM All Gpus -> GPU(column) total bandwidth (GB/s)
          0         1         2         3         4         5         6         7
0    167.68    167.74    167.81    167.82    167.81    167.82    167.81    167.83

SUM all_to_one_write_sm 1342.33

Running all_to_one_read_sm.
memcpy SM All GPUs <- GPU(column) total bandwidth (GB/s)
          0         1         2         3         4         5         6         7
0    167.64    167.66    167.65    167.67    167.69    167.63    167.65    167.64

SUM all_to_one_read_sm 1341.22

Running one_to_all_write_sm.
memcpy SM GPU(column) -> All GPUs total bandwidth (GB/s)
          0         1         2         3         4         5         6         7
0    146.35    137.24    151.35    146.34    165.73    163.10    151.48    146.06

SUM one_to_all_write_sm 1207.66

Running one_to_all_read_sm.
memcpy SM GPU(column) <- All GPUs total bandwidth (GB/s)
          0         1         2         3         4         5         6         7
0    154.42    154.54    160.02    165.89    139.31    139.18    160.02    149.01

SUM one_to_all_read_sm 1222.39
```

运行特定的测试用例：

```
> ./nvbandwidth -t device_to_device_memcpy_read_ce
nvbandwidth Version: v0.2
Built from Git version: 42e94d2

NOTE: This tool reports current measured bandwidth on your system.
Additional system-specific tuning may be required to achieve maximal peak bandwidth.

CUDA Runtime Version: 11080
CUDA Driver Version: 12000
Driver Version: 525.105.17

Device 0: NVIDIA H800
Device 1: NVIDIA H800
Device 2: NVIDIA H800
Device 3: NVIDIA H800
Device 4: NVIDIA H800
Device 5: NVIDIA H800
Device 6: NVIDIA H800
Device 7: NVIDIA H800

Running device_to_device_memcpy_read_ce.
memcpy CE GPU(row) -> GPU(column) bandwidth (GB/s)
          0         1         2         3         4         5         6         7
0       N/A    172.32    172.57    172.57    172.32    172.21    172.13    172.35
1    172.18       N/A    172.05    172.29    172.16    172.16    172.24    172.24
2    171.99    172.18       N/A    172.29    172.21    172.16    172.16    172.24
3    172.07    172.07    172.27       N/A    172.16    172.16    172.21    172.16
4    172.16    171.88    172.16    172.13       N/A    172.27    172.05    172.32
5    172.10    172.18    172.18    172.16    172.32       N/A    172.05    172.07
6    171.88    172.16    172.16    172.16    172.32    172.05       N/A    172.32
7    172.10    172.10    172.07    172.02    172.27    172.24    172.07       N/A

SUM device_to_device_memcpy_read_ce 9642.08
```


打印详细结果：
```
./nvbandwidth -v >> nvbandwitch.txt
```



## 测试详情

实现了两种类型的 copies：Copy Engine (CE)  或 Steaming Multiprocessor (SM)

CE copies 使用 memcpy API。 SM copies 使用 kernels。

SM copies将截断copy大小以均匀地适合目标设备，从而正确报告带宽。 copy的实际字节大小为：

```
(threadsPerBlock * deviceSMCount) * floor(copySize / (threadsPerBlock * deviceSMCount))
```
threadsPerBlock 设置为 512。

### 测量详情

![image](https://github.com/liguodongiot/llm-action/assets/13220186/29fff486-6f54-4363-a817-4e38d23742b6)

阻塞 kernel 和 CUDA 事件用于测量通过 SM 或 CE 执行 copies 的时间，并根据一系列 copies 计算带宽。

首先，我们将一个spin kernel排入队列，该kernel在主机内存中的标志上旋转。 spin kernel在设备上旋转，直到所有用于测量的事件已完全排队到测量流中。 这确保了排队操作的开销被排除在互连上的实际传输的测量之外。 

接下来，我们将一个开始事件、一次或多次 memcpy 迭代（具体取决于 LoopCount）以及最后一个停止事件排队。 最后，我们释放标志来开始测量。

该过程重复 3 次，并报告每次试验的中值带宽。



### 单向带宽测试

```
Running host_to_device_memcpy_ce.
memcpy CE CPU(row) -> GPU(column) bandwidth (GB/s)
          0         1         2         3         4         5         6         7
0     26.03     25.94     25.97     26.00     26.19     25.95     26.00     25.97
```
单向测试分别测量输出矩阵中每对之间的带宽。 流量不是同时发送的。


### 双向主机 <-> 设备带宽测试

```
Running host_to_device_bidirectional_memcpy_ce.
memcpy CE CPU(row) <-> GPU(column) bandwidth (GB/s)
          0         1         2         3         4         5         6         7
0     18.56     18.37     19.37     19.59     18.71     18.79     18.46     18.61
```

双向主机到设备带宽传输的设置如下所示：

![image](https://github.com/liguodongiot/llm-action/assets/13220186/edabb22a-6fb9-45bd-99f5-7c309fc7267f)

Stream 0（测量流）对设备执行写入操作，而相反方向的interfering stream则产生读取操作。 

这种模式相反，用于测量双向设备到主机的带宽，如下所示。
![image](https://github.com/liguodongiot/llm-action/assets/13220186/c0c18882-1379-43bc-a854-4bb726180bd9)



### 双向设备 <-> 设备带宽测试

双向设备到设备传输的设置如下所示：

![image](https://github.com/liguodongiot/llm-action/assets/13220186/9b1302bd-2b31-4228-9542-3557b579389d)

该测试在两个流上启动流量：在设备 0 上启动的流 0 执行从设备 0 到设备 1 的写入操作，而interference stream 1 启动相反的流量，执行从设备 1 到设备 0 的写入操作。

CE 双向带宽测试计算measured stream上的带宽：

```
CE bidir. bandwidth = (size of data on measured stream) / (time on measured stream)
```

然而，SM 双向测试在源 GPU 和对等 GPU 上作为独立流启动 memcpy kernels，并按如下方式计算带宽：

```
SM bidir. bandwidth = size/(time on stream1) + size/(time on stream2)
```

官方文档：https://github.com/NVIDIA/nvbandwidth
