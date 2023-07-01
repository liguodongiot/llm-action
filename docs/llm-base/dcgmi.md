


下表列出了不同 GPU 产品上支持的功能。

| Feature Group | Tesla | Titan | Quadro | GeForce |
| --- | --- | --- | --- | --- |
| Field Value Watches (GPU metrics) | X | X | X | X |
| Configuration Management | X | X | X | X |
| Active Health Checks (GPU subsystems) | X | X | X | X |
| Job Statistics | X | X | X | X |
| Topology | X | X | X | X |
| Introspection | X | X | X | X |
| Policy Notification | X | | | |
| GPU Diagnostics (Diagnostic Levels - 1, 2, 3) | All Levels | Level 1 | Level 1 | Level 1 |

## dcgmi discovery

验证是否能够找到 GPU 设备

```
> dcgmi discovery -l
8 GPUs found.
+--------+----------------------------------------------------------------------+
| GPU ID | Device Information                                                   |
+--------+----------------------------------------------------------------------+
| 0      | Name: NVIDIA H800                                                    |
|        | PCI Bus ID: 00000000:18:00.0                                         |
|        | Device UUID: GPU-34bf77d1-c686-6821-79a8-32d326c5039c                |
+--------+----------------------------------------------------------------------+
| 1      | Name: NVIDIA H800                                                    |
|        | PCI Bus ID: 00000000:3E:00.0                                         |
|        | Device UUID: GPU-f5046fa5-3db4-45e8-870a-dc1376becaa5                |
+--------+----------------------------------------------------------------------+
| 2      | Name: NVIDIA H800                                                    |
|        | PCI Bus ID: 00000000:51:00.0                                         |
|        | Device UUID: GPU-9de407ad-ba9c-af12-ce09-65828829a67c                |
+--------+----------------------------------------------------------------------+
| 3      | Name: NVIDIA H800                                                    |
|        | PCI Bus ID: 00000000:65:00.0                                         |
|        | Device UUID: GPU-b54d703a-dee5-a9da-aeb9-465003acdd4b                |
+--------+----------------------------------------------------------------------+
| 4      | Name: NVIDIA H800                                                    |
|        | PCI Bus ID: 00000000:98:00.0                                         |
|        | Device UUID: GPU-09c6e33a-ffcf-b330-e68b-e1e9f745eae6                |
+--------+----------------------------------------------------------------------+
| 5      | Name: NVIDIA H800                                                    |
|        | PCI Bus ID: 00000000:BD:00.0                                         |
|        | Device UUID: GPU-9a8ef0b8-9816-459d-fa13-cda74cf19d37                |
+--------+----------------------------------------------------------------------+
| 6      | Name: NVIDIA H800                                                    |
|        | PCI Bus ID: 00000000:CF:00.0                                         |
|        | Device UUID: GPU-70c5b9a8-82a3-4199-d7f5-adb9186459eb                |
+--------+----------------------------------------------------------------------+
| 7      | Name: NVIDIA H800                                                    |
|        | PCI Bus ID: 00000000:E2:00.0                                         |
|        | Device UUID: GPU-474d838c-171f-d249-4f45-bbc01a8eb74a                |
+--------+----------------------------------------------------------------------+
0 NvSwitches found.
+-----------+
| Switch ID |
+-----------+
+-----------+
```


## dcgmi group

```
dcgmi group -l
+-------------------+----------------------------------------------------------+
| GROUPS                                                                       |
| 2 groups found.                                                              |
+===================+==========================================================+
| Groups            |                                                          |
| -> 0              |                                                          |
|    -> Group ID    | 0                                                        |
|    -> Group Name  | DCGM_ALL_SUPPORTED_GPUS                                  |
|    -> Entities    | GPU 0, GPU 1, GPU 2, GPU 3, GPU 4, GPU 5, GPU 6, GPU 7   |
| -> 1              |                                                          |
|    -> Group ID    | 1                                                        |
|    -> Group Name  | DCGM_ALL_SUPPORTED_NVSWITCHES                            |
|    -> Entities    | None                                                     |
+-------------------+----------------------------------------------------------+

```



## dcgmi dmon

用于监控 GPU 及其统计数据
```
> dcgmi dmon --help

 dmon -- Used to monitor GPUs and their stats.

Usage: dcgmi dmon
   dcgmi dmon -i <gpuId> -g <groupId> -f <fieldGroupId> -e <fieldId> -d
        <delay> -c <count> -l

Flags:
      --host       IP/FQDN    Connects to specified IP or fully-qualified domain
                               name. To connect to a host engine that was
                               started with -d (unix socket), prefix the unix
                               socket filename with 'unix://'. [default =
                               localhost]
  -f  --field-group-idfieldGroupId  The field group to query on the specified
                               host.
  -e  --field-id   fieldId     Field identifier to view/inject.（要查看的字段ID）
  -l  --list                  List to look up the long names, short names and
                               field ids.
  -h  --help                  Displays usage information and exits.
  -i  --gpu-id     gpuId       The comma separated list of GPU/GPU-I/GPU-CI IDs
                               to run the dmon on. Default is -1 which runs for
                               all supported GPU. Run dcgmi discovery -c to
                               check list of available GPU entities （用于运行守护程序的 GPU/GPU-I/GPU-CI ID 的逗号分隔列表。 默认值为 -1，适用于所有支持的 GPU。 运行 dcgmi discovery -c 以检查可用 GPU 实体列表）
  -g  --group-id   groupId     The group to query on the specified host.
  -d  --delay      delay       In milliseconds. Integer representing how often
                               to query results from DCGM and print them for all
                               of the entities. [default = 1000 msec,  Minimum
                               value = 1 msec.]
  -c  --count      count       Integer representing How many times to loop
                               before exiting. [default- runs forever.]
  --  --ignore_rest           Ignores the rest of the labeled arguments
                               following this flag.


NVIDIA Datacenter GPU Management Interface
```

## dcgmi nvlink
用于获取系统中 GPU 和 NvSwitch 的 NvLink 链接状态或错误计数

```
> dcgmi nvlink --help

 nvlink -- Used to get NvLink link status or error counts for GPUs and
 NvSwitches in the system

 NVLINK Error description
 =========================
 CRC FLIT Error => Data link receive flow control digit CRC error.
 CRC Data Error => Data link receive data CRC error.
 Replay Error   => Data link transmit replay error.
 Recovery Error => Data link transmit recovery error.

Usage: dcgmi nvlink
   dcgmi nvlink --host <IP/FQDN> -g <gpuId> -e -j
   dcgmi nvlink --host <IP/FQDN> -s

Flags:
      --host       IP/FQDN    Connects to specified IP or fully-qualified domain
                               name. To connect to a host engine that was
                               started with -d (unix socket), prefix the unix
                               socket filename with 'unix://'. [default =
                               localhost]
  -e  --errors                Print NvLink errors for a given gpuId (-g).
  -s  --link-status           Print NvLink link status for all GPUs and
                               NvSwitches in the system.
  -h  --help                  Displays usage information and exits.
  -g  --gpuid      gpuId      The GPU ID to query. Required for -e
  -j  --json                  Print the output in a json format
  --  --ignore_rest           Ignores the rest of the labeled arguments
                               following this flag.

NVIDIA Datacenter GPU Management Interface
```




## 指标
支持以下新的设备级分析指标。 列出了定义和相应的 DCGM 字段 ID。 

默认情况下，DCGM 以 1Hz（每 1000毫秒(ms)）的采样率提供指标。 用户可以以任何可配置的频率（最小为 100 毫秒(ms)）从 DCGM 查询指标（例如：dcgmi dmon -d）。


以下是设备水平（level）的GPU指标
| Metric | Definition | DCGM Field Name (DCGM_FI_*) and ID |
| --- | --- | --- |
| Graphics Engine Activity | The fraction of time any portion of the graphics or compute engines were active. The graphics engine is active if a graphics/compute context is bound and the graphics/compute pipe is busy. The value represents an average over a time interval and is not an instantaneous value. | PROF_GR_ENGINE_ACTIVE (ID: 1001) |
| SM Activity | The fraction of time at least one warp was active on a multiprocessor, averaged over all multiprocessors. Note that “active” does not necessarily mean a warp is actively computing. For instance, warps waiting on memory requests are considered active. The value represents an average over a time interval and is not an instantaneous value. A value of 0.8 or greater is necessary, but not sufficient, for effective use of the GPU. A value less than 0.5 likely indicates ineffective GPU usage.Given a simplified GPU architectural view, if a GPU has N SMs then a kernel using N blocks that runs over the entire time interval will correspond to an activity of 1 (100%). A kernel using N/5 blocks that runs over the entire time interval will correspond to an activity of 0.2 (20%). A kernel using N blocks that runs over one fifth of the time interval, with the SMs otherwise idle, will also have an activity of 0.2 (20%). The value is insensitive to the number of threads per block (see `DCGM_FI_PROF_SM_OCCUPANCY`). | PROF_SM_ACTIVE (ID: 1002) |
| SM Occupancy | The fraction of resident warps on a multiprocessor, relative to the maximum number of concurrent warps supported on a multiprocessor. The value represents an average over a time interval and is not an instantaneous value. Higher occupancy does not necessarily indicate better GPU usage. For GPU memory bandwidth limited workloads (see `DCGM_FI_PROF_DRAM_ACTIVE`), higher occupancy is indicative of more effective GPU usage. However if the workload is compute limited (i.e. not GPU memory bandwidth or latency limited), then higher occupancy does not necessarily correlate with more effective GPU usage.Calculating occupancy is not simple and depends on factors such as the GPU properties, the number of threads per block, registers per thread, and shared memory per block. Use the [CUDA Occupancy Calculator](https://docs.nvidia.com/cuda/cuda-occupancy-calculator/index.html) to explore various occupancy scenarios. | PROF_SM_OCCUPANCY (ID: 1003) |
| Tensor Activity | The fraction of cycles the tensor (HMMA / IMMA) pipe was active. The value represents an average over a time interval and is not an instantaneous value. Higher values indicate higher utilization of the Tensor Cores. An activity of 1 (100%) is equivalent to issuing a tensor instruction every other cycle for the entire time interval. An activity of 0.2 (20%) could indicate 20% of the SMs are at 100% utilization over the entire time period, 100% of the SMs are at 20% utilization over the entire time period, 100% of the SMs are at 100% utilization for 20% of the time period, or any combination in between (see `DCGM_FI_PROF_SM_ACTIVE` to help disambiguate these possibilities). | PROF_PIPE_TENSOR_ACTIVE (ID: 1004) |
| FP64 Engine Activity | The fraction of cycles the FP64 (double precision) pipe was active. The value represents an average over a time interval and is not an instantaneous value. Higher values indicate higher utilization of the FP64 cores. An activity of 1 (100%) is equivalent to a FP64 instruction on [every SM every fourth cycle](https://docs.nvidia.com/cuda/volta-tuning-guide/index.html#sm-scheduling) on Volta over the entire time interval. An activity of 0.2 (20%) could indicate 20% of the SMs are at 100% utilization over the entire time period, 100% of the SMs are at 20% utilization over the entire time period, 100% of the SMs are at 100% utilization for 20% of the time period, or any combination in between (see DCGM_FI_PROF_SM_ACTIVE to help disambiguate these possibilities). | PROF_PIPE_FP64_ACTIVE (ID: 1006) |
| FP32 Engine Activity | The fraction of cycles the FMA (FP32 (single precision), and integer) pipe was active. The value represents an average over a time interval and is not an instantaneous value. Higher values indicate higher utilization of the FP32 cores. An activity of 1 (100%) is equivalent to a FP32 instruction every other cycle over the entire time interval. An activity of 0.2 (20%) could indicate 20% of the SMs are at 100% utilization over the entire time period, 100% of the SMs are at 20% utilization over the entire time period, 100% of the SMs are at 100% utilization for 20% of the time period, or any combination in between (see `DCGM_FI_PROF_SM_ACTIVE` to help disambiguate these possibilities). | PROF_PIPE_FP32_ACTIVE (ID: 1007) |
| FP16 Engine Activity | The fraction of cycles the FP16 (half precision) pipe was active. The value represents an average over a time interval and is not an instantaneous value. Higher values indicate higher utilization of the FP16 cores. An activity of 1 (100%) is equivalent to a FP16 instruction every other cycle over the entire time interval. An activity of 0.2 (20%) could indicate 20% of the SMs are at 100% utilization over the entire time period, 100% of the SMs are at 20% utilization over the entire time period, 100% of the SMs are at 100% utilization for 20% of the time period, or any combination in between (see `DCGM_FI_PROF_SM_ACTIVE` to help disambiguate these possibilities). | PROF_PIPE_FP16_ACTIVE (ID: 1008) |
| Memory BW Utilization | The fraction of cycles where data was sent to or received from device memory. The value represents an average over a time interval and is not an instantaneous value. Higher values indicate higher utilization of device memory. An activity of 1 (100%) is equivalent to a DRAM instruction every cycle over the entire time interval (in practice a peak of ~0.8 (80%) is the maximum achievable). An activity of 0.2 (20%) indicates that 20% of the cycles are reading from or writing to device memory over the time interval. | PROF_DRAM_ACTIVE (ID: 1005) |
| NVLink Bandwidth | 通过 NVLink 传输/接收的数据速率（不包括协议头(protocol headers)），以字节/秒为单位。 该值表示一段时间间隔内的平均值，而不是瞬时值。 该速率是时间间隔内的平均值。 例如，如果 1 秒内传输 1 GB 数据，则无论数据以恒定速率还是突发传输，速率均为 1 GB/s。 NVLink Gen2 的理论最大带宽为每个链路每个方向 25 GB/s。 | PROF_NVLINK_TX_BYTES (1011) and PROF_NVLINK_RX_BYTES (1012) |
| PCIe Bandwidth | The rate of data transmitted / received over the PCIe bus, including both protocol headers and data payloads, in bytes per second. The value represents an average over a time interval and is not an instantaneous value. The rate is averaged over the time interval. For example, if 1 GB of data is transferred over 1 second, the rate is 1 GB/s regardless of the data transferred at a constant rate or in bursts. The theoretical maximum PCIe Gen3 bandwidth is 985 MB/s per lane. | PROF_PCIE_[T\|R]X_BYTES (ID: 1009 (TX); 1010 (RX)) |





