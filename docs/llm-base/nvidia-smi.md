
# nvidia-smi

## 基本概念
- Tx是发送数据的意思，Rx是接收数据的意思。



## 基本操作

```
nvidia-smi
```

### 可以列出所有GPU设备的详细信息

使用 nvidia-smi -q 可以列出所有GPU设备的详细信息。

如果只想列出某一GPU的详细信息，可使用 -i 选项指定。

```
> nvidia-smi -q -i 0

==============NVSMI LOG==============

Timestamp                                 : Fri Jun 30 17:20:02 2023
Driver Version                            : 525.105.17
CUDA Version                              : 12.0

Attached GPUs                             : 8
GPU 00000000:18:00.0
    Product Name                          : NVIDIA H800
    Product Brand                         : NVIDIA
    Product Architecture                  : Hopper
    Display Mode                          : Disabled
    Display Active                        : Disabled
    Persistence Mode                      : Enabled
    MIG Mode
        Current                           : Disabled
        Pending                           : Disabled
    Accounting Mode                       : Disabled
    Accounting Mode Buffer Size           : 4000
    Driver Model
        Current                           : N/A
        Pending                           : N/A
    Serial Number                         : 1651323018291
    GPU UUID                              : GPU-34bf77d1-c686-6821-79a8-32d326c5039c
    Minor Number                          : 0
    VBIOS Version                         : 96.00.51.00.0E
    MultiGPU Board                        : No
    Board ID                              : 0x1800
    Board Part Number                     : 692-2G520-0205-000
    GPU Part Number                       : 2324-865-A1
    Module ID                             : 5
    Inforom Version
        Image Version                     : G520.0205.00.02
        OEM Object                        : 2.1
        ECC Object                        : 7.16
        Power Management Object           : N/A
    GPU Operation Mode
        Current                           : N/A
        Pending                           : N/A
    GSP Firmware Version                  : 525.105.17
    GPU Virtualization Mode
        Virtualization Mode               : None
        Host VGPU Mode                    : N/A
    IBMNPU
        Relaxed Ordering Mode             : N/A
    PCI
        Bus                               : 0x18
        Device                            : 0x00
        Domain                            : 0x0000
        Device Id                         : 0x232410DE
        Bus Id                            : 00000000:18:00.0
        Sub System Id                     : 0x17A610DE
        GPU Link Info
            PCIe Generation
                Max                       : 5
                Current                   : 5
                Device Current            : 5
                Device Max                : 5
                Host Max                  : 5
            Link Width
                Max                       : 16x
                Current                   : 16x
        Bridge Chip
            Type                          : N/A
            Firmware                      : N/A
        Replays Since Reset               : 0
        Replay Number Rollovers           : 0
        Tx Throughput                     : 123000 KB/s
        Rx Throughput                     : 111000 KB/s
        Atomic Caps Inbound               : N/A
        Atomic Caps Outbound              : N/A
    Fan Speed                             : N/A
    Performance State                     : P0
    Clocks Throttle Reasons
        Idle                              : Active
        Applications Clocks Setting       : Not Active
        SW Power Cap                      : Not Active
        HW Slowdown                       : Not Active
            HW Thermal Slowdown           : Not Active
            HW Power Brake Slowdown       : Not Active
        Sync Boost                        : Not Active
        SW Thermal Slowdown               : Not Active
        Display Clock Setting             : Not Active
    FB Memory Usage
        Total                             : 81559 MiB
        Reserved                          : 516 MiB
        Used                              : 0 MiB
        Free                              : 81042 MiB
    BAR1 Memory Usage
        Total                             : 131072 MiB
        Used                              : 1 MiB
        Free                              : 131071 MiB
    Compute Mode                          : Default
    Utilization
        Gpu                               : 0 %
        Memory                            : 0 %
        Encoder                           : 0 %
        Decoder                           : 0 %
    Encoder Stats
        Active Sessions                   : 0
        Average FPS                       : 0
        Average Latency                   : 0
    FBC Stats
        Active Sessions                   : 0
        Average FPS                       : 0
        Average Latency                   : 0
    Ecc Mode
        Current                           : Disabled
        Pending                           : Disabled
    ECC Errors
        Volatile
            SRAM Correctable              : N/A
            SRAM Uncorrectable            : N/A
            DRAM Correctable              : N/A
            DRAM Uncorrectable            : N/A
        Aggregate
            SRAM Correctable              : N/A
            SRAM Uncorrectable            : N/A
            DRAM Correctable              : N/A
            DRAM Uncorrectable            : N/A
    Retired Pages
        Single Bit ECC                    : N/A
        Double Bit ECC                    : N/A
        Pending Page Blacklist            : N/A
    Remapped Rows
        Correctable Error                 : 0
        Uncorrectable Error               : 0
        Pending                           : No
        Remapping Failure Occurred        : No
        Bank Remap Availability Histogram
            Max                           : 2560 bank(s)
            High                          : 0 bank(s)
            Partial                       : 0 bank(s)
            Low                           : 0 bank(s)
            None                          : 0 bank(s)
    Temperature
        GPU Current Temp                  : 35 C
        GPU T.Limit Temp                  : 51 C
        GPU Shutdown Temp                 : 92 C
        GPU Slowdown Temp                 : 89 C
        GPU Max Operating Temp            : 87 C
        GPU Target Temperature            : N/A
        Memory Current Temp               : 44 C
        Memory Max Operating Temp         : 95 C
    Power Readings
        Power Management                  : Supported
        Power Draw                        : 74.71 W
        Power Limit                       : 700.00 W
        Default Power Limit               : 700.00 W
        Enforced Power Limit              : 700.00 W
        Min Power Limit                   : 200.00 W
        Max Power Limit                   : 700.00 W
    Clocks
        Graphics                          : 345 MHz
        SM                                : 345 MHz
        Memory                            : 2619 MHz
        Video                             : 765 MHz
    Applications Clocks
        Graphics                          : 1980 MHz
        Memory                            : 2619 MHz
    Default Applications Clocks
        Graphics                          : 1980 MHz
        Memory                            : 2619 MHz
    Deferred Clocks
        Memory                            : N/A
    Max Clocks
        Graphics                          : 1980 MHz
        SM                                : 1980 MHz
        Memory                            : 2619 MHz
        Video                             : 1545 MHz
    Max Customer Boost Clocks
        Graphics                          : 1980 MHz
    Clock Policy
        Auto Boost                        : N/A
        Auto Boost Default                : N/A
    Voltage
        Graphics                          : 725.000 mV
    Fabric
        State                             : Completed
        Status                            : Success
    Processes                             : None
```



## nvlink


```
> nvidia-smi nvlink --help

    nvlink -- Display NvLink information.

    Usage: nvidia-smi nvlink [options]

    Options include:
    [-h | --help]: Display help information
    [-i | --id]: Enumeration index, PCI bus ID or UUID.
    [-l | --link]: Limit a command to a specific link.  Without this flag, all link information is displayed.
    [-s | --status]: Display link state (active/inactive).
    [-c | --capabilities]: Display link capabilities.
    [-p | --pcibusid]: Display remote node PCI bus ID for a link.
    [-R | --remotelinkinfo]: Display remote device PCI bus ID and NvLink ID for a link.
    [-sc | --setcontrol]: Setting counter control is deprecated!
    [-gc | --getcontrol]: Getting counter control is deprecated!
    [-g | --getcounters]: Getting counters using option -g is deprecated. Please use option -gt/--getthroughput instead.
    [-r | --resetcounters]: Resetting counters is deprecated!
    [-e | --errorcounters]: Display error counters for a link.
    [-ec | --crcerrorcounters]: Display per-lane CRC error counters for a link.
    [-re | --reseterrorcounters]: Reset all error counters to zero.
    [-gt | --getthroughput]: Display link throughput counters for specified counter type
       The arguments consist of character string representing the type of traffic counted:
          d: Display tx and rx data payload in KiB
          r: Display tx and rx data payload and protocol overhead in KiB if supported

    [-sLowPwrThres | --setLowPowerThreshold]: Set NvLink Low Power Threshold (value inunits of 100us/default)
    [-gLowPwrInfo | --getLowPowerInfo]: Get NvLink Low Power Info
    [-cBridge | --checkBridge]: Check NvLink Bridge presence

```


### 查看系统/GPU 拓扑和 NVLink

```
# 查看系统/GPU 拓扑
> nvidia-smi topo --matrix
        GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7  CPU Affinity    NUMA Affinity
GPU0     X      NV8     NV8     NV8     NV8     NV8     NV8     NV8   -31,64-95       0
GPU1    NV8      X      NV8     NV8     NV8     NV8     NV8     NV8   -31,64-95       0
GPU2    NV8     NV8      X      NV8     NV8     NV8     NV8     NV8   -31,64-95       0
GPU3    NV8     NV8     NV8      X      NV8     NV8     NV8     NV8   -31,64-95       0
GPU4    NV8     NV8     NV8     NV8      X      NV8     NV8     NV8   32-63,96-127    1
GPU5    NV8     NV8     NV8     NV8     NV8      X      NV8     NV8   32-63,96-127    1
GPU6    NV8     NV8     NV8     NV8     NV8     NV8      X      NV8   32-63,96-127    1
GPU7    NV8     NV8     NV8     NV8     NV8     NV8     NV8      X    2-63,96-127     1
NIC0    NODE    PIX     NODE    NODE    SYS     SYS     SYS     SYS
NIC1    NODE    NODE    NODE    PIX     SYS     SYS     SYS     SYS
NIC2    SYS     SYS     SYS     SYS     NODE    NODE    NODE    PIX
NIC3    SYS     SYS     SYS     SYS     NODE    NODE    NODE    PIX

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect beI)
  NODE = Connection traversing PCIe as well as the interconnect betweeUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typ
  PXB  = Connection traversing multiple PCIe bridges (without traversi
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks

NIC Legend:

  NIC0: mlx5_0
  NIC1: mlx5_1
  NIC2: mlx5_2
  NIC3: mlx5_3         
```

### 状态

```
# 查询 NVLink 连接本身以确保状态、功能和运行状况。
> nvidia-smi nvlink --status
GPU 0: NVIDIA H800 (UUID: GPU-34bf77d1-c686-6821-79a8-32d326c5039c)
         Link 0: 26.562 GB/s
         Link 1: 26.562 GB/s
         Link 2: 26.562 GB/s
         Link 3: 26.562 GB/s
         Link 4: 26.562 GB/s
         Link 5: 26.562 GB/s
         Link 6: 26.562 GB/s
         Link 7: 26.562 GB/s
GPU 1: NVIDIA H800 (UUID: GPU-f5046fa5-3db4-45e8-870a-dc1376becaa5)
         Link 0: 26.562 GB/s
         Link 1: 26.562 GB/s
         Link 2: 26.562 GB/s
         Link 3: 26.562 GB/s
         Link 4: 26.562 GB/s
         Link 5: 26.562 GB/s
         Link 6: 26.562 GB/s
         Link 7: 26.562 GB/s
...
GPU 7: NVIDIA H800 (UUID: GPU-474d838c-171f-d249-4f45-bbc01a8eb74a)
         Link 0: 26.562 GB/s
         Link 1: 26.562 GB/s
         Link 2: 26.562 GB/s
         Link 3: 26.562 GB/s
         Link 4: 26.562 GB/s
         Link 5: 26.562 GB/s
         Link 6: 26.562 GB/s
         Link 7: 26.562 GB/s
```

```
# 查询 NVLink 连接本身以确保状态、功能和运行状况。
> nvidia-smi nvlink --capabilities
GPU 0: NVIDIA H800 (UUID: GPU-34bf77d1-c686-6821-79a8-32d326c5039c)
         Link 0, P2P is supported: true
         Link 0, Access to system memory supported: true
         Link 0, P2P atomics supported: true
         Link 0, System memory atomics supported: true
         Link 0, SLI is supported: true
         Link 0, Link is supported: false
         Link 1, P2P is supported: true
         Link 1, Access to system memory supported: true
         Link 1, P2P atomics supported: true
         Link 1, System memory atomics supported: true
         Link 1, SLI is supported: true
         Link 1, Link is supported: false
         Link 2, P2P is supported: true
         Link 2, Access to system memory supported: true
         Link 2, P2P atomics supported: true
         Link 2, System memory atomics supported: true
         Link 2, SLI is supported: true
         Link 2, Link is supported: false
         Link 3, P2P is supported: true
         Link 3, Access to system memory supported: true
         Link 3, P2P atomics supported: true
         Link 3, System memory atomics supported: true
         Link 3, SLI is supported: true
         Link 3, Link is supported: false
         Link 4, P2P is supported: true
         Link 4, Access to system memory supported: true
         Link 4, P2P atomics supported: true
         Link 4, System memory atomics supported: true
         Link 4, SLI is supported: true
         Link 4, Link is supported: false
         Link 5, P2P is supported: true
         Link 5, Access to system memory supported: true
         Link 5, P2P atomics supported: true
         Link 5, System memory atomics supported: true
         Link 5, SLI is supported: true
         Link 5, Link is supported: false
         Link 6, P2P is supported: true
         Link 6, Access to system memory supported: true
         Link 6, P2P atomics supported: true
         Link 6, System memory atomics supported: true
         Link 6, SLI is supported: true
         Link 6, Link is supported: false
         Link 7, P2P is supported: true
         Link 7, Access to system memory supported: true
         Link 7, P2P atomics supported: true
         Link 7, System memory atomics supported: true
         Link 7, SLI is supported: true
         Link 7, Link is supported: false
GPU 1: NVIDIA H800 (UUID: GPU-f5046fa5-3db4-45e8-870a-dc1376becaa5)
         Link 0, P2P is supported: true
         Link 0, Access to system memory supported: true
         Link 0, P2P atomics supported: true
         Link 0, System memory atomics supported: true
         Link 0, SLI is supported: true
         Link 0, Link is supported: false
         Link 1, P2P is supported: true
         Link 1, Access to system memory supported: true
         Link 1, P2P atomics supported: true
         Link 1, System memory atomics supported: true
         Link 1, SLI is supported: true
         Link 1, Link is supported: false
         Link 2, P2P is supported: true
         Link 2, Access to system memory supported: true
         Link 2, P2P atomics supported: true
         Link 2, System memory atomics supported: true
         Link 2, SLI is supported: true
         Link 2, Link is supported: false
         Link 3, P2P is supported: true
         Link 3, Access to system memory supported: true
         Link 3, P2P atomics supported: true
         Link 3, System memory atomics supported: true
         Link 3, SLI is supported: true
         Link 3, Link is supported: false
         Link 4, P2P is supported: true
         Link 4, Access to system memory supported: true
         Link 4, P2P atomics supported: true
         Link 4, System memory atomics supported: true
         Link 4, SLI is supported: true
         Link 4, Link is supported: false
         Link 5, P2P is supported: true
         Link 5, Access to system memory supported: true
         Link 5, P2P atomics supported: true
         Link 5, System memory atomics supported: true
         Link 5, SLI is supported: true
         Link 5, Link is supported: false
         Link 6, P2P is supported: true
         Link 6, Access to system memory supported: true
         Link 6, P2P atomics supported: true
         Link 6, System memory atomics supported: true
         Link 6, SLI is supported: true
         Link 6, Link is supported: false
         Link 7, P2P is supported: true
         Link 7, Access to system memory supported: true
         Link 7, P2P atomics supported: true
         Link 7, System memory atomics supported: true
         Link 7, SLI is supported: true
         Link 7, Link is supported: false
...
GPU 7: NVIDIA H800 (UUID: GPU-474d838c-171f-d249-4f45-bbc01a8eb74a)
         Link 0, P2P is supported: true
         Link 0, Access to system memory supported: true
         Link 0, P2P atomics supported: true
         Link 0, System memory atomics supported: true
         Link 0, SLI is supported: true
         Link 0, Link is supported: false
         Link 1, P2P is supported: true
         Link 1, Access to system memory supported: true
         Link 1, P2P atomics supported: true
         Link 1, System memory atomics supported: true
         Link 1, SLI is supported: true
         Link 1, Link is supported: false
         Link 2, P2P is supported: true
         Link 2, Access to system memory supported: true
         Link 2, P2P atomics supported: true
         Link 2, System memory atomics supported: true
         Link 2, SLI is supported: true
         Link 2, Link is supported: false
         Link 3, P2P is supported: true
         Link 3, Access to system memory supported: true
         Link 3, P2P atomics supported: true
         Link 3, System memory atomics supported: true
         Link 3, SLI is supported: true
         Link 3, Link is supported: false
         Link 4, P2P is supported: true
         Link 4, Access to system memory supported: true
         Link 4, P2P atomics supported: true
         Link 4, System memory atomics supported: true
         Link 4, SLI is supported: true
         Link 4, Link is supported: false
         Link 5, P2P is supported: true
         Link 5, Access to system memory supported: true
         Link 5, P2P atomics supported: true
         Link 5, System memory atomics supported: true
         Link 5, SLI is supported: true
         Link 5, Link is supported: false
         Link 6, P2P is supported: true
         Link 6, Access to system memory supported: true
         Link 6, P2P atomics supported: true
         Link 6, System memory atomics supported: true
         Link 6, SLI is supported: true
         Link 6, Link is supported: false
         Link 7, P2P is supported: true
         Link 7, Access to system memory supported: true
         Link 7, P2P atomics supported: true
         Link 7, System memory atomics supported: true
         Link 7, SLI is supported: true
         Link 7, Link is supported: false
```




### 显示指定计数器类型的链路吞吐量计数器

显示指定计数器类型的链路吞吐量计数器 参数由表示计数的流量类型的字符串组成：

- d：以 KiB 显示 tx 和 rx 数据有效负载
- r：如果支持，以 KiB 显示 tx 和 rx 数据有效负载和协议开销


```
# nvidia-smi nvlink -gt d
> nvidia-smi nvlink -gt r
nvidia-smi nvlink -gt r
GPU 0: NVIDIA H800 (UUID: GPU-34bf77d1-c686-6821-79a8-32d326c5039c)
         Link 0: Raw Tx: 25305739 KiB
         Link 0: Raw Rx: 25554089 KiB
         Link 1: Raw Tx: 8508772 KiB
         Link 1: Raw Rx: 8727784 KiB
         Link 2: Raw Tx: 8511892 KiB
         Link 2: Raw Rx: 8734905 KiB
         Link 3: Raw Tx: 8506576 KiB
         Link 3: Raw Rx: 8720965 KiB
         Link 4: Raw Tx: 8508545 KiB
         Link 4: Raw Rx: 8724705 KiB
         Link 5: Raw Tx: 8503240 KiB
         Link 5: Raw Rx: 8724306 KiB
         Link 6: Raw Tx: 8514361 KiB
         Link 6: Raw Rx: 8727629 KiB
         Link 7: Raw Tx: 8517607 KiB
         Link 7: Raw Rx: 8730994 KiB
GPU 1: NVIDIA H800 (UUID: GPU-f5046fa5-3db4-45e8-870a-dc1376becaa5)
         Link 0: Raw Tx: 25415406 KiB
         Link 0: Raw Rx: 25651111 KiB
         Link 1: Raw Tx: 8614105 KiB
         Link 1: Raw Rx: 8827532 KiB
         Link 2: Raw Tx: 8618737 KiB
         Link 2: Raw Rx: 8838429 KiB
         Link 3: Raw Tx: 8617599 KiB
         Link 3: Raw Rx: 8828039 KiB
         Link 4: Raw Tx: 8619360 KiB
         Link 4: Raw Rx: 8848343 KiB
         Link 5: Raw Tx: 8616291 KiB
         Link 5: Raw Rx: 8830590 KiB
         Link 6: Raw Tx: 8621649 KiB
         Link 6: Raw Rx: 8828622 KiB
         Link 7: Raw Tx: 8626449 KiB
         Link 7: Raw Rx: 8834243 KiB
GPU 2: NVIDIA H800 (UUID: GPU-9de407ad-ba9c-af12-ce09-65828829a67c)
         Link 0: Raw Tx: 576438247 KiB
         Link 0: Raw Rx: 591715253 KiB
         Link 1: Raw Tx: 591684300 KiB
         Link 1: Raw Rx: 625042437 KiB
         Link 2: Raw Tx: 561660326 KiB
         Link 2: Raw Rx: 576223159 KiB
         Link 3: Raw Tx: 560468850 KiB
         Link 3: Raw Rx: 575052692 KiB
         Link 4: Raw Tx: 542530345 KiB
         Link 4: Raw Rx: 583254750 KiB
         Link 5: Raw Tx: 541527825 KiB
         Link 5: Raw Rx: 582292048 KiB
         Link 6: Raw Tx: 524762470 KiB
         Link 6: Raw Rx: 550225067 KiB
         Link 7: Raw Tx: 544014147 KiB
         Link 7: Raw Rx: 582742696 KiB
...
GPU 6: NVIDIA H800 (UUID: GPU-70c5b9a8-82a3-4199-d7f5-adb9186459eb)
         Link 0: Raw Tx: 25421053 KiB
         Link 0: Raw Rx: 25651584 KiB
         Link 1: Raw Tx: 8624284 KiB
         Link 1: Raw Rx: 8834680 KiB
         Link 2: Raw Tx: 8629640 KiB
         Link 2: Raw Rx: 8841618 KiB
         Link 3: Raw Tx: 8622803 KiB
         Link 3: Raw Rx: 8834635 KiB
         Link 4: Raw Tx: 8625793 KiB
         Link 4: Raw Rx: 8848148 KiB
         Link 5: Raw Tx: 8617230 KiB
         Link 5: Raw Rx: 8832897 KiB
         Link 6: Raw Tx: 8629904 KiB
         Link 6: Raw Rx: 8848920 KiB
         Link 7: Raw Tx: 8637145 KiB
         Link 7: Raw Rx: 8831474 KiB
GPU 7: NVIDIA H800 (UUID: GPU-474d838c-171f-d249-4f45-bbc01a8eb74a)
         Link 0: Raw Tx: 25310096 KiB
         Link 0: Raw Rx: 25546907 KiB
         Link 1: Raw Tx: 8512574 KiB
         Link 1: Raw Rx: 8747250 KiB
         Link 2: Raw Tx: 8514172 KiB
         Link 2: Raw Rx: 8734000 KiB
         Link 3: Raw Tx: 8512622 KiB
         Link 3: Raw Rx: 8730435 KiB
         Link 4: Raw Tx: 8510112 KiB
         Link 4: Raw Rx: 8745800 KiB
         Link 5: Raw Tx: 8507350 KiB
         Link 5: Raw Rx: 8737562 KiB
         Link 6: Raw Tx: 8515623 KiB
         Link 6: Raw Rx: 8729950 KiB
         Link 7: Raw Tx: 8520721 KiB
         Link 7: Raw Rx: 8731188 KiB
```

显示 0 号 GPU 设备的所有 链路 的 吞吐量计数器的数据有效载荷
```
# 未做任何操作时
> nvidia-smi nvlink -i 0 -gt d
GPU 0: NVIDIA H800 (UUID: GPU-34bf77d1-c686-6821-79a8-32d326c5039c)
         Link 0: Data Tx: 5863784 KiB
         Link 0: Data Rx: 5863677 KiB
         Link 1: Data Tx: 5864084 KiB
         Link 1: Data Rx: 5862999 KiB
         Link 2: Data Tx: 5864630 KiB
         Link 2: Data Rx: 5863940 KiB
         Link 3: Data Tx: 5863784 KiB
         Link 3: Data Rx: 5861984 KiB
         Link 4: Data Tx: 5861984 KiB
         Link 4: Data Rx: 5862774 KiB
         Link 5: Data Tx: 5861984 KiB
         Link 5: Data Rx: 5862696 KiB
         Link 6: Data Tx: 5861984 KiB
         Link 6: Data Rx: 5862999 KiB
         Link 7: Data Tx: 5861984 KiB
         Link 7: Data Rx: 5863146 KiB

# 模型训练时（LLaMA-13B）
> nvidia-smi nvlink -i 0 -gt d
GPU 0: NVIDIA H800 (UUID: GPU-34bf77d1-c686-6821-79a8-32d326c5039c)
         Link 0: Data Tx: 1390774681 KiB
         Link 0: Data Rx: 1387831436 KiB
         Link 1: Data Tx: 1390715554 KiB
         Link 1: Data Rx: 1387856699 KiB
         Link 2: Data Tx: 1390689916 KiB
         Link 2: Data Rx: 1387846800 KiB
         Link 3: Data Tx: 1390772616 KiB
         Link 3: Data Rx: 1387795114 KiB
         Link 4: Data Tx: 1391305436 KiB
         Link 4: Data Rx: 1387910526 KiB
         Link 5: Data Tx: 1391288579 KiB
         Link 5: Data Rx: 1387888125 KiB
         Link 6: Data Tx: 1391348992 KiB
         Link 6: Data Rx: 1387832695 KiB
         Link 7: Data Tx: 1391348007 KiB
         Link 7: Data Rx: 1387855953 KiB
> nvidia-smi nvlink -i 0 -gt r
GPU 0: NVIDIA H800 (UUID: GPU-34bf77d1-c686-6821-79a8-32d326c5039c)
         Link 0: Raw Tx: 1933426555 KiB
         Link 0: Raw Rx: 1975423057 KiB
         Link 1: Raw Tx: 1915132335 KiB
         Link 1: Raw Rx: 1958569530 KiB
         Link 2: Raw Tx: 1916865102 KiB
         Link 2: Raw Rx: 1958463156 KiB
         Link 3: Raw Tx: 1916412075 KiB
         Link 3: Raw Rx: 1958028986 KiB
         Link 4: Raw Tx: 1913329166 KiB
         Link 4: Raw Rx: 1957374521 KiB
         Link 5: Raw Tx: 1913784453 KiB
         Link 5: Raw Rx: 1957230286 KiB
         Link 6: Raw Tx: 1916726453 KiB
         Link 6: Raw Rx: 1957614726 KiB
         Link 7: Raw Tx: 1919300185 KiB
         Link 7: Raw Rx: 1957241622 KiB

```


## 参考文档
- [nvidia-smi命令详解和一些高阶技巧讲解](https://www.inte.net/news/270918.html)
