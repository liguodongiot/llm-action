


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
  -e  --field-id   fieldId     Field identifier to view/inject.
  -l  --list                  List to look up the long names, short names and
                               field ids.
  -h  --help                  Displays usage information and exits.
  -i  --gpu-id     gpuId       The comma separated list of GPU/GPU-I/GPU-CI IDs
                               to run the dmon on. Default is -1 which runs for
                               all supported GPU. Run dcgmi discovery -c to
                               check list of available GPU entities
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


