
# nvidia-smi


## 基本操作

```
nvidia-smi
```



## nvlink

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



## 参考文档
- [nvidia-smi命令详解和一些高阶技巧讲解](https://www.inte.net/news/270918.html)
