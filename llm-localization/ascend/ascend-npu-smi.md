



```
npu-smi info -t device-share -i 1
```



## GPU/Memory 使用率
```
 npu-smi info -t common -i 1
        NPU ID                         : 1
        Chip Count                     : 1

        Chip ID                        : 0
        Memory Usage Rate(%)           : 0
        HBM Usage Rate(%)              : 91
        Aicore Usage Rate(%)           : 6
        Aicore Freq(MHZ)               : 1800
        Aicore curFreq(MHZ)            : 1800
        Aicore Count                   : 20
        Temperature(C)                 : 46
        NPU Real-time Power(W)         : 130.4

        Chip Name                      : mcu
        Temperature(C)                 : 40

npu-smi info -t common -i 7
        NPU ID                         : 7
        Chip Count                     : 1

        Chip ID                        : 0
        Memory Usage Rate(%)           : 0
        HBM Usage Rate(%)              : 82
        Aicore Usage Rate(%)           : 70
        Aicore Freq(MHZ)               : 1800
        Aicore curFreq(MHZ)            : 1800
        Aicore Count                   : 20
        Temperature(C)                 : 66
        NPU Real-time Power(W)         : 331.1

        Chip Name                      : mcu
        Temperature(C)                 : 58


```




## 查看和修改网卡配置

```
hccn_tool -i 3 -status -g

Netdev status:Settings for eth3:
        Supported ports: [ FIBRE ]
        Supported link modes:   40000baseCR4/Full
                                40000baseSR4/Full
                                40000baseLR4/Full
                                25000baseCR/Full
                                25000baseSR/Full
                                50000baseCR2/Full
                                100000baseSR4/Full
                                100000baseCR4/Full
                                100000baseLR4_ER4/Full
                                50000baseSR2/Full
                                1000baseX/Full
                                10000baseCR/Full
                                10000baseSR/Full
                                10000baseLR/Full
                                50000baseLR_ER_FR/Full
                                200000baseSR4/Full
                                200000baseLR4_ER4_FR4/Full
                                200000baseCR4/Full
        Supported pause frame use: Symmetric
        Supports auto-negotiation: No
        Supported FEC modes: None        RS
        Advertised link modes:  Not reported
        Advertised pause frame use: No
        Advertised auto-negotiation: No
        Advertised FEC modes: None       RS
        Speed: 200000Mb/s
        Duplex: Full
        Auto-negotiation: off
        Port: Direct Attach Copper
        PHYAD: 0
        Transceiver: internal
        Current message level: 0x00000036 (54)
                               probe link ifdown ifup
        Link detected: yes

```

## 查看网卡的 IP 地址和路由

```
> hccn_tool -i 3 -ip -g

ipaddr:10.20.11.24
netmask:255.255.255.0

> hccn_tool -i 3 -route -g
Routing table:
Destination     Gateway         Genmask         Flags Metric Ref    Use Iface
10.20.11.0      *               255.255.255.0   U     0      0        0 eth3
127.0.0.1       *               255.255.255.255 UH    0      0        0 lo
192.168.1.0     *               255.255.255.0   U     0      0        0 end3v0
192.168.2.0     *               255.255.255.0   U     0      0        0 end3v0


```

RDMA 网卡的启动配置其实在配置文件
```
> cat /etc/hccn.conf # RDMA 网卡 0-7 的配置
address_0=10.20.11.11
netmask_0=255.255.255.0
address_1=10.20.11.12
netmask_1=255.255.255.0
address_2=10.20.11.13
netmask_2=255.255.255.0
address_3=10.20.11.14
netmask_3=255.255.255.0
address_4=10.20.11.15
netmask_4=255.255.255.0
address_5=10.20.11.16
netmask_5=255.255.255.0
address_6=10.20.11.17
netmask_6=255.255.255.0
address_7=10.20.11.18
netmask_7=255.255.255.0

```


## RDMA ping

```
> hccn_tool -i 3 -ping -g address 10.20.11.16
device 3 PING 10.20.11.16
recv seq=0,time=0.137000ms
recv seq=1,time=0.046000ms
recv seq=2,time=0.058000ms
3 packets transmitted, 3 received, 0.00% packet loss

```





