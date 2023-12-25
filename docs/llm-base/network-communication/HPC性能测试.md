



- HPC-单机&多机点对点RDMA网络性能测试：https://www.volcengine.com/docs/6419/164863


```
apt update && apt install -y infiniband-diags




yum install infiniband-diags




ibstatus


```


单机测试：

```
dpkg -l perftest ibverbs-providers libibumad3 libibverbs1 libnl-3-200 libnl-route-3-200 librdmacm1



ib_write_bw -d mlx5_1 &



ib_write_bw -d mlx5_1 127.0.0.1 --report_gbits
```




多机测试：

```
进行多机测试时，请确认两台实例已加入同一高性能计算集群。

在 A 实例中执行ib_write_bw -d mlx5_1 -x 3命令。


ib_write_bw -d mlx5_1 -x 3



在 B 实例中输入如下命令，<MACHINE_A_HOST> 请替换为 A 实例的 RDMA 网卡 IP，本文以名为mlx5_1的RDMA网卡为例。

ib_write_bw -d mlx5_1 -x 3 <MACHINE_A_HOST> --report_gbits

回显如下，带宽值接近 200Gb/s。

```







- https://github.com/linux-rdma/infiniband-diags


- 【分布式】入门级NCCL多机并行实践 - 02：https://blog.csdn.net/u013013023/article/details/133950028



- 通过 RDMA 网络加速训练：https://www.volcengine.com/docs/6459/96563



- 验证镜像是否支持 RDMA：https://www.volcengine.com/docs/6459/119595#centos-2



