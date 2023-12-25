

- https://github.com/NVIDIA/nccl-tests
- https://cloud.baidu.com/doc/GPU/s/Yl3mr0ren




要构建测试，只需键入make。

如果CUDA未安装在/usr/local/cuda中，则指定CUDA_HOME。同样，如果NCCL未安装在/usr中，则指定NCCL_HOME。

```
$ make CUDA_HOME=/path/to/cuda NCCL_HOME=/path/to/nccl
```

NCCL tests rely on MPI to work on multiple processes, hence multiple nodes. If you want to compile the tests with MPI support, you need to set MPI=1 and set MPI_HOME to the path where MPI is installed.

NCCL tests 依赖于MPI来处理多个进程，从而处理多个节点。如果要使用支持MPI编译 tests，则需要将MPI设置为1，并将MPI_HOME设置为安装MPI的路径。


```
$ make MPI=1 MPI_HOME=/path/to/mpi CUDA_HOME=/path/to/cuda NCCL_HOME=/path/to/nccl
```







