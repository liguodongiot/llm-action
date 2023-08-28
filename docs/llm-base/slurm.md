

- Slurm简介: http://hmli.ustc.edu.cn/doc/linux/slurm-install/slurm-install.html



## 简介

所有需运行的作业，无论是用于程序调试还是业务计算，都可以通过交互式并行 srun 、批处理式 sbatch 或分配式 salloc 等命令提交，提交后可以利用相关命令查询作业状态。





## 架构


Slurm采用slurmctld服务（守护进程）作为中心管理器用于监测资源和作业，为了提高可用性，还可以配置另一个备份冗余管理器。

各计算节点需启动slurmd守护进程，以便被用于作为远程shell使用：等待作业、执行作业、返回状态、再等待更多作业。

slurmdbd(Slurm DataBase Daemon)数据库守护进程（非必需，建议采用，也可以记录到纯文本中等），可以将多个slurm管理的集群的记账信息记录在同一个数据库中。

还可以启用slurmrestd(Slurm REST API Daemon)服务（非必需），该服务可以通过REST API与Slurm进行交互，所有功能都对应的API。


用户工具包含 
- srun 运行作业、 
- scancel 终止排队中或运行中的作业、 
- sinfo 查看系统状态、 
- squeue 查看作业状态、 
- sacct 查看运行中或结束了的作业及作业步信息等命令。 
- sview 命令可以图形化显示系统和作业状态（可含有网络拓扑）。 
- scontrol 作为管理工具，可以监控、修改集群的配置和状态信息等。
- 用于管理数据库的命令是 sacctmgr ，可认证集群、有效用户、有效记账账户等。







```python
import os
from multiprocessing import Pool, cpu_count

# function you want to run in parallel:
def myfunction(a, b):
  return a + b

# list of tuples to serve as arguments to function:
args = [(1, 2), (9, 11), (6, 2)]

# number of cores you have allocated for your slurm task:
number_of_cores = int(os.environ['SLURM_CPUS_PER_TASK'])

print("number_of_cores: ", number_of_cores)


# number_of_cores = cpu_count() # if not on the cluster you should do this instead

# multiprocssing pool to distribute tasks to:
with Pool(number_of_cores) as pool:
    # distribute computations and collect results:
    results = pool.starmap(myfunction, args)

```




节点：

Head Node：头节点、管理节点、控制节点，运行slurmctld管理服务的节点。

Compute Node：计算节点，运行作业计算任务的节点，需运行slurmd服务。

Login Node：用户登录节点，用于用户登录的节点。

SlurmDBD Node：SlurmDBD节点、SlurmDBD数据库节点，存储调度策略、记账和作业等信息的节点，需运行slurmdbd服务。

客户节点：含计算节点和用户登录节点。


用户:

account：账户，一个账户可以含有多个用户。

user：用户，多个用户可以共享一个账户。

bank account：银行账户，对应机时费等。

资源:

GRES：Generic Resource，通用资源。

TRES：Trackable RESources，可追踪资源。

QOS：Quality of Service，服务质量，作业优先级。

association：关联。可利用其实现，如用户的关联不在数据库中，这将阻止用户运行作业。该选项可以阻止用户访问无效账户。

Partition：队列、分区。用于对计算节点、作业并行规模、作业时长、用户等进行分组管理，以合理分配资源。






