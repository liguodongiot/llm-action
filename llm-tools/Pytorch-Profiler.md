



##  示例

- base-profiler.py
- profiler-recipe.py
- tensorboard-profiler.py




## PyTorch 分析器
```

PyTorch 分析器是通过上下文管理器启用的，它接受许多参数，其中一些最有用的是：

activities - 要分析的活动列表：

ProfilerActivity.CPU - PyTorch 运算符、TorchScript 函数和用户定义的代码标签;

ProfilerActivity.CUDA - 设备上的 CUDA 内核;

ProfilerActivity.XPU - 设备上的 XPU 内核;

record_shapes - 是否记录运算符输入的形状;

profile_memory - 是否报告模型的 Tensor 消耗的内存量;


```



## 使用 Profiler 记录执行事件


```
with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet18'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
) as prof:
    for step, batch_data in enumerate(train_loader):
        prof.step()  # Need to call this at each step to notify profiler of steps' boundary.
        if step >= 1 + 1 + 3:
            break
        train(batch_data)
```



分析器通过上下文管理器启用，并接受多个参数，其中一些最有用的参数是：

schedule - 可调用，它将步骤 （int） 作为单个参数，并返回在每个步骤中要执行的 Profiler 作。

在此示例中，Profiler wait=1, warmup=1, active=3, repeat=1 

将跳过第一步/迭代，开始预热第二步，记录以下三次迭代，之后跟踪将变为可用，并调用 on_trace_ready（如果设置）。该循环总共重复一次。
每个循环在 TensorBoard 插件中称为 “span”。


在等待步骤中，分析器处于禁用状态。
在预热步骤中，分析器开始跟踪，但结果将被丢弃。这是为了减少性能分析开销。性
能分析开始时的开销很高，很容易给性能分析结果带来偏差。
在active步骤期间，分析器会工作并记录事件。


on_trace_ready - 在每个周期结束时调用的可调用对象; 在此示例中, torch.profiler.tensorboard_trace_handler ，我们使用为 TensorBoard 生成结果文件。分析后，结果文件将保存到 ./log/resnet18 目录中。将此目录指定为 logdir 参数，用于分析 TensorBoard 中的配置文件。

record_shapes — 是否记录运算符输入的形状。

profile_memory - 跟踪张量内存分配/释放。注意，对于旧版本的 pytorch 版本低于 1.10，如果性能分析时间过长，请禁用它或升级到新版本。

with_stack - 记录ops的源信息（文件和行号）。如果在 VS Code （ 参考 ） 中启动 TensorBoard，则单击堆栈帧将导航到特定代码行。


也支持以下非上下文管理器 start/stop。

```
prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet18'),
        record_shapes=True,
        with_stack=True)


prof.start()
for step, batch_data in enumerate(train_loader):
    prof.step()
    if step >= 1 + 1 + 3:
        break
    train(batch_data)
prof.stop()

```




## 跟踪长时间运行的作业

PyTorch 分析器提供了一个额外的 API 来处理长时间运行的作业（例如训练循环）。

跟踪所有执行可能会很慢，并导致跟踪文件非常大。为避免这种情况，请使用可选参数：

- schedule - 指定一个函数，该函数将整数参数（步骤编号）作为输入并返回 Profiler 的action，使用此参数的最佳方法是使用 torch.profiler.schedule 辅助函数，该函数可以为您生成schedule。

- on_trace_ready - 指定一个函数，该函数将对分析器的引用作为输入，并在每次新跟踪准备就绪时由分析器调用。



torch.profiler.schedule 辅助函数：

```
from torch.profiler import schedule

my_schedule = schedule(
    skip_first=10,
    wait=5,
    warmup=1,
    active=3,
    repeat=2)

```

Profiler 假定长时间运行的作业由步骤组成，这些步骤从零开始编号。

上面的示例为性能分析器定义了以下作序列：

参数 skip_first 告诉分析器它应该忽略前 10 个步骤（skip_first 的默认值为零）;

在前 skip_first 个步骤之后，分析器开始执行分析器循环;

每个周期包括三个阶段：

Idling （wait=5 steps），在此阶段 Profiler 未激活;

预热（Warmup=1 steps），在此阶段，Profiler 开始跟踪，但结果被丢弃; 此阶段用于丢弃 Profiler 在跟踪开始时获得的样本，因为它们通常会因额外的开销而产生偏差;

active 跟踪 （ active = 3 steps），在此阶段，Profiler 跟踪并记录数据;

可选的 repeat 参数指定循环数的上限。默认情况下（零值），只要作业运行，分析器就会执行循环。


因此，在上面的示例中，profiler 将跳过前 15 个步骤，将下一步用于热身，active记录接下来的 3 个步骤，再跳过 5 个步骤，将下一步用于热身，主动记录另外 3 个步骤。由于指定了 repeat=2 参数值，因此性能分析器将在前两个周期后停止记录。

```
sort_by_keyword = "self_" + device + "_time_total"

def trace_handler(p):
    output = p.key_averages().table(sort_by=sort_by_keyword, row_limit=10)
    print(output)
    p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")

with profile(
    activities=activities,
    schedule=torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=2),
    on_trace_ready=trace_handler
) as p:
    for idx in range(8):
        model(inputs)
        p.step()
```













