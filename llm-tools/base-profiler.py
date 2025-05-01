

import torch
import numpy as np
from torch import nn
import torch.autograd.profiler as profiler



class MyModule(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(MyModule, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

    def forward(self, input, mask):
        with profiler.record_function("LINEAR PASS"):
            out = self.linear(input)

        with profiler.record_function("MASK INDICES"):
            threshold = out.sum(axis=1).mean().item()


            # forward(13)中，通过 aten::copy_ 运算符将 mask 复制到 CPU，以便它可以使用 NumPy 的 argwhere 函数。
            hi_idx = np.argwhere(mask.cpu().numpy() > threshold)
            # aten::copy_ 在 forward(13) 将数组作为张量复制回 CUDA。
            hi_idx = torch.from_numpy(hi_idx).cuda()

        return out, hi_idx


# 分析前向传递

model = MyModule(500, 10).cuda()
input = torch.rand(128, 500).cuda()
mask = torch.rand((500, 500, 500), dtype=torch.double).cuda()

# warm-up
model(input, mask)

with profiler.profile(with_stack=True, profile_memory=True) as prof:
    out, idx = model(input, mask)        



# 打印性能分析器结果

print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))



"""
(Some columns are omitted)

-------------  ------------  ------------  ------------  ---------------------------------
         Name    Self CPU %      Self CPU  Self CPU Mem   Source Location
-------------  ------------  ------------  ------------  ---------------------------------
 MASK INDICES        87.88%        5.212s    -953.67 Mb  /mnt/xarfuse/.../torch/au
                                                         <ipython-input-...>(10): forward
                                                         /mnt/xarfuse/.../torch/nn
                                                         <ipython-input-...>(9): <module>
                                                         /mnt/xarfuse/.../IPython/

  aten::copy_        12.07%     715.848ms           0 b  <ipython-input-...>(12): forward
                                                         /mnt/xarfuse/.../torch/nn
                                                         <ipython-input-...>(9): <module>
                                                         /mnt/xarfuse/.../IPython/
                                                         /mnt/xarfuse/.../IPython/

  LINEAR PASS         0.01%     350.151us         -20 b  /mnt/xarfuse/.../torch/au
                                                         <ipython-input-...>(7): forward
                                                         /mnt/xarfuse/.../torch/nn
                                                         <ipython-input-...>(9): <module>
                                                         /mnt/xarfuse/.../IPython/

  aten::addmm         0.00%     293.342us           0 b  /mnt/xarfuse/.../torch/nn
                                                         /mnt/xarfuse/.../torch/nn
                                                         /mnt/xarfuse/.../torch/nn
                                                         <ipython-input-...>(8): forward
                                                         /mnt/xarfuse/.../torch/nn

   aten::mean         0.00%     235.095us           0 b  <ipython-input-...>(11): forward
                                                         /mnt/xarfuse/.../torch/nn
                                                         <ipython-input-...>(9): <module>
                                                         /mnt/xarfuse/.../IPython/
                                                         /mnt/xarfuse/.../IPython/

-----------------------------  ------------  ---------- ----------------------------------
Self CPU time total: 5.931s

"""




# mask 使用 torch.double 数据类型初始化。通过将其转换为 torch.float 来减少内存占用。


model = MyModule(500, 10).cuda()
input = torch.rand(128, 500).cuda()
mask = torch.rand((500, 500, 500), dtype=torch.float).cuda()

# warm-up
model(input, mask)

with profiler.profile(with_stack=True, profile_memory=True) as prof:
    out, idx = model(input, mask)

print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))

"""
(Some columns are omitted)

-----------------  ------------  ------------  ------------  --------------------------------
             Name    Self CPU %      Self CPU  Self CPU Mem   Source Location
-----------------  ------------  ------------  ------------  --------------------------------
     MASK INDICES        93.61%        5.006s    -476.84 Mb  /mnt/xarfuse/.../torch/au
                                                             <ipython-input-...>(10): forward
                                                             /mnt/xarfuse/  /torch/nn
                                                             <ipython-input-...>(9): <module>
                                                             /mnt/xarfuse/.../IPython/

      aten::copy_         6.34%     338.759ms           0 b  <ipython-input-...>(12): forward
                                                             /mnt/xarfuse/.../torch/nn
                                                             <ipython-input-...>(9): <module>
                                                             /mnt/xarfuse/.../IPython/
                                                             /mnt/xarfuse/.../IPython/

 aten::as_strided         0.01%     281.808us           0 b  <ipython-input-...>(11): forward
                                                             /mnt/xarfuse/.../torch/nn
                                                             <ipython-input-...>(9): <module>
                                                             /mnt/xarfuse/.../IPython/
                                                             /mnt/xarfuse/.../IPython/

      aten::addmm         0.01%     275.721us           0 b  /mnt/xarfuse/.../torch/nn
                                                             /mnt/xarfuse/.../torch/nn
                                                             /mnt/xarfuse/.../torch/nn
                                                             <ipython-input-...>(8): forward
                                                             /mnt/xarfuse/.../torch/nn

      aten::_local        0.01%     268.650us           0 b  <ipython-input-...>(11): forward
      _scalar_dense                                          /mnt/xarfuse/.../torch/nn
                                                             <ipython-input-...>(9): <module>
                                                             /mnt/xarfuse/.../IPython/
                                                             /mnt/xarfuse/.../IPython/

-----------------  ------------  ------------  ------------  --------------------------------
Self CPU time total: 5.347s

"""

class MyModule(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(MyModule, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

    def forward(self, input, mask):
        with profiler.record_function("LINEAR PASS"):
            out = self.linear(input)

        with profiler.record_function("MASK INDICES"):
            threshold = out.sum(axis=1).mean()
            # 使用 torch 函数 nonzero() 代替
            hi_idx = (mask > threshold).nonzero(as_tuple=True)

        return out, hi_idx


model = MyModule(500, 10).cuda()
input = torch.rand(128, 500).cuda()
mask = torch.rand((500, 500, 500), dtype=torch.float).cuda()

# warm-up
model(input, mask)

with profiler.profile(with_stack=True, profile_memory=True) as prof:
    out, idx = model(input, mask)

print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))

"""
(Some columns are omitted)

--------------  ------------  ------------  ------------  ---------------------------------
          Name    Self CPU %      Self CPU  Self CPU Mem   Source Location
--------------  ------------  ------------  ------------  ---------------------------------
      aten::gt        57.17%     129.089ms           0 b  <ipython-input-...>(12): forward
                                                          /mnt/xarfuse/.../torch/nn
                                                          <ipython-input-...>(25): <module>
                                                          /mnt/xarfuse/.../IPython/
                                                          /mnt/xarfuse/.../IPython/

 aten::nonzero        37.38%      84.402ms           0 b  <ipython-input-...>(12): forward
                                                          /mnt/xarfuse/.../torch/nn
                                                          <ipython-input-...>(25): <module>
                                                          /mnt/xarfuse/.../IPython/
                                                          /mnt/xarfuse/.../IPython/

   INDEX SCORE         3.32%       7.491ms    -119.21 Mb  /mnt/xarfuse/.../torch/au
                                                          <ipython-input-...>(10): forward
                                                          /mnt/xarfuse/.../torch/nn
                                                          <ipython-input-...>(25): <module>
                                                          /mnt/xarfuse/.../IPython/

aten::as_strided         0.20%    441.587us          0 b  <ipython-input-...>(12): forward
                                                          /mnt/xarfuse/.../torch/nn
                                                          <ipython-input-...>(25): <module>
                                                          /mnt/xarfuse/.../IPython/
                                                          /mnt/xarfuse/.../IPython/

 aten::nonzero
     _numpy             0.18%     395.602us           0 b  <ipython-input-...>(12): forward
                                                          /mnt/xarfuse/.../torch/nn
                                                          <ipython-input-...>(25): <module>
                                                          /mnt/xarfuse/.../IPython/
                                                          /mnt/xarfuse/.../IPython/
--------------  ------------  ------------  ------------  ---------------------------------
Self CPU time total: 225.801ms

"""


