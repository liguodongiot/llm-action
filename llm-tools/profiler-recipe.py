# pip install torch torchvision

import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity


model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)


with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(inputs)


print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))


# ---------------------------------  ------------  ------------  ------------  ------------
#                              Name      Self CPU     CPU total  CPU time avg    # of Calls
# ---------------------------------  ------------  ------------  ------------  ------------
#                   model_inference       5.509ms      57.503ms      57.503ms             1
#                      aten::conv2d     231.000us      31.931ms       1.597ms            20
#                 aten::convolution     250.000us      31.700ms       1.585ms            20
#                aten::_convolution     336.000us      31.450ms       1.573ms            20
#          aten::mkldnn_convolution      30.838ms      31.114ms       1.556ms            20
#                  aten::batch_norm     211.000us      14.693ms     734.650us            20
#      aten::_batch_norm_impl_index     319.000us      14.482ms     724.100us            20
#           aten::native_batch_norm       9.229ms      14.109ms     705.450us            20
#                        aten::mean     332.000us       2.631ms     125.286us            21
#                      aten::select       1.668ms       2.292ms       8.988us           255
# ---------------------------------  ------------  ------------  ------------  ------------
# Self CPU time total: 57.549m


# 要获得更精细粒度的结果并包含运算符输入形状
print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))


# ---------------------------------  ------------  -------------------------------------------
#                              Name     CPU total                                 Input Shapes
# ---------------------------------  ------------  -------------------------------------------
#                   model_inference      57.503ms                                           []
#                      aten::conv2d       8.008ms      [5,64,56,56], [64,64,3,3], [], ..., []]
#                 aten::convolution       7.956ms     [[5,64,56,56], [64,64,3,3], [], ..., []]
#                aten::_convolution       7.909ms     [[5,64,56,56], [64,64,3,3], [], ..., []]
#          aten::mkldnn_convolution       7.834ms     [[5,64,56,56], [64,64,3,3], [], ..., []]
#                      aten::conv2d       6.332ms    [[5,512,7,7], [512,512,3,3], [], ..., []]
#                 aten::convolution       6.303ms    [[5,512,7,7], [512,512,3,3], [], ..., []]
#                aten::_convolution       6.273ms    [[5,512,7,7], [512,512,3,3], [], ..., []]
#          aten::mkldnn_convolution       6.233ms    [[5,512,7,7], [512,512,3,3], [], ..., []]
#                      aten::conv2d       4.751ms  [[5,256,14,14], [256,256,3,3], [], ..., []]
# ---------------------------------  ------------  -------------------------------------------
# Self CPU time total: 57.549ms


# Profiler 还可用于分析在 GPU 和 XPU 上执行的模型的性能

if torch.cuda.is_available():
    device = 'cuda'
elif torch.xpu.is_available():
    device = 'xpu'
else:
    print('Neither CUDA nor XPU devices are available to demonstrate profiling on acceleration devices')
    import sys
    sys.exit(0)

activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA, ProfilerActivity.XPU]
sort_by_keyword = device + "_time_total"

model = models.resnet18().to(device)
inputs = torch.randn(5, 3, 224, 224).to(device)

with profile(activities=activities, record_shapes=True) as prof:
    with record_function("model_inference"):
        model(inputs)

print(prof.key_averages().table(sort_by=sort_by_keyword, row_limit=10))


# -------------------------------------------------------  ------------  ------------
#                                                    Name     Self CUDA    CUDA total
# -------------------------------------------------------  ------------  ------------
#                                         model_inference       0.000us      11.666ms
#                                            aten::conv2d       0.000us      10.484ms
#                                       aten::convolution       0.000us      10.484ms
#                                      aten::_convolution       0.000us      10.484ms
#                              aten::_convolution_nogroup       0.000us      10.484ms
#                                       aten::thnn_conv2d       0.000us      10.484ms
#                               aten::thnn_conv2d_forward      10.484ms      10.484ms
# void at::native::im2col_kernel<float>(long, float co...       3.844ms       3.844ms
#                                       sgemm_32x32x32_NN       3.206ms       3.206ms
#                                   sgemm_32x32x32_NN_vec       3.093ms       3.093ms
# -------------------------------------------------------  ------------  ------------
# Self CPU time total: 23.015ms
# Self CUDA time total: 11.666ms




##########################################################################################

# PyTorch 分析器还可以显示在执行模型算子期间分配（或释放）的内存量（由模型的张量使用）。

model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)

with profile(activities=[ProfilerActivity.CPU],
        profile_memory=True, record_shapes=True) as prof:
    model(inputs)


# ‘self’ memory 对应于运算符分配 （释放） 的内存，不包括对其他运算符的 children 调用。

print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

# (omitting some columns)
# ---------------------------------  ------------  ------------  ------------
#                              Name       CPU Mem  Self CPU Mem    # of Calls
# ---------------------------------  ------------  ------------  ------------
#                       aten::empty      94.79 Mb      94.79 Mb           121
#     aten::max_pool2d_with_indices      11.48 Mb      11.48 Mb             1
#                       aten::addmm      19.53 Kb      19.53 Kb             1
#               aten::empty_strided         572 b         572 b            25
#                     aten::resize_         240 b         240 b             6
#                         aten::abs         480 b         240 b             4
#                         aten::add         160 b         160 b            20
#               aten::masked_select         120 b         112 b             1
#                          aten::ne         122 b          53 b             6
#                          aten::eq          60 b          30 b             2
# ---------------------------------  ------------  ------------  ------------
# Self CPU time total: 53.064ms

print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))


# ---------------------------------  ------------  ------------  ------------
#                              Name       CPU Mem  Self CPU Mem    # of Calls
# ---------------------------------  ------------  ------------  ------------
#                       aten::empty      94.79 Mb      94.79 Mb           121
#                  aten::batch_norm      47.41 Mb           0 b            20
#      aten::_batch_norm_impl_index      47.41 Mb           0 b            20
#           aten::native_batch_norm      47.41 Mb           0 b            20
#                      aten::conv2d      47.37 Mb           0 b            20
#                 aten::convolution      47.37 Mb           0 b            20
#                aten::_convolution      47.37 Mb           0 b            20
#          aten::mkldnn_convolution      47.37 Mb           0 b            20
#                  aten::max_pool2d      11.48 Mb           0 b             1
#     aten::max_pool2d_with_indices      11.48 Mb      11.48 Mb             1
# ---------------------------------  ------------  ------------  ------------
# Self CPU time total: 53.064ms



# 性能分析结果可以输出为 .json 跟踪文件： 跟踪 CUDA 或 XPU 内核 
# 用户可以在 cpu、cuda 和 xpu 之间切换


device = 'cuda'

activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA, ProfilerActivity.XPU]

model = models.resnet18().to(device)
inputs = torch.randn(5, 3, 224, 224).to(device)

with profile(activities=activities) as prof:
    model(inputs)

prof.export_chrome_trace("trace.json")




# Profiler 可用于分析 Python 和 TorchScript 堆栈跟踪

sort_by_keyword = "self_" + device + "_time_total"

with profile(
    activities=activities,
    with_stack=True,
) as prof:
    model(inputs)

# Print aggregated stats
print(prof.key_averages(group_by_stack_n=5).table(sort_by=sort_by_keyword, row_limit=2))

# -------------------------  -----------------------------------------------------------
#                      Name  Source Location
# -------------------------  -----------------------------------------------------------
# aten::thnn_conv2d_forward  .../torch/nn/modules/conv.py(439): _conv_forward
#                            .../torch/nn/modules/conv.py(443): forward
#                            .../torch/nn/modules/module.py(1051): _call_impl
#                            .../site-packages/torchvision/models/resnet.py(63): forward
#                            .../torch/nn/modules/module.py(1051): _call_impl
# aten::thnn_conv2d_forward  .../torch/nn/modules/conv.py(439): _conv_forward
#                            .../torch/nn/modules/conv.py(443): forward
#                            .../torch/nn/modules/module.py(1051): _call_impl
#                            .../site-packages/torchvision/models/resnet.py(59): forward
#                            .../torch/nn/modules/module.py(1051): _call_impl
# -------------------------  -----------------------------------------------------------
# Self CPU time total: 34.016ms
# Self CUDA time total: 11.659ms














