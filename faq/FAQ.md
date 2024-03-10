

## FAQ


### baichuan2报错

- 'BitsAndBytesConfig' object is not subscriptable

https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat/discussions/2



- AttributeError: 'BaichuanTokenizer' object has no attribute 'sp_model'

降低版本到4.34.0及以下 ： pip install transformers==4.34.0





- AttributeError: 'Parameter' object has no attribute 'ds_status'

使用deepspeed + transformers 全量微调时报错
deepspeed 使用的zero3
eval的时候报错；错误详情：
AttributeError: 'Parameter' object has no attribute 'ds_status'

https://github.com/baichuan-inc/Baichuan2/issues/215




### chatglm3


- NotImplementedError: Cannot copy out of meta tensor; no data! 

在使用 DeepSpeed 进行分布式训练时，DeepSpeed 会将模型分割成多个分块并分配给不同的 GPU 进行训练。在这个过程中，DeepSpeed 会使用 PyTorch 的 DistributedDataParallel 包装器来实现分布式训练，而 DistributedDataParallel 需要对模型进行初始化。
当您使用 AutoModel.from_pretrained() 方法加载预训练模型时，模型权重会被存储在 PyTorch 的 nn.Parameter 对象中。在没有指定 empty_init=False 参数时，nn.Parameter 对象的值将被初始化为全零的张量。但是，由于 nn.Parameter 对象不是真正的张量，而是具有元数据的张量包装器，因此无法将这些对象直接复制到 DeepSpeed 使用的元数据张量中。
在指定 empty_init=False 参数后，nn.Parameter 对象将被初始化为包含预训练权重的张量，这使得 DeepSpeed 能够正常地将权重复制到元数据张量中，从而避免了 NotImplementedError: Cannot copy out of meta tensor; no data! 错误的出现。
综上所述，您遇到的问题是因为在使用 DeepSpeed 进行分布式训练时，模型权重的初始化方式与普通的训练方式不同，因此需要通过指定 empty_init=False 参数来解决。

NotImplementedError: Cannot copy out of meta tensor; no data! 这个错误通常是由于Deepspeed在使用自定义权重初始化时出现问题，而这些初始化可能需要从先前的训练中加载权重。如果在使用Deepspeed进行分布式训练时出现此错误，则需要在初始化模型时指定empty_init=False，以便在加载权重之前，权重矩阵不会被初始化为空。
AutoModel.from_pretrained是Hugging Face Transformers库中的一个方法，用于从预训练模型中加载权重。在Deepspeed分布式训练中，模型的初始化和权重加载可能需要特殊处理，因此需要使用empty_init=False参数来指定在加载权重之前不要将权重矩阵初始化为空。
在其他模型中，可能不需要这个参数是因为它们的初始化和权重加载不需要特殊处理，或者因为它们的代码已经进行了相应的修改以适应Deepspeed的分布式训练流程。


https://github.com/THUDM/ChatGLM-6B/issues/530




## Deepspeed



- Unsupported gpu architecture 'compute_89'（使用deepspeed zero3 offload 在RTX 4090 上遇到问题）

使用的 CUDA 编译器版本不支持指定的 GPU 架构。

参考1：

https://github.com/microsoft/DeepSpeed/issues/3488

```
torch_adam: true

The torch.optim.Adam works fine for cpu offloading.

```

参考2：

原文链接：https://blog.csdn.net/rellvera/article/details/130337185

```
解决方法
有两种解决方案：

1.更新 CUDA 版本：检查当前 CUDA 版本是否支持 GPU 架构。如果不支持，更新CUDA版本即可。
2.更改 GPU 架构：也就是降低CPU的算力水平。可以在编译命令中使用 -arch 参数来指定目标架构。例如，-arch=compute_75。

我选择了第一种解决方案，即更新cuda版本。我的GPU为4090，算力是’compute_89’，经查，将CUDA升级到11.8以上版本，即可解决该问题。
```





## Pytorch

- RuntimeError: DataLoader worker (pid xxxxx) is killed by signal: Killed.

方案一：

可能共享内存太小了

--shm-size 4G 


方案二：

https://blog.csdn.net/wjinjie/article/details/129733252

```
通过设置num_workers，DataLoader实例可以使用多少个子进程进行数据加载，从而加快网络的训练过程。
默认情况下，num_workers值被设置为0，0值代表告诉加载器在主进程内部加载数据。
但是num_workers并不能无限制设置的很大，因为这和你的机器硬件性能也有关。


最简单的办法，就是将num_workers设置的小一点；

---(最终解决)----  如果还是有问题，可以直接将num_workers设置成默认值0；

当然，也可以通过增加机器内存来尝试解决。
```


- Pytorch dataloader 错误 “DataLoader worker (pid xxx) is killed by signal” 解决方法：https://cloud.tencent.com/developer/article/2066826







