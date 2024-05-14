import argparse

import torch
import torch.multiprocessing as mp

from torch.distributed._tensor import DeviceMesh
from torch.distributed.tensor.parallel import PairwiseParallel, parallelize_module
from utils import cleanup, setup, ToyModel


"""
This is the script to test Tensor Parallel(TP) on a toy model in a
Megetron-LM SPMD style. We show an E2E working flow from forward,
backward and optimization.

More context about API designs can be found in the design:

https://github.com/pytorch/pytorch/issues/89884.

And it is built on top of Distributed Tensor which is proposed in:

https://github.com/pytorch/pytorch/issues/88838.

We use the example of two `nn.Linear` layers with an element-wise `nn.RELU`
in between to show an example of Megatron-LM, which was proposed in paper:

https://arxiv.org/abs/1909.08053.

The basic idea is that we parallelize the first linear layer by column
and also parallelize the second linear layer by row so that we only need
one all reduce in the end of the second linear layer.

We can speed up the model training by avoiding communications between
two layers.


To parallelize a nn module, we need to specify what parallel style we want
to use and our `parallelize_module` API will parse and parallelize the modules
based on the given `ParallelStyle`. We are using this PyTorch native Tensor
Parallelism APIs in this example to show users how to use them.
"""


def demo_tp(rank, args):
    """
    Main body of the demo of a basic version of tensor parallel by using
    PyTorch native APIs.
    """
    print(f"Running basic Megatron style TP example on rank : {rank} , world_size : {args.world_size}. ")
    setup(rank, args.world_size)

    # create a sharding plan based on the given world_size.
    device_mesh = DeviceMesh("cuda", torch.arange(0, args.world_size))

    # create model and move it to GPU with id rank
    model = ToyModel().cuda(rank)
    # Create a optimizer for the parallelized module.
    LR = 0.25
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    # Parallelize the module based on the given Parallel Style.
    model = parallelize_module(model, device_mesh, PairwiseParallel())

    # Perform a num of iterations of forward/backward
    # and optimizations for the sharded module.
    for i in range(args.iter_nums):
        # For TP, input needs to be same across all TP ranks.
        # Setting the random seed is to mimic the behavior of dataloader.
        if rank==0:
            print(f"-----------{i}--------------")
        torch.manual_seed(i)
        inp = torch.rand(20, 10).cuda(rank)
        if rank==0:
            print(f"rank: {rank} , input shape: {inp.shape}")
        output = model(inp)
        if rank==0:
            print(f"rank: {rank} , input shape: {output.shape}")
        output.sum().backward()
        optimizer.step()
        #print("-------------------------")

    cleanup()


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    parser = argparse.ArgumentParser()
    # This is passed in via cmd
    parser.add_argument("--world_size", type=int, default=n_gpus)
    parser.add_argument("--iter_nums", type=int, default=10000)
    args = parser.parse_args()
    # The main entry point is called directly without using subprocess
    if n_gpus < 2:
        print("Requires at least 2 GPUs to run.")
    else:
        mp.spawn(demo_tp, args=(args,), nprocs=args.world_size, join=True)


