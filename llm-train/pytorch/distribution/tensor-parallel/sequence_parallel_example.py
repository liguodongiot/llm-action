import argparse

import torch
import torch.multiprocessing as mp

from torch.distributed._tensor import DeviceMesh
from torch.distributed.tensor.parallel import parallelize_module
from utils import cleanup, setup, ToyModel

try:
    from torch.distributed.tensor.parallel import (
        SequenceParallel
    )
    SP_AVAILABLE = True
except BaseException as e:
    pass


"""
This is the script to test Sequence Parallel(SP) on a toy model in a
Megetron-LM SPMD style. We show an E2E working flow from forward,
backward and optimization.

We use the example of two `nn.Linear` layers with an element-wise `nn.RELU`
in between to show an example of sequence parallel, which was proposed in paper:

https://arxiv.org/pdf/2205.05198.pdf.

Like tensor parallel, we parallelize the first linear layer by column
and also parallelize the second linear layer by row. But the input in each rank
now is different so that we need one all-gather for input and one reduce-scatter
in the end of the second linear layer.
"""


def demo_sp(rank, args):
    """
    Main body of the demo of a basic version of sequence parallel by using
    PyTorch native APIs.
    """
    print(f"Running SP example on rank {rank}.")
    setup(rank, args.world_size)

    # create a sharding plan based on the given world_size.
    device_mesh = DeviceMesh("cuda", torch.arange(0, args.world_size))

    # create model and move it to GPU with id rank
    model = ToyModel().cuda(rank)
    # Create a optimizer for the parallelized module.
    LR = 0.25
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    # Parallelize the module based on the given Parallel Style.
    model = parallelize_module(model, device_mesh, SequenceParallel())

    # Perform a num of iterations of forward/backward
    # and optimizations for the sharded module.
    for _ in range(args.iter_nums):
        # For SP, input can be different across all ranks.
        inp = torch.rand(20, 10).cuda(rank)
        output = model(inp)
        output.sum().backward()
        optimizer.step()

    cleanup()


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    parser = argparse.ArgumentParser()
    # This is passed in via cmd
    parser.add_argument("--world_size", type=int, default=n_gpus)
    parser.add_argument("--iter_nums", type=int, default=10)
    args = parser.parse_args()
    # The main entry point is called directly without using subprocess
    if n_gpus < 2:
        print("Requires at least 2 GPUs to run.")
    elif not SP_AVAILABLE:
        print(
            "PyTorch doesn't have Sequence Parallelism available,"
            " need nightly build."
        )
    else:
        mp.spawn(demo_sp, args=(args,), nprocs=args.world_size, join=True)

