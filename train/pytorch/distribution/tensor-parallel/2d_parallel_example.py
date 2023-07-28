import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.distributed._tensor import DeviceMesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor.parallel import (
    PairwiseParallel,
    parallelize_module,
)
from torch.distributed.tensor.parallel.fsdp import enable_2d_with_fsdp

from utils import cleanup, setup, ToyModel
try:
    from torch.distributed.tensor.parallel import (
        SequenceParallel
    )
    SP_AVAILABLE = True
except BaseException as e:
    pass


"""
This is the script to test 2D Parallel which combines Tensor/Sequence
parallel with Fully Sharded Data Parallel (TP/SP + FSDP) on a toy model
in the SPMD style. We show an E2E working flow from forward, backward
and optimization.

We enabled Fully Sharded Data Parallel + Tensor Parallel in
separate parallel dimensions:
    Data Parallel across hosts
    Tensor Parallel within each host

 We use a simple diagram to illustrate below:

======================================================================
------------       ------------       ------------       ------------
| Host 1   |       | Host 2   |       |          |       | Host N   |
| 8 GPUs   |       | 8 GPUs   |       |          |       | 8 GPUs   |
|          |       |          |       |    ...   |       |          |
| (TP)     |       | (TP)     |       |          |       | (TP)     |
|[0,1,..,7]|       |[8,9..,15]|       |          |       |[8N-8,8N-7|
|          |       |          |       |          |       | .., 8N-1]|
|          |       |          |       |          |       |          |
------------       ------------       ------------       ------------
FSDP:
[0, 8, ..., 8N-8], [1, 9, ..., 8N-7], ..., [7, 15, ..., 8N-1]
======================================================================

More details can be seen in the slide:
https://docs.google.com/presentation/d/17g6WqrO00rP3MsxbRENsPpjrlSkwiA_QB4r93_eB5is/
"""


def demo_2d(rank, args):
    """
    Main body of the demo of a basic version of tensor parallel by using
    PyTorch native APIs.
    """
    print(f"Running basic Megatron style TP example on rank {rank}.")
    setup(rank, args.world_size)
    assert (
        args.world_size % args.tp_size == 0
    ), "World size needs to be divisible by TP size"

    # create a sharding plan based on the given world_size.
    device_mesh = DeviceMesh(
        "cuda", torch.arange(0, args.world_size).view(-1, args.tp_size)
    )

    # create model and move it to GPU with id rank
    model = ToyModel().cuda(rank)
    # Create a optimizer for the parallelized module.
    LR = 0.25
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    # Parallelize the module based on the given Parallel Style.
    parallel_style = SequenceParallel() if args.run_seq_parallel else PairwiseParallel()
    model = parallelize_module(model, device_mesh, parallel_style, tp_mesh_dim=1)

    # We need to register hooks for TP + FSDP integration.
    assert (
        enable_2d_with_fsdp()
    ), "FSDP 2D hook is not registered. Please use PyTorch with version >= 2.0"
    dp_pg = device_mesh.get_dim_groups()[0]
    model = FSDP(model, process_group=dp_pg)

    # Perform a num of iterations of forward/backward
    # and optimizations for the sharded module.
    for i in range(args.iter_nums):
        # For TP, input needs to be same across all TP ranks.
        # while for SP, input can be different across all ranks.
        # Setting the random seed is to mimic the behavior of dataloader.
        dp_rank = (
            rank
            if args.run_seq_parallel
            else dist.get_rank(dp_pg)
        )
        torch.manual_seed(i + dp_rank)
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
    parser.add_argument("--run_seq_parallel", type=bool, default=False)
    parser.add_argument("--tp_size", type=int, default=2)
    args = parser.parse_args()
    # The main entry point is called directly without using subprocess
    if n_gpus < 4:
        print("Requires at least 4 GPUs to run.")
    elif not SP_AVAILABLE:
        print(
            "PyTorch doesn't have Sequence Parallelism available,"
            " need nightly build."
        )
    else:
        mp.spawn(demo_2d, args=(args,), nprocs=args.world_size, join=True)


