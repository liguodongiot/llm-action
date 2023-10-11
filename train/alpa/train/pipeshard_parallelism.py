"""
Distributed Training with Both Shard and Pipeline Parallelism
=============================================================

Alpa can automatically parallelizes jax functions with both shard
parallelism (a.k.a. intra-operator parallelism) and pipeline parallelism
(a.k.a. inter-operator parallelism). Shard parallelism includes
data parallelism, operator parallelism, and their combinations.
The previous :ref:`quick start <alpa-quickstart>` tutorial focuses on
using Alpa for shard parallelism.

In this tutorial, we show how to use Alpa with both shard and pipeline parallelism.
First, we show how to use Alpa to manually assign stages for pipeline parallelism.
Then we show how to use Alpa to automate this process.
"""

################################################################################
# Import Libraries and Initialize Environment
# -------------------------------------------
# First, import the required libraries.

import alpa
from alpa.testing import assert_allclose
import copy
from flax import linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
from jax import random
import optax
import ray

alpa.util.disable_tqdm_globally()

################################################################################
# Connect to a Ray Cluster
# ------------------------
# Alpa uses a distributed framework `ray <https://docs.ray.io/>`_ to manage
# the cluster and disributed workers. We initialize ray and alpa.

ray.init()
alpa.init(cluster="ray")

# Alternatively, you can use the following command to connect to an existing
# ray cluster.
# ray.init(address="auto")
#
# Note: `alpa.init(cluster="ray")` uses the gpus resources of the whole ray
# cluster. To configure Alpa to only use a subset of gpu resources, one can 
# specific the number of nodes and number of gpus per node.
# For example, only run 2 gpus when 8 gpus are available 
# alpa.init('ray', devices_per_node=2, num_nodes=1)  

################################################################################
# Train an MLP on a Single Device
# -------------------------------
# In this tutorial, we use a toy dataset to train an MLP model.
# Specifically, we use the model to fit the function: :math:`y = Wx + b`.
# Note that now this model is being executed on CPU because we force the driver
# process to use the CPU.


class MLPModel(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.hidden_dim * 4)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dim * 4)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        return x


dim = 2048
batch_size = 2048

# Generate ground truth W and b
rngkey = jax.random.PRNGKey(0)
k1, k2 = random.split(rngkey)
W = random.normal(k1, (dim, dim))
b = random.normal(k2, (dim,))

# Generate the training data
ksample, knoise = random.split(k1)
x = random.normal(ksample, (batch_size, dim))
y = (x @ W + b) + 0.1 * random.normal(knoise, (batch_size, dim))

# Initialize a train state, which includes the model paramter and optimizer
# state.
model = MLPModel(hidden_dim=dim)
params = model.init(rngkey, x)
tx = optax.adam(learning_rate=1e-3)
state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)


# Define the training step
def train_step(state, batch):

    def loss_func(params):
        out = model.apply(params, batch["x"])
        loss = jnp.mean((out - batch["y"])**2)
        return loss

    grads = jax.grad(loss_func)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state


batch = {"x": x, "y": y}
expected_state = train_step(state, batch)

################################################################################
# Pipeline Parallelism with Manual Assignment
# -------------------------------------------
# Pipeline paralleism requires partitioning the model into several pipeline
# stages. To manually assign stages, we can use ``alpa.mark_pipeline_boundary``
# to mark the boundary of each pipeline stage in the forward function.
# Note that each pipeline stage is also automatically parallelized by the
# shard parallel pass.


# Define a MLP model with manual stage boundaries.
class ManualPipelineMLPModel(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.hidden_dim * 4)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        # Use this boundary marker to separate the model into two stages.
        alpa.mark_pipeline_boundary()
        x = nn.Dense(features=self.hidden_dim * 4)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        return x


# Initialize the train state with the same parameters as the single-device
# model.
manual_pipeline_model = ManualPipelineMLPModel(hidden_dim=dim)
manual_pipeline_state = TrainState.create(apply_fn=manual_pipeline_model.apply,
                                          params=copy.deepcopy(params),
                                          tx=tx)


# Define the training step.
# We use the "alpa.PipeshardParallel" option to let alpa use both
# pipeline parallelism and shard parallelism. To make pipeline parallelism
# efficient, we need to fill the pipeline with many micro batches,
# so a `num_micro_batches` should be specified.
@alpa.parallelize(method=alpa.PipeshardParallel(num_micro_batches=16,
                                                layer_option="manual"))
def manual_pipeline_train_step(state, batch):

    def loss_func(params):
        out = state.apply_fn(params, batch["x"])
        loss = jnp.mean((out - batch["y"])**2)
        return loss

    # We use `alpa.grad` here to separate the apply gradient stage with the
    # forward/backward stages in the pipeline. This is necessary to ensure that
    # the gradient accumulation is correct.
    grads = alpa.grad(loss_func)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state


manual_pipeline_actual_state = manual_pipeline_train_step(
    manual_pipeline_state, batch)
assert_allclose(expected_state.params,
                manual_pipeline_actual_state.params,
                atol=5e-3)

alpa.shutdown()

####################
#
# .. note::
#
#   In addition, Alpa supports more flexible manual assignments of pipeline
#   parallelism strategies. In the above example, each partitioned stages will
#   be assigned an equal number of devices to run. If you want to control the
#   device assignment of each stage, you can use the more advanced
#   ``stage_option=alpa.ManualStageOption``.

################################################################################
# Pipeline Parallelism with Automatic Assignment
# ----------------------------------------------
# Alpa also supports automatically partitioning the model into multiple
# pipeline stages and assign each pipeline stage a device mesh such that
# the total execution latency is minimized. Specifically, the automatic
# partitioning algorithm consists of the following steps:
#
# 1. **Layer Construction:** In this step, the operators in the model are
#    clustered into "layers" based on a graph clustering algorithm. The
#    user needs to specify the total number of layers (i.e. clusters) as
#    a hyperparameter.
# 2. **Stage Construction and Mesh Slicing:** In this step, we partition
#    the device cluster (device mesh) to multiple submeshes and assign
#    layers to submeshes to form pipeline stages to minimize the total
#    pipeline execution latency.

alpa.init(cluster="ray")

# Define the parallel method.
# `alpa.AutoLayerOption(layer_num=2)` means we use the auto layer construcion
# algorithm to cluster primitive operators into two layers.
# `stage_option="auto"` means we enable the auto stage construction algorithm.
method = alpa.PipeshardParallel(num_micro_batches=16,
                                layer_option=alpa.AutoLayerOption(layer_num=2),
                                stage_option="auto")


# Define the training step. The function body is the same as the above one.
@alpa.parallelize(method=method)
def auto_pipeline_train_step(state, batch):

    def loss_func(params):
        out = state.apply_fn(params, batch["x"])
        loss = jnp.mean((out - batch["y"])**2)
        return loss

    # Again, we use `alpa.grad` here to separate the apply gradient stage with
    # the forward/backward stages in the pipeline.
    grads = alpa.grad(loss_func)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state


# In the first call, alpa triggers the compilation.
# The compilation first profiles several costs and solves an optimization
# problem to get the optimal pipeline assignments.
auto_pipeline_actual_state = auto_pipeline_train_step(state, batch)
assert_allclose(expected_state.params,
                auto_pipeline_actual_state.params,
                atol=5e-3)

alpa.shutdown()

################################################################################
# Interpret the Results
# ---------------------
# **Some basic concepts**
# - Cluster mesh and submeshes
#     - Cluster mesh is a computer cluster that contains GPUs. A ``N×M`` cluster mesh means the cluster has ``N`` physical machines and each machine has ``M`` GPUs.
#     - Submeshes can be obtained by slicing from the cluster mesh. For example, given a ``N×M`` cluster mesh, a submesh ``(1, M)`` means using all GPUs in one physical machine.
#     - For more details on how Alpa uses submeshes to solve *inter-operator parallelism*, you can read the **Section 5: Inter-Operator Parallelism** in the `Alpa paper <https://arxiv.org/pdf/2201.12023.pdf>`_.
# - Device mesh and logical mesh
#     - A device mesh is a 2-dimensional logical view of a set of physical devices.
#     - For a set of physical devices, there can be multiple logical views. For example, given 2 nodes and 8 GPUs per node (i.e., 16 devices in total), we can view them as a 2×8, 1×16, 4×4, 8×2, or 16×1 device mesh.
#     - The mapping between physical devices and the logical device mesh view is optimized by the inter-op pass
#         - Hence, you can see ``Result mesh_shapes`` and the corresponding ``Result logical_mesh_shapes`` in the optimization output.
#
# With the basic concepts in mind, you now can better understand the ``ModuleProfileResult``:
# - ``ModuleProfileResult``: ``result[(i, j, s, c), m]`` means this stage contains forward layers ``i, i+1, ..., j`` and corresponding backward layers, and runs under the ``s``-th submesh and the ``c``-th auto sharding config for the submesh. The ``m = 0`` means the result is for the forward pass, and ``m = 1`` for backward pass.