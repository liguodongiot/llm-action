"""
Training Transformer models using Distributed Data Parallel and Pipeline Parallelism
====================================================================================

**Author**: `Pritam Damania <https://github.com/pritamdamania87>`_

This tutorial demonstrates how to train a large Transformer model across
multiple GPUs using `Distributed Data Parallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`__ and
`Pipeline Parallelism <https://pytorch.org/docs/stable/pipeline.html>`__. This tutorial is an extension of the
`Sequence-to-Sequence Modeling with nn.Transformer and TorchText <https://pytorch.org/tutorials/beginner/transformer_tutorial.html>`__ tutorial
and scales up the same model to demonstrate how Distributed Data Parallel and
Pipeline Parallelism can be used to train Transformer models.

Prerequisites:

    * `Pipeline Parallelism <https://pytorch.org/docs/stable/pipeline.html>`__
    * `Sequence-to-Sequence Modeling with nn.Transformer and TorchText <https://pytorch.org/tutorials/beginner/transformer_tutorial.html>`__
    * `Getting Started with Distributed Data Parallel <https://pytorch.org/tutorials/intermediate/ddp_tutorial.html>`__
"""


######################################################################
# Define the model
# ----------------
#

######################################################################
# ``PositionalEncoding`` module injects some information about the
# relative or absolute position of the tokens in the sequence. The
# positional encodings have the same dimension as the embeddings so that
# the two can be summed. Here, we use ``sine`` and ``cosine`` functions of
# different frequencies.

import sys
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import tempfile
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.pe = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


######################################################################
# In this tutorial, we will split a Transformer model across two GPUs and use
# pipeline parallelism to train the model. In addition to this, we use
# `Distributed Data Parallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`__
# to train two replicas of this pipeline. We have one process driving a pipe across
# GPUs 0 and 1 and another process driving a pipe across GPUs 2 and 3. Both these
# processes then use Distributed Data Parallel to train the two replicas. The
# model is exactly the same model used in the `Sequence-to-Sequence Modeling with nn.Transformer and TorchText
# <https://pytorch.org/tutorials/beginner/transformer_tutorial.html>`__ tutorial,
# but is split into two stages. The largest number of parameters belong to the
# `nn.TransformerEncoder <https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html>`__ layer.
# The `nn.TransformerEncoder <https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html>`__
# itself consists of ``nlayers`` of `nn.TransformerEncoderLayer <https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html>`__.
# As a result, our focus is on ``nn.TransformerEncoder`` and we split the model
# such that half of the ``nn.TransformerEncoderLayer`` are on one GPU and the
# other half are on another. To do this, we pull out the ``Encoder`` and
# ``Decoder`` sections into separate modules and then build an ``nn.Sequential``
# representing the original Transformer module.


if sys.platform == 'win32':
    print('Windows platform is not supported for pipeline parallelism')
    sys.exit(0)
if torch.cuda.device_count() < 4:
    print('Need at least four GPU devices for this tutorial')
    sys.exit(0)

class Encoder(nn.Module):
    def __init__(self, ntoken, ninp, dropout=0.5):
        super(Encoder, self).__init__()
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        # Need (S, N) format for encoder.
        src = src.t()
        src = self.encoder(src) * math.sqrt(self.ninp)
        return self.pos_encoder(src)

class Decoder(nn.Module):
    def __init__(self, ntoken, ninp):
        super(Decoder, self).__init__()
        self.decoder = nn.Linear(ninp, ntoken)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, inp):
        # Need batch dimension first for output of pipeline.
        return self.decoder(inp).permute(1, 0, 2)

######################################################################
# Start multiple processes for training
# -------------------------------------
#


######################################################################
# We start two processes where each process drives its own pipeline across two
# GPUs. ``run_worker`` is executed for each process.

def run_worker(rank, world_size):


######################################################################
# Load and batch data
# -------------------
#


######################################################################
# The training process uses Wikitext-2 dataset from ``torchtext``. 
# To access torchtext datasets, please install torchdata following instructions at https://github.com/pytorch/data.
#
# The vocab object is built based on the train dataset and is used to numericalize
# tokens into tensors. Starting from sequential data, the ``batchify()``
# function arranges the dataset into columns, trimming off any tokens remaining
# after the data has been divided into batches of size ``batch_size``.
# For instance, with the alphabet as the sequence (total length of 26)
# and a batch size of 4, we would divide the alphabet into 4 sequences of
# length 6:
#
# .. math::
#
#     \begin{bmatrix}
#    \text{A} & \text{B} & \text{C} & \ldots & \text{X} & \text{Y} & \text{Z}
#    \end{bmatrix}
#    \Rightarrow
#    \begin{bmatrix}
#    \begin{bmatrix}\text{A} \\ \text{B} \\ \text{C} \\ \text{D} \\ \text{E} \\ \text{F}\end{bmatrix} &
#    \begin{bmatrix}\text{G} \\ \text{H} \\ \text{I} \\ \text{J} \\ \text{K} \\ \text{L}\end{bmatrix} &
#    \begin{bmatrix}\text{M} \\ \text{N} \\ \text{O} \\ \text{P} \\ \text{Q} \\ \text{R}\end{bmatrix} &
#    \begin{bmatrix}\text{S} \\ \text{T} \\ \text{U} \\ \text{V} \\ \text{W} \\ \text{X}\end{bmatrix}
#    \end{bmatrix}
#
# These columns are treated as independent by the model, which means that
# the dependence of ``G`` and ``F`` can not be learned, but allows more
# efficient batch processing.
#

# In 'run_worker'
    def print_with_rank(msg):
        print('[RANK {}]: {}'.format(rank, msg))

    from torchtext.datasets import WikiText2
    from torchtext.data.utils import get_tokenizer
    from torchtext.vocab import build_vocab_from_iterator

    train_iter = WikiText2(split='train')
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"]) 

    def data_process(raw_text_iter):
      data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
      return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    train_iter, val_iter, test_iter = WikiText2()
    train_data = data_process(train_iter)
    val_data = data_process(val_iter)
    test_data = data_process(test_iter)

    device = torch.device(2 * rank)

    def batchify(data, bsz, rank, world_size, is_train=False):
        # Divide the dataset into ``bsz`` parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the ``bsz`` batches.
        data = data.view(bsz, -1).t().contiguous()
        # Divide the data across the ranks only for training data.
        if is_train:
            data_per_rank = data.size(0) // world_size
            data = data[rank * data_per_rank : (rank + 1) * data_per_rank]
        return data.to(device)

    batch_size = 20
    eval_batch_size = 10
    train_data = batchify(train_data, batch_size, rank, world_size, True)
    val_data = batchify(val_data, eval_batch_size, rank, world_size)
    test_data = batchify(test_data, eval_batch_size, rank, world_size)


######################################################################
# Functions to generate input and target sequence
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


######################################################################
# ``get_batch()`` function generates the input and target sequence for
# the transformer model. It subdivides the source data into chunks of
# length ``bptt``. For the language modeling task, the model needs the
# following words as ``Target``. For example, with a ``bptt`` value of 2,
# we’d get the following two Variables for ``i`` = 0:
#
# .. image:: ../_static/img/transformer_input_target.png
#
# It should be noted that the chunks are along dimension 0, consistent
# with the ``S`` dimension in the Transformer model. The batch dimension
# ``N`` is along dimension 1.
#

# In 'run_worker'
    bptt = 35
    def get_batch(source, i):
        seq_len = min(bptt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].view(-1)
        # Need batch dimension first for pipeline parallelism.
        return data.t(), target

######################################################################
# Model scale and Pipe initialization
# -----------------------------------
#


######################################################################
# To demonstrate training large Transformer models using pipeline parallelism,
# we scale up the Transformer layers appropriately. We use an embedding
# dimension of 4096, hidden size of 4096, 16 attention heads and 8 total
# transformer layers (``nn.TransformerEncoderLayer``). This creates a model with
# **~1 billion** parameters.
#
# We need to initialize the `RPC Framework <https://pytorch.org/docs/stable/rpc.html>`__
# since Pipe depends on the RPC framework via `RRef <https://pytorch.org/docs/stable/rpc.html#rref>`__
# which allows for future expansion to cross host pipelining. We need to
# initialize the RPC framework with only a single worker since we're using a
# single process to drive multiple GPUs.
#
# The pipeline is then initialized with 8 transformer layers on one GPU and 8
# transformer layers on the other GPU. One pipe is setup across GPUs 0 and 1 and
# another across GPUs 2 and 3. Both pipes are then replicated using ``DistributedDataParallel``.

# In 'run_worker'
    ntokens = len(vocab) # the size of vocabulary
    emsize = 4096 # embedding dimension
    nhid = 4096 # the dimension of the feedforward network model in ``nn.TransformerEncoder``
    nlayers = 8 # the number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
    nhead = 16 # the number of heads in the Multihead Attention models
    dropout = 0.2 # the dropout value

    from torch.distributed import rpc
    tmpfile = tempfile.NamedTemporaryFile()
    
    rpc.init_rpc(
        name="worker",
        rank=0,
        world_size=1,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
            init_method="file://{}".format(tmpfile.name),
            # Specifying _transports and _channels is a workaround and we no longer
            # will have to specify _transports and _channels for PyTorch
            # versions >= 1.8.1
            _transports=["ibv", "uv"],
            _channels=["cuda_ipc", "cuda_basic"],
        )
    )

    # Number of GPUs for model parallelism.
    num_gpus = 2
    partition_len = ((nlayers - 1) // num_gpus) + 1
    print("partition_len:", partition_len, "rank:", rank)

    # Add encoder in the beginning.
    tmp_list = [Encoder(ntokens, emsize, dropout).cuda(2 * rank)]
    module_list = []

    # Add all the necessary transformer blocks.
    for i in range(nlayers):
        transformer_block = TransformerEncoderLayer(emsize, nhead, nhid, dropout)
        if i != 0 and i % (partition_len) == 0:
            module_list.append(nn.Sequential(*tmp_list))
            tmp_list = []
        device = i // (partition_len)
        tmp_list.append(transformer_block.to(2 * rank + device))

    # Add decoder in the end.
    tmp_list.append(Decoder(ntokens, emsize).cuda(2 * rank + num_gpus - 1))
    module_list.append(nn.Sequential(*tmp_list))

    # Need to use 'checkpoint=never' since as of PyTorch 1.8, Pipe checkpointing
    # doesn't work with DDP.
    from torch.distributed.pipeline.sync import Pipe
    chunks = 8
    model = Pipe(torch.nn.Sequential(*module_list), chunks = chunks, checkpoint="never")

    # Initialize process group and wrap model in DDP.
    from torch.nn.parallel import DistributedDataParallel
    import torch.distributed as dist

    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(
                backend="nccl", rank=rank, world_size=world_size)
    model = DistributedDataParallel(model)

    def get_total_params(module: torch.nn.Module):
        total_params = 0
        for param in module.parameters():
            total_params += param.numel()
        return total_params

    print_with_rank('Total parameters in model: {:,}'.format(get_total_params(model)))

######################################################################
# Run the model
# -------------
#


######################################################################
# `CrossEntropyLoss <https://pytorch.org/docs/master/nn.html?highlight=crossentropyloss#torch.nn.CrossEntropyLoss>`__
# is applied to track the loss and
# `SGD <https://pytorch.org/docs/master/optim.html?highlight=sgd#torch.optim.SGD>`__
# implements stochastic gradient descent method as the optimizer. The initial
# learning rate is set to 5.0. `StepLR <https://pytorch.org/docs/master/optim.html?highlight=steplr#torch.optim.lr_scheduler.StepLR>`__ is
# applied to adjust the learn rate through epochs. During the
# training, we use
# `nn.utils.clip_grad_norm\_ <https://pytorch.org/docs/master/nn.html?highlight=nn%20utils%20clip_grad_norm#torch.nn.utils.clip_grad_norm_>`__
# function to scale all the gradient together to prevent exploding.
#

# In 'run_worker'
    criterion = nn.CrossEntropyLoss()
    lr = 5.0 # learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    import time
    def train():
        model.train() # Turn on the train mode
        total_loss = 0.
        start_time = time.time()
        ntokens = len(vocab)

        # Train only for 50 batches to keep script execution time low.
        nbatches = min(50 * bptt, train_data.size(0) - 1)

        for batch, i in enumerate(range(0, nbatches, bptt)):
            data, targets = get_batch(train_data, i)
            optimizer.zero_grad()
            # Since the Pipe is only within a single host and process the ``RRef``
            # returned by forward method is local to this node and can simply
            # retrieved via ``RRef.local_value()``.
            output = model(data).local_value()
            # Need to move targets to the device where the output of the
            # pipeline resides.
            loss = criterion(output.view(-1, ntokens), targets.cuda(2 * rank + 1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            log_interval = 10
            if batch % log_interval == 0 and batch > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print_with_rank('| epoch {:3d} | {:5d}/{:5d} batches | '
                      'lr {:02.2f} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | ppl {:8.2f}'.format(
                        epoch, batch, nbatches // bptt, scheduler.get_last_lr()[0],
                        elapsed * 1000 / log_interval,
                        cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()

    def evaluate(eval_model, data_source):
        eval_model.eval() # Turn on the evaluation mode
        total_loss = 0.
        ntokens = len(vocab)
        # Evaluate only for 50 batches to keep script execution time low.
        nbatches = min(50 * bptt, data_source.size(0) - 1)
        with torch.no_grad():
            for i in range(0, nbatches, bptt):
                data, targets = get_batch(data_source, i)
                output = eval_model(data).local_value()
                output_flat = output.view(-1, ntokens)
                # Need to move targets to the device where the output of the
                # pipeline resides.
                total_loss += len(data) * criterion(output_flat, targets.cuda(2 * rank + 1)).item()
        return total_loss / (len(data_source) - 1)

######################################################################
# Loop over epochs. Save the model if the validation loss is the best
# we've seen so far. Adjust the learning rate after each epoch.

# In 'run_worker'
    best_val_loss = float("inf")
    epochs = 3 # The number of epochs
    best_model = None

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(model, val_data)
        print_with_rank('-' * 89)
        print_with_rank('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        print_with_rank('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

        scheduler.step()


######################################################################
# Evaluate the model with the test dataset
# -------------------------------------
#
# Apply the best model to check the result with the test dataset.

# In 'run_worker'
    test_loss = evaluate(best_model, test_data)
    print_with_rank('=' * 89)
    print_with_rank('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print_with_rank('=' * 89)

# Main execution
import torch.multiprocessing as mp

if __name__=="__main__":
    world_size = 2
    mp.spawn(run_worker, args=(world_size, ), nprocs=world_size, join=True)
######################################################################
# Output
# ------
#


######################################################################
#.. code-block:: py
#
#    [RANK 0]: | epoch   1 |    10/   50 batches | lr 5.00 | ms/batch 778.97 | loss 43.31 | ppl 6432469059895903232.00
#    [RANK 1]: | epoch   1 |    10/   50 batches | lr 5.00 | ms/batch 778.90 | loss 44.50 | ppl 21245447128217366528.00
#    [RANK 0]: | epoch   1 |    20/   50 batches | lr 5.00 | ms/batch 699.89 | loss 44.50 | ppl 21176949187407757312.00
#    [RANK 1]: | epoch   1 |    20/   50 batches | lr 5.00 | ms/batch 699.87 | loss 44.62 | ppl 23975861229620961280.00
#    [RANK 0]: | epoch   1 |    30/   50 batches | lr 5.00 | ms/batch 698.86 | loss 41.62 | ppl 1193312915629888256.00
#    [RANK 1]: | epoch   1 |    30/   50 batches | lr 5.00 | ms/batch 698.87 | loss 40.69 | ppl 471605759847546240.00
#    [RANK 0]: | epoch   1 |    40/   50 batches | lr 5.00 | ms/batch 698.34 | loss 45.20 | ppl 42812308420836458496.00
#    [RANK 1]: | epoch   1 |    40/   50 batches | lr 5.00 | ms/batch 698.33 | loss 45.68 | ppl 68839569686012223488.00
#    [RANK 1]: -----------------------------------------------------------------------------------------
#    [RANK 1]: | end of epoch   1 | time: 40.08s | valid loss  0.80 | valid ppl     2.22
#    [RANK 1]: -----------------------------------------------------------------------------------------
#    [RANK 0]: -----------------------------------------------------------------------------------------
#    [RANK 0]: | end of epoch   1 | time: 40.09s | valid loss  0.80 | valid ppl     2.22
#    [RANK 0]: -----------------------------------------------------------------------------------------
#    [RANK 0]: | epoch   2 |    10/   50 batches | lr 4.75 | ms/batch 768.51 | loss 36.34 | ppl 6063529544668166.00
#    [RANK 1]: | epoch   2 |    10/   50 batches | lr 4.75 | ms/batch 769.23 | loss 37.41 | ppl 17651211266236086.00
#    [RANK 0]: | epoch   2 |    20/   50 batches | lr 4.75 | ms/batch 699.57 | loss 28.97 | ppl 3798441739584.11
#    [RANK 1]: | epoch   2 |    20/   50 batches | lr 4.75 | ms/batch 699.56 | loss 29.28 | ppl 5203636967575.47
#    [RANK 0]: | epoch   2 |    30/   50 batches | lr 4.75 | ms/batch 699.04 | loss 28.43 | ppl 2212498693571.25
#    [RANK 1]: | epoch   2 |    30/   50 batches | lr 4.75 | ms/batch 699.05 | loss 28.33 | ppl 2015144761281.48
#    [RANK 0]: | epoch   2 |    40/   50 batches | lr 4.75 | ms/batch 699.10 | loss 23.30 | ppl 13121380184.92
#    [RANK 1]: | epoch   2 |    40/   50 batches | lr 4.75 | ms/batch 699.09 | loss 23.41 | ppl 14653799192.87
#    [RANK 0]: -----------------------------------------------------------------------------------------
#    [RANK 0]: | end of epoch   2 | time: 39.97s | valid loss  0.24 | valid ppl     1.27
#    [RANK 0]: -----------------------------------------------------------------------------------------
#    [RANK 1]: -----------------------------------------------------------------------------------------
#    [RANK 1]: | end of epoch   2 | time: 39.98s | valid loss  0.24 | valid ppl     1.27
#    [RANK 1]: -----------------------------------------------------------------------------------------
#    [RANK 0]: | epoch   3 |    10/   50 batches | lr 4.51 | ms/batch 769.36 | loss 12.80 | ppl 361681.11
#    [RANK 1]: | epoch   3 |    10/   50 batches | lr 4.51 | ms/batch 768.97 | loss 12.57 | ppl 287876.61
#    [RANK 0]: | epoch   3 |    20/   50 batches | lr 4.51 | ms/batch 698.27 | loss 12.01 | ppl 164364.60
#    [RANK 1]: | epoch   3 |    20/   50 batches | lr 4.51 | ms/batch 698.30 | loss 11.98 | ppl 159095.89
#    [RANK 0]: | epoch   3 |    30/   50 batches | lr 4.51 | ms/batch 697.75 | loss 10.90 | ppl 54261.91
#    [RANK 1]: | epoch   3 |    30/   50 batches | lr 4.51 | ms/batch 697.72 | loss 10.89 | ppl 53372.39
#    [RANK 0]: | epoch   3 |    40/   50 batches | lr 4.51 | ms/batch 699.49 | loss 10.78 | ppl 47948.35
#    [RANK 1]: | epoch   3 |    40/   50 batches | lr 4.51 | ms/batch 699.50 | loss 10.79 | ppl 48664.42
#    [RANK 0]: -----------------------------------------------------------------------------------------
#    [RANK 0]: | end of epoch   3 | time: 39.96s | valid loss  0.38 | valid ppl     1.46
#    [RANK 0]: -----------------------------------------------------------------------------------------
#    [RANK 1]: -----------------------------------------------------------------------------------------
#    [RANK 1]: | end of epoch   3 | time: 39.96s | valid loss  0.38 | valid ppl     1.46
#    [RANK 1]: -----------------------------------------------------------------------------------------
#    [RANK 0]: =========================================================================================
#    [RANK 0]: | End of training | test loss  0.33 | test ppl     1.39
#    [RANK 0]: =========================================================================================
#    [RANK 1]: =========================================================================================
#    [RANK 1]: | End of training | test loss  0.33 | test ppl     1.39
#    [RANK 1]: =========================================================================================
# 
