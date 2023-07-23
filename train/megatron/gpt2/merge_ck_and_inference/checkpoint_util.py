import argparse
import importlib
import torch.multiprocessing as mp
import os
import sys

# A loader is a python file with at least two functions
# - add_arguments - takes in a parser and adds any arguments needed
# - load_checkpoint - takes in the queue and parsed arguments

# A saver is similar but has save_checkpoint instead of
# load_checkpoint

# The loader and saver process are each given a queue, the loader
# should load the checkpoint and send the weights in messages in the
# following order, the saver should receive them in this order and
# save the checkpoints. A message consists of a python dictionary with
# a "name" for error checking and an entry for each tensor as
# indicated below. Note that the weight sent over the queue are the
# full model weights, nothing split.

# If the loader ever sends "exit" to the queue, that means something
# went wrong and it is exiting.

# - Metadata Namespace with the following attributes:
#     model_type - GPT, BERT, T5, etc.  (Part of protocol to allow this to be deduced later instead of given on command line)
#     num_layers - Number of transformer layers
#     hidden_size
#     seq_length
#     num_attention_heads
#     max_position_embeddings
#     tokenizer_type
#     iteration
#     params_dtype
#     bert_binary_head - Used only if model_type is BERT
#     previous_tensor_parallel_size - Optional
#     previous_pipeline_parallel_size - Optional
#     true_vocab_size
#     make_vocab_size_divisble_by
#     consumed_train_samples
#     consumed_valid_samples
# messages
# {
#   "name": "embeddings"
#   "position embeddings"
#   "word embeddings"
# }
# (for each transformer layer):
# {
#   "name": "transformer layer N"
#   "input layernorm weight"
#   "input layernorm bias"
#   "qkv weight"
#   "qkv bias"
#   "dense weight"
#   "dense bias"
#   "post layernorm weight"
#   "post layernorm bias"
#   "mlp l0 weight"
#   "mlp l0 bias"
#   "mlp l1 weight"
#   "mlp l1 bias"
# }
# {
#   "name": "final layer norm"
#   "weight"
#   "bias"
# }
# if present (i.e. for BERT):
# {
#   "name": "pooler"
#   "weight"
#   "bias"
# }
# {
#   "name": "lm head"
#   "dense weight"
#   "dense bias"
#   "layernorm weight"
#   "layernorm bias"
# }
# {
#   "name": "binary head"
#   "weight"
#   "bias"
# }
# - "done"

def load_plugin(plugin_type, name):
    module_name = f"checkpoint_{plugin_type}_{name}"
    try:
        plugin = importlib.import_module(module_name)
    except ModuleNotFoundError:
        module_name = name
        try:
            plugin = importlib.import_module(module_name)
        except ModuleNotFoundError:
            sys.exit(f"Unable to load {plugin_type} plugin {name}. Exiting.")

    if not hasattr(plugin, 'add_arguments'):
        sys.exit(f"{module_name} module is not a plugin. Exiting.")

    print(f"Loaded {module_name} as the {plugin_type}.")
    return plugin

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Megatron Checkpoint Utility Arguments",
                                     allow_abbrev=False, conflict_handler='resolve')

    parser.add_argument('--model-type', type=str, required=True,
                        choices=['GPT', 'BERT'],
                        help='Type of the model')
    parser.add_argument('--loader', type=str, default='megatron',
                        help='Module name to load checkpoint, should be on python path')
    parser.add_argument('--saver', type=str, default='megatron',
                        help='Module name to save checkpoint, shdoul be on python path')
    parser.add_argument('--load-dir', type=str, required=True,
                        help='Directory to load model checkpoint from')
    parser.add_argument('--save-dir', type=str, required=True,
                        help='Directory to save model checkpoint to')
    parser.add_argument('--max-queue-size', type=int, default=50,
                        help='Maximum number of tensors in the queue')
    parser.add_argument('--no-checking', action='store_false',
                        help='Do not perform checking on the name and ordering of weights',
                        dest='checking')

    known_args, _ = parser.parse_known_args()
    loader = load_plugin('loader', known_args.loader)
    saver = load_plugin('saver', known_args.saver)

    loader.add_arguments(parser)
    saver.add_arguments(parser)

    args = parser.parse_args()

    queue = mp.Queue(maxsize=args.max_queue_size)

    print("Starting saver...")
    saver_proc = mp.Process(target=saver.save_checkpoint, args=(queue, args))
    saver_proc.start()

    print("Starting loader...")
    loader.load_checkpoint(queue, args)

    print("Waiting for saver to complete...")
    saver_proc.join()


if __name__ == '__main__':
    main()
