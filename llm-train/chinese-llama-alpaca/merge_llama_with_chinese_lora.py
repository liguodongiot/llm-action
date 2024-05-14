"""
Usage:
python merge_llama_with_chinese_lora.py \
    --base_model path/to/llama/model \
    --lora_model path/to/first/lora/model [path/to/second/lora/model] \
    --output_type [pth|huggingface] \
    --output_dir path/to/output/dir
"""
import argparse
import json
import os
import gc
import torch
import peft
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer
from huggingface_hub import hf_hub_download

parser = argparse.ArgumentParser()
parser.add_argument('--base_model', default=None, required=True,
                    type=str, help="Please specify a base_model")
parser.add_argument('--tokenizer_path', default=None, required=True,
                    type=str, help="Please specify a tokenizer_path")
parser.add_argument('--lora_model', default=None, required=True,
                    type=str,
                    help="Please specify LoRA models to be merged (ordered); use commas to separate multiple LoRA models.")
parser.add_argument('--offload_dir', default=None, type=str,
                    help="(Optional) Please specify a temp folder for offloading (useful for low-RAM machines). Default None (disable offload).")
parser.add_argument('--output_type', default='pth', choices=['pth', 'huggingface'], type=str,
                    help="save the merged model in pth or huggingface format.")
parser.add_argument('--output_dir', default='./', type=str)

emb_to_model_size = {
    4096: '7B',
    5120: '13B',
    6656: '30B',
    8192: '65B',
}
num_shards_of_models = {'7B': 1, '13B': 2}
params_of_models = {
    '7B':
        {
            "dim": 4096,
            "multiple_of": 256,
            "n_heads": 32,
            "n_layers": 32,
            "norm_eps": 1e-06,
            "vocab_size": -1,
        },
    '13B':
        {
            "dim": 5120,
            "multiple_of": 256,
            "n_heads": 40,
            "n_layers": 40,
            "norm_eps": 1e-06,
            "vocab_size": -1,
        },
}


def transpose(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight


# Borrowed and modified from https://github.com/tloen/alpaca-lora
def translate_state_dict_key(k):
    k = k.replace("base_model.model.", "")
    if k == "model.embed_tokens.weight":
        return "tok_embeddings.weight"
    elif k == "model.norm.weight":
        return "norm.weight"
    elif k == "lm_head.weight":
        return "output.weight"
    elif k.startswith("model.layers."):
        layer = k.split(".")[2]
        if k.endswith(".self_attn.q_proj.weight"):
            return f"layers.{layer}.attention.wq.weight"
        elif k.endswith(".self_attn.k_proj.weight"):
            return f"layers.{layer}.attention.wk.weight"
        elif k.endswith(".self_attn.v_proj.weight"):
            return f"layers.{layer}.attention.wv.weight"
        elif k.endswith(".self_attn.o_proj.weight"):
            return f"layers.{layer}.attention.wo.weight"
        elif k.endswith(".mlp.gate_proj.weight"):
            return f"layers.{layer}.feed_forward.w1.weight"
        elif k.endswith(".mlp.down_proj.weight"):
            return f"layers.{layer}.feed_forward.w2.weight"
        elif k.endswith(".mlp.up_proj.weight"):
            return f"layers.{layer}.feed_forward.w3.weight"
        elif k.endswith(".input_layernorm.weight"):
            return f"layers.{layer}.attention_norm.weight"
        elif k.endswith(".post_attention_layernorm.weight"):
            return f"layers.{layer}.ffn_norm.weight"
        elif k.endswith("rotary_emb.inv_freq") or "lora" in k:
            return None
        else:
            print(layer, k)
            raise NotImplementedError
    else:
        print(k)
        raise NotImplementedError


def unpermute(w):
    return (
        w.view(n_heads, 2, dim // n_heads // 2, dim).transpose(1, 2).reshape(dim, dim)
    )


def save_shards(model_sd, num_shards: int):
    # Add the no_grad context manager
    with torch.no_grad():
        if num_shards == 1:
            new_state_dict = {}
            for k, v in model_sd.items():
                new_k = translate_state_dict_key(k)
                if new_k is not None:
                    if "wq" in new_k or "wk" in new_k:
                        new_state_dict[new_k] = unpermute(v)
                    else:
                        new_state_dict[new_k] = v

            os.makedirs(output_dir, exist_ok=True)
            print(f"Saving shard 1 of {num_shards} into {output_dir}/consolidated.00.pth")
            torch.save(new_state_dict, output_dir + "/consolidated.00.pth")
            with open(output_dir + "/params.json", "w") as f:
                json.dump(params, f)
        else:
            new_state_dicts = [dict() for _ in range(num_shards)]
            for k in list(model_sd.keys()):
                v = model_sd[k]
                new_k = translate_state_dict_key(k)
                if new_k is not None:
                    if new_k == 'tok_embeddings.weight':
                        print(f"Processing {new_k}")
                        assert v.size(1) % num_shards == 0
                        splits = v.split(v.size(1) // num_shards, dim=1)
                    elif new_k == 'output.weight':
                        print(f"Processing {new_k}")
                        if v.size(0) % num_shards == 0:
                            splits = v.split(v.size(0) // num_shards, dim=0)
                        else:
                            size_list = [v.size(0) // num_shards] * num_shards
                            size_list[-1] += v.size(0) % num_shards
                            splits = v.split(size_list, dim=0)  # 13B: size_list == [24976,24977]
                    elif new_k == 'norm.weight':
                        print(f"Processing {new_k}")
                        splits = [v] * num_shards
                    elif 'ffn_norm.weight' in new_k:
                        print(f"Processing {new_k}")
                        splits = [v] * num_shards
                    elif 'attention_norm.weight' in new_k:
                        print(f"Processing {new_k}")
                        splits = [v] * num_shards


                    elif 'w1.weight' in new_k:
                        print(f"Processing {new_k}")
                        splits = v.split(v.size(0) // num_shards, dim=0)
                    elif 'w2.weight' in new_k:
                        print(f"Processing {new_k}")
                        splits = v.split(v.size(1) // num_shards, dim=1)
                    elif 'w3.weight' in new_k:
                        print(f"Processing {new_k}")
                        splits = v.split(v.size(0) // num_shards, dim=0)


                    elif 'wo.weight' in new_k:
                        print(f"Processing {new_k}")
                        splits = v.split(v.size(1) // num_shards, dim=1)

                    elif 'wv.weight' in new_k:
                        print(f"Processing {new_k}")
                        splits = v.split(v.size(0) // num_shards, dim=0)

                    elif "wq.weight" in new_k or "wk.weight" in new_k:
                        print(f"Processing {new_k}")
                        v = unpermute(v)
                        splits = v.split(v.size(0) // num_shards, dim=0)
                    else:
                        print(f"Unexpected key {new_k}")
                        raise ValueError
                    for sd, split in zip(new_state_dicts, splits):
                        sd[new_k] = split.clone()
                        del split
                    del splits
                del model_sd[k], v
                gc.collect()  # Effectively enforce garbage collection

            os.makedirs(output_dir, exist_ok=True)
            for i, new_state_dict in enumerate(new_state_dicts):
                print(f"Saving shard {i + 1} of {num_shards} into {output_dir}/consolidated.0{i}.pth")
                torch.save(new_state_dict, output_dir + f"/consolidated.0{i}.pth")
            with open(output_dir + "/params.json", "w") as f:
                print(f"Saving params.json into {output_dir}/params.json")
                json.dump(params, f)


if __name__ == '__main__':

    args = parser.parse_args()
    base_model_path = args.base_model
    lora_model_paths = [s.strip() for s in args.lora_model.split(',') if len(s.strip()) != 0]

    tokenizer_paths = [s.strip() for s in args.tokenizer_path.split(',') if len(s.strip()) != 0]
    output_dir = args.output_dir
    output_type = args.output_type
    offload_dir = args.offload_dir

    print(f"Base model: {base_model_path}")
    print(f"LoRA model(s) {lora_model_paths}:")

    if offload_dir is not None:
        # Load with offloading, which is useful for low-RAM machines.
        # Note that if you have enough RAM, please use original method instead, as it is faster.
        base_model = LlamaForCausalLM.from_pretrained(
            base_model_path,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            offload_folder=offload_dir,
            offload_state_dict=True,
            low_cpu_mem_usage=True,
            device_map={"": "cpu"},
        )
    else:
        # Original method without offloading
        base_model = LlamaForCausalLM.from_pretrained(
            base_model_path,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map={"": "cpu"},
        )

    ## infer the model size from the checkpoint
    embedding_size = base_model.get_input_embeddings().weight.size(1)
    model_size = emb_to_model_size[embedding_size]
    print(f"Peft version: {peft.__version__}")
    print(f"Loading LoRA for {model_size} model")

    lora_model = None
    lora_model_sd = None
    for lora_index, lora_model_path in enumerate(lora_model_paths):
        tokenizer_path = tokenizer_paths[lora_index]
        print(f"Loading LoRA: {lora_model_path} , tokenizer path: {tokenizer_path} ")
        # tokenizer = LlamaTokenizer.from_pretrained(lora_model_path)
        tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)

        if base_model.get_input_embeddings().weight.size(0) != len(tokenizer):
            base_model.resize_token_embeddings(len(tokenizer))
            print(f"Extended vocabulary size to {len(tokenizer)}")

        first_weight = base_model.model.layers[0].self_attn.q_proj.weight
        first_weight_old = first_weight.clone()

        if hasattr(peft.LoraModel, 'merge_and_unload'):
            lora_model = PeftModel.from_pretrained(
                base_model,
                lora_model_path,
                device_map={"": "cpu"},
                torch_dtype=torch.float16,
            )
            assert torch.allclose(first_weight_old, first_weight)
            print(f"Merging with merge_and_unload...")
            base_model = lora_model.merge_and_unload()
        else:
            base_model_sd = base_model.state_dict()
            try:
                lora_model_sd = torch.load(os.path.join(lora_model_path, 'adapter_model.bin'), map_location='cpu')
            except FileNotFoundError:
                print("Cannot find lora model on the disk. Downloading lora model from hub...")
                filename = hf_hub_download(repo_id=lora_model_path, filename='adapter_model.bin')
                lora_model_sd = torch.load(filename, map_location='cpu')

            lora_config = peft.LoraConfig.from_pretrained(lora_model_path)
            lora_scaling = lora_config.lora_alpha / lora_config.r
            fan_in_fan_out = lora_config.fan_in_fan_out
            lora_keys = [k for k in lora_model_sd if 'lora_A' in k]
            non_lora_keys = [k for k in lora_model_sd if not 'lora_' in k]

            for k in non_lora_keys:
                print(f"merging {k}")
                original_k = k.replace('base_model.model.', '')
                base_model_sd[original_k].copy_(lora_model_sd[k])

            for k in lora_keys:
                print(f"merging {k}")
                original_key = k.replace('.lora_A', '').replace('base_model.model.', '')
                assert original_key in base_model_sd
                lora_a_key = k
                lora_b_key = k.replace('lora_A', 'lora_B')
                base_model_sd[original_key] += (
                        transpose(lora_model_sd[lora_b_key].float() @ lora_model_sd[lora_a_key].float(),
                                  fan_in_fan_out) * lora_scaling
                )
                assert base_model_sd[original_key].dtype == torch.float16

        # did we do anything?
        assert not torch.allclose(first_weight_old, first_weight)

    tokenizer.save_pretrained(output_dir)

    if output_type == 'huggingface':
        print("Saving to Hugging Face format...")
        LlamaForCausalLM.save_pretrained(base_model, output_dir)  # , state_dict=deloreanized_sd)
    else:  # output_type=='pth
        print("Saving to pth format...")

        base_model_sd = base_model.state_dict()
        del lora_model, base_model, lora_model_sd

        params = params_of_models[model_size]
        num_shards = num_shards_of_models[model_size]
        n_layers = params["n_layers"]
        n_heads = params["n_heads"]
        dim = params["dim"]
        dims_per_head = dim // n_heads
        base = 10000.0
        inv_freq = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))

        save_shards(model_sd=base_model_sd, num_shards=num_shards)
