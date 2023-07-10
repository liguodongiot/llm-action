import argparse
from transformers import AutoModel
import mindspore as ms
import torch as pt

parser = argparse.ArgumentParser()
parser.add_argument('--src_checkpoint_dir', type=str,
                    default='/home/huggingface-glm-6b', help='Checkpoint dir to load on.')
parser.add_argument('--dst_checkpoint_file', type=str,
                    default='./ms_glm_6b.ckpt', help='Checkpoint file path.')

args = parser.parse_args()

pt_ckpt_path = args.src_checkpoint_dir
model = AutoModel.from_pretrained(pt_ckpt_path, trust_remote_code=True).half()

pt_param = model.state_dict()

type_map = {"torch.float16": "ms.float16",
            "torch.float32": "ms.float32"}
ms_param = []
with open("check_pt_ckpt.txt", "w") as fp:
    for k, v in pt_param.items():
        if "word_embeddings.weight" in k:
            k = k.replace("weight", "embedding_table")
        if "post_attention_layernorm" in k or "input_layernorm" in k or "final_layernorm" in k :
            k = k.replace("weight", "gamma")
            k = k.replace("bias", "beta")
        ms_param.append({"name": k, "data": ms.Tensor(v.numpy())})

ms.save_checkpoint(ms_param, args.dst_checkpoint_file)

