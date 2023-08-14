# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaTokenizer, LlamaForCausalLM

import torch
import json
import sys
import os

print("Loading tokenizer")

tokenizer = LlamaTokenizer.from_pretrained("/home/guodong.li/model/llama-7b-hf")
print("Tokenizer loaded!")
print("Loading model")

model = LlamaForCausalLM.from_pretrained("/home/guodong.li/model/llama-7b-hf", device_map='auto')

print("Model loaded!")

n_vocab = 500 # number of initial tokens for synthesizing data on each GPU.

i_start = sys.argv[1]
if os.path.exists("gen_data/gen.chunk."+str(i_start).zfill(2)+".jsonl"):
    with open("gen_data/gen.chunk."+str(i_start).zfill(2)+".jsonl", "r") as f:
        lines = f.readlines()
        inner_loop = len(lines) % n_vocab
        outer_loop = len(lines) // n_vocab
else:
    inner_loop = 0
    outer_loop = 0

for j in range(3 + outer_loop, 6):
    for i in range(int(i_start) * n_vocab + inner_loop, (int(i_start)+1) * n_vocab):
        print(i)
        input_ids = torch.tensor([[i]])
        input_ids = input_ids.cuda()
        print("generating")
        outputs1 = model.generate(input_ids, do_sample=False, max_length=j)
        outputs = model.generate(outputs1, do_sample=True, max_length=1024)
        gen_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        text_dict = {"text" : gen_text[0]}
        with open("gen_data/gen.chunk."+str(i_start).zfill(2)+".jsonl", "a") as f:
            f.write(json.dumps(text_dict))
            f.write('\n')
