# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import random

all_text = []

for i_start in range(8):
    if i_start % 10 == 0:
        print(i_start)
    for line in open("gen_data/gen.chunk."+str(i_start).zfill(2)+".jsonl", 'r'):
        all_text.append(json.loads(line))


with open("gen_data/all_gen.jsonl", "a") as f:
    for i in range(len(all_text)):
        f.write(json.dumps(all_text[i]))
        f.write('\n')
