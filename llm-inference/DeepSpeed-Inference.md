


Automatic Tensor Parallelism for HuggingFace Models
https://www.deepspeed.ai/tutorials/automatic-tensor-parallelism/


```
# ---------------------------------------
# New automatic tensor parallelism method
# ---------------------------------------
import os
import torch
import transformers
import deepspeed
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))
# create the model pipeline
pipe = transformers.pipeline(task="text2text-generation", model="google/t5-v1_1-small", device=local_rank)
# Initialize the DeepSpeed-Inference engine
pipe.model = deepspeed.init_inference(
    pipe.model,
    mp_size=world_size,
    dtype=torch.float
)
output = pipe('Input String')
```
