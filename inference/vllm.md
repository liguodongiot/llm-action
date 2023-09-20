
- https://vllm.readthedocs.io/en/latest/getting_started/quickstart.html





## 安装

```
# (Optional) Create a new conda environment.
conda create -n myenv python=3.8 -y
conda activate myenv

# Install vLLM.
pip install vllm
```


```
# Pull the Docker image with CUDA 11.8.
docker run --gpus all -it --rm --shm-size=8g nvcr.io/nvidia/pytorch:22.12-py3
```




## 示例


### 离线批次推理

```
from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
llm = LLM(model="facebook/opt-125m")


# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```



### API Server (LLM服务化)


- 入参：https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py

```
python -m vllm.entrypoints.api_server --model facebook/opt-125m


python -m vllm.entrypoints.openai.api_server \
    --model facebook/opt-125m
```


---


```
curl http://localhost:8000/generate \
    -d '{
        "prompt": "San Francisco is a",
        "use_beam_search": true,
        "n": 4,
        "temperature": 0
    }'
    
```




