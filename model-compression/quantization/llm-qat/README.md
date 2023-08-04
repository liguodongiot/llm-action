


```
cd /home/guodong.li/virtual-venv
virtualenv -p /usr/bin/python3.10 llm-qat-venv-py310-cu117
source /home/guodong.li/virtual-venv/llm-qat-venv-py310-cu117/bin/activate


pip install torch-1.13.1+cu117-cp310-cp310-linux_x86_64.whl
pip install torchvision-0.14.1+cu117-cp310-cp310-linux_x86_64.whl

git clone https://github.com/NVIDIA/apex
cd apex
git checkout 22.04-dev
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

pip install -r requirement.txt



/home/guodong.li/model/llama-7b-hf

合成数据：


config.json
tokenizer_config.json



mkdir -p gen_data/

CUDA_VISIBLE_DEVICES=0 python generate_data.py 0
CUDA_VISIBLE_DEVICES=1 python generate_data.py 1
CUDA_VISIBLE_DEVICES=2 python generate_data.py 2
CUDA_VISIBLE_DEVICES=3 python generate_data.py 3
CUDA_VISIBLE_DEVICES=4 python generate_data.py 4
CUDA_VISIBLE_DEVICES=5 python generate_data.py 5
CUDA_VISIBLE_DEVICES=6 python generate_data.py 6
CUDA_VISIBLE_DEVICES=7 python generate_data.py 7
```


```
tree -h gen_data/
gen_data/
├── [4.1M]  gen.chunk.00.jsonl
├── [4.9M]  gen.chunk.01.jsonl
├── [4.7M]  gen.chunk.02.jsonl
├── [4.8M]  gen.chunk.03.jsonl
├── [4.5M]  gen.chunk.04.jsonl
├── [4.4M]  gen.chunk.05.jsonl
├── [4.5M]  gen.chunk.06.jsonl
└── [4.6M]  gen.chunk.07.jsonl

0 directories, 8 files
(llm-qat-venv-py310-cu117) [guodong.li@ai-app-2-46-msxf LLM-QAT]$ wc -l gen_data/gen.chunk.07.jsonl
1500 gen_data/gen.chunk.07.jsonl




head -n1 gen_data/gen.chunk.02.jsonl
{"text": "ied with the idea of it and they've only one thing in mind: to find it. ...I needed the first, but I needed the second more, because I cried from beginning to"}


------------------------------


python merge_gen_data.py

wc -l  all_gen.jsonl
12000 all_gen.jsonl


------------------------

```

```
sh run_train_chunk.sh 8 8 8
```
