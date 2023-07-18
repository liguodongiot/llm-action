

## 环境


```
cd /home/guodong.li/virtual-venv
virtualenv -p /usr/bin/python3.10 peft-venv-py310-cu117
source /home/guodong.li/virtual-venv/peft-venv-py310-cu117/bin/activate


pip install torch-1.13.1+cu117-cp310-cp310-linux_x86_64.whl
pip install torchvision-0.14.1+cu117-cp310-cp310-linux_x86_64.whl

git clone https://github.com/huggingface/peft
cd peft
git checkout 42ab106
pip install -e .

pip install datasets

pip install jupyterlab

pip install deepspeed
```

生成配置文件：
```
> jupyter lab --generate-config
Writing default config to: /home/guodong.li/.jupyter/jupyter_lab_config.py
```

对密码进行加密：
```
from jupyter_server.auth import passwd; passwd()
```


修改配置文件：
```
c.ServerApp.allow_origin = '*'
c.ServerApp.allow_remote_access = True
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.open_browser = False  
c.ServerApp.password = '加密后的密码'
c.ServerApp.port = 9999
```

启动：
```
jupyter lab --allow-root
nohup jupyter lab --allow-root > jupyterlab.log 2>&1 &
```


## 高效微调

### 大模型参数高效微调技术实战（一）-Prefix Tuning 
### 大模型参数高效微调技术实战（二）-Prompt Tuning
### 大模型参数高效微调技术实战（三）-P-Tuning

### 大模型参数高效微调技术实战（三）-LoRA


<details><summary>详细输出：</summary><p>
```

```
</p></details>



```
> tree -h /data/nfs/llm/model/bloomz-560m_LORA_CAUSAL_LM
/data/nfs/llm/model/bloomz-560m_LORA_CAUSAL_LM
├── [ 447]  adapter_config.json
├── [ 14K]  adapter_model.bin
└── [  93]  README.md

0 directories, 3 files
```

### 大模型参数高效微调技术实战（四）-AdaLoRA


### 大模型参数高效微调技术实战（五）-QLoRA










- 大模型参数高效微调技术实战（一）-Prefix Tuning 
- 大模型参数高效微调技术实战（二）-Prompt Tuning
- 大模型参数高效微调技术实战（三）-P-Tuning
- 大模型参数高效微调技术实战（三）-LoRA
- 大模型参数高效微调技术实战（四）-AdaLoRA
- 大模型参数高效微调技术实战（五）-QLoRA


