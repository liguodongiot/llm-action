

```
https://hf-mirror.com/

yum install python3-pip



virtualenv -p python3 venv-py3
source /home/aicc/venv-py3/bin/activate




pip3 install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com



huggingface-cli download --token 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 --resume-download Baichuan2-7B-Base --local-dir Baichuan2-7B-Base

huggingface-cli download --token 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 --resume-download baichuan-inc/Baichuan2-7B-Chat --local-dir Baichuan2-7B-Chat --local-dir-use-symlinks False


nohup huggingface-cli download --token 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 --resume-download baichuan-inc/Baichuan2-7B-Chat --local-dir Baichuan2-7B-Chat --local-dir-use-symlinks False > Baichuan2.log 2>&1 &



nohup huggingface-cli download --token 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 --resume-download THUDM/chatglm3-6b --local-dir chatglm3-6b-chat --local-dir-use-symlinks False > chatglm3.log 2>&1 &


export HF_ENDPOINT=https://hf-mirror.com

nohup huggingface-cli download --token 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 --resume-download Qwen/Qwen-72B-Chat --local-dir Qwen-72B-Chat --local-dir-use-symlinks False > qwen-72b.log 2>&1 &


```




```
cd /home/aicc

mkdir -p ./model_from_hf/Baichuan2-7B-Chat/
cd ./model_from_hf/Baichuan2-7B-Chat/
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/resolve/main/config.json
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/resolve/main/configuration_baichuan.py
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/resolve/main/generation_utils.py
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/resolve/main/modeling_baichuan.py
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/resolve/main/pytorch_model-00001-of-00002.bin
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/resolve/main/pytorch_model-00002-of-00002.bin
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/resolve/main/pytorch_model.bin.index.json
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/resolve/main/quantizer.py
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/resolve/main/special_tokens_map.json
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/resolve/main/tokenization_baichuan.py
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/resolve/main/tokenizer.model
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/resolve/main/tokenizer_config.json
cd ../../
```







```
cd /home/aicc

mkdir -p ./model_from_hf/Baichuan2-7B-Chat/
cd ./model_from_hf/Baichuan2-7B-Chat/
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat/resolve/main/config.json
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat/resolve/main/configuration_baichuan.py
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat/resolve/main/generation_config.json
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat/resolve/main/generation_utils.py
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat/resolve/main/modeling_baichuan.py
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat/resolve/main/pytorch_model.bin
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat/resolve/main/quantizer.py
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat/resolve/main/special_tokens_map.json
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat/resolve/main/tokenization_baichuan.py
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat/resolve/main/tokenizer.model
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat/resolve/main/tokenizer_config.json
cd ../../

```