

## FAQ


### baichuan2报错

- 'BitsAndBytesConfig' object is not subscriptable

https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat/discussions/2



- AttributeError: 'BaichuanTokenizer' object has no attribute 'sp_model'

降低版本到4.34.0及以下 ： pip install transformers==4.34.0



## Pytorch

- RuntimeError: DataLoader worker (pid xxxxx) is killed by signal: Killed.

可能共享内存太小了

--shm-size 4G 



