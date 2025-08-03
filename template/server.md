



优云智算-公共模型目录：/model/

```
base_path=`pwd`

mkdir -p "$base_path/workspace"
mkdir -p "$base_path/workspace/code"

```


/model/ModelScope/Qwen/Qwen3-0.6B




```
pip install transformers accelerate seaborn matplotlib



conda deactivate

# 将虚拟环境添加到 Jupyter lab内核
python -m ipykernel install --user  --name=py310

# 确认已经成功添加
jupyter kernelspec list
```



```
pip install torch==2.4.0 transformers accelerate matplotlib


nohup jupyter lab --allow-root --no-browser --ip=0.0.0.0 --port=8888 > jupyter.log 2>&1 & 
sleep 2
tail -100f jupyter.log | grep http://

http://117.50.213.xxx:8888/lab?token=e6d93f34f936c4a485d06f6ca267614ae09e585497adbeae
```



```
git config --global user.email "liguodongiot@foxmail.com"
git config --global user.name "wintfru"
```

```
scp -r lambada root@ucloud:/root/workspace/data
```



