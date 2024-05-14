



```
docker pull codegeex/codegeex:latest
# To enable GPU support, clarify device ids with --device
docker run --gpus '"device=0,1"' -it --ipc=host --name=codegeex codegeex/codegeex
```

```
git clone git@github.com:THUDM/CodeGeeX.git
cd CodeGeeX
pip install -e .
```


