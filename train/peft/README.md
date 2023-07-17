

## 环境

```
pip install jupyterlab
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







