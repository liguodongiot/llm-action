


RuntimeError: Start method 'spawn' was requested, but 'fork' was already set.

- 解决方法：https://sanic.dev/en/guide/running/manager.html#sanic-and-start-methods
- python多进程(一)Fork模式和Spawn模式的优缺点: https://blog.csdn.net/weixin_42575811/article/details/134041691





- https://ida3.cn/zh/guide/deployment/running.html#sanic-%E6%9C%8D%E5%8A%A1%E5%99%A8-sanic-server


sanic server.app --host=0.0.0.0 --port=1337 --workers=4
