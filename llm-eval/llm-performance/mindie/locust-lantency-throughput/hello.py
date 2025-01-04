
# Locust用户脚本就是Python模块  
import time  
from locust import HttpUser, task, between  

# 定义用户行为
# 类继承自HttpUser  
class QuickstartUser(HttpUser):  

	# 被@task装饰的才会并发执行  
    @task  
    def hello_world(self):  
        # client属性是HttpSession实例，用来发送HTTP请求  
        self.client.get("/v1/models")  
