
# Locust用户脚本就是Python模块  
import time  
from locust import HttpUser, task, between  

# 定义用户行为
# 类继承自HttpUser  
class QuickstartUser(HttpUser):  

    # 每个模拟用户等待1~2.5秒  
    wait_time = between(1, 2)  

    # 被@task装饰的才会并发执行  
    @task  
    def hello_world(self):  
        # client属性是HttpSession实例，用来发送HTTP请求  
        self.client.get("/hello")  
        self.client.get("/world")  

    # 每个类只会有一个task被选中执行  
    # 3代表weight权重  
    # 权重越大越容易被选中执行  
    # view_items比hello_wolrd多3倍概率被选中执行  
    @task(3)  
    def view_items(self):  
        for item_id in range(10):  
            # name参数作用是把统计结果按同一名称进行分组  
            # 这里防止URL参数不同会产生10个不同记录不便于观察  
            # 把10个汇总成1个"/item"记录  
            self.client.get(f"/item?id={item_id}", name="/item")  
            time.sleep(1)  

    # 每个模拟用户开始运行时都会执行  
    def on_start(self):  
        self.client.post("/login", json={"username":"foo", "password":"bar"})



# 用于设置性能测试

class WebsiteUser(HttpLocust):

    # 指向一个定义的用户行为类。
    task_set = UserBehavior

    # 执行事务之间用户等待时间的下界（单位：毫秒）。如果TaskSet类中有覆盖，以TaskSet 中的定义为准。

    min_wait = 3000

    # 执行事务之间用户等待时间的上界（单位：毫秒）。如果TaskSet类中有覆盖，以TaskSet中的定义为准。

    max_wait= 6000

    # 设置 Locust 多少秒后超时，如果为 None ,则不会超时。

    stop_timeout = 5

    # 一个Locust实例被挑选执行的权重，数值越大，执行频率越高。在一个 locustfile.py 文件中可以同时定义多个 HttpLocust 子类，然后分配他们的执行权重

    weight = 3

    # 脚本指定host执行测试时则不在需要指定

    host = "https://www.baidu.com"
