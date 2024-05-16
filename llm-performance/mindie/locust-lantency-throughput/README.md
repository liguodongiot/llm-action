




```
pip install locust
```



```
locust -f hello.py --headless --users 10 --spawn-rate 5 -H http://192.xxx.16.211:1025
```



- -r, --spawn-rate 生成用户的速率（每秒用户数）。主要与--headless或--autostart一起使用
- -u, --users  Locust 并发用户的峰值数量。主要与--headless或--autostart一起使用。
- --headless 关闭web界面，立即启动测试。使用-u和-t来控制用户数量和运行时间
- --autostart 立即开始测试（如 –-headless，但不禁用 Web UI）
- -t, --run-time 在指定时间后停止，例如（300s、20m、3h、1h30m 等）。仅与 –headless 或 –autostart 一起使用。默认永远运行。


```
Type     Name                                                        # reqs      # fails |    Avg     Min     Max    Med |   req/s  failures/s
--------|----------------------------------------------------------|-------|-------------|-------|-------|-------|-------|--------|-----------
GET      /v1/models                                                   50096     0(0.00%) |     10       1      33     11 |  765.60        0.00
--------|----------------------------------------------------------|-------|-------------|-------|-------|-------|-------|--------|-----------
         Aggregated                                                   50096     0(0.00%) |     10       1      33     11 |  765.60        0.00
```

```
Type：请求类型，即接口的请求方法；

Name：请求路径；

requests：当前已完成的请求数量；

fails：当前失败的数量；

Median：响应时间的中间值，即50%的响应时间在这个数值范围内，单位为毫秒；

Average：平均响应时间，单位为毫秒；

Min：最小响应时间，单位为毫秒；

Max：最大响应时间，单位为毫秒；

Content Size：所有请求的数据量，单位为字节；

reqs/sec：每秒钟处理请求的数量，即QPS；

failures/s：每秒钟处理请求失败的数量
```



```
locust -f 910b4-qwen.py --headless --users 20 --spawn-rate 20 -H http://192.xxx.16.211:1025 --run-time 5m --headless


locust -f 910b4-qwen.py --headless --users 100 --spawn-rate 10 -H http://192.xxx.16.211:1025 --run-time 10m --headless



locust -f 910b4-qwen.py --headless --users 10 --spawn-rate 5 -H http://192.xxx.16.211:1025


Type     Name                                                        # reqs      # fails |    Avg     Min     Max    Med |   req/s  failures/s
--------|----------------------------------------------------------|-------|-------------|-------|-------|-------|-------|--------|-----------
POST     /v1/chat/completions                                          1907     0(0.00%) |   4098     179    9106   2600 |    2.20        0.00
--------|----------------------------------------------------------|-------|-------------|-------|-------|-------|-------|--------|-----------
         Aggregated                                                    1907     0(0.00%) |   4098     179    9106   2600 |    2.20        0.00


```





## Qwen1.5-14B-4TP


- http://192.xxx.16.211:8089/



```
locust -f 910b4-qwen.py --users 20 --spawn-rate 20 -H http://192.xxx.16.211:1025 
```


```
locust -f 910b4-qwen.py --headless --users 20 --spawn-rate 20 -H http://192.xxx.16.211:1025 --run-time 10m 

list_token: 1786
prompt_tokens:  33157 completion_tokens:  231409 total_tokens:  264566


[2024-05-06 20:57:41,029] localhost.localdomain/INFO/locust.main: Shutting down (exit code 0)

Type     Name                                  # reqs      # fails |    Avg     Min     Max    Med |   req/s  failures/s
--------|------------------------------------|-------|-------------|-------|-------|-------|-------|--------|-----------
POST     /v1/chat/completions                    1786     0(0.00%) |   6665     188   14428   4200 |    2.98        0.00
--------|------------------------------------|-------|-------------|-------|-------|-------|-------|--------|-----------
         Aggregated                              1786     0(0.00%) |   6665     188   14428   4200 |    2.98        0.00

Response time percentiles (approximated)
Type     Name                                50%    66%    75%    80%    90%    95%    98%    99%  99.9% 99.99%   100% # reqs
--------|------------------------------|--------|------|------|------|------|------|------|------|------|------|------|------
POST     /v1/chat/completions               4200  12000  13000  13000  13000  13000  14000  14000  14000  14000  14000   1786
--------|------------------------------|--------|------|------|------|------|------|------|------|------|------|------|------
         Aggregated                         4200  12000  13000  13000  13000  13000  14000  14000  14000  14000  14000   1786
```





```
locust -f 910b4-qwen.py --users 100 --spawn-rate 100 -H http://192.xxx.16.211:1025 --run-time 10m 


list_token: 4265
prompt_tokens:  79384 completion_tokens:  538563 total_tokens:  617947


8tp:
locust -f qwen-8tp.py --users 100 --spawn-rate 100 -H http://192.xxx.16.211:1025 --run-time 10m 

list_token: 3577
prompt_tokens:  65916 completion_tokens:  458532 total_tokens:  524448



4tp:
list_token: 4285
prompt_tokens:  79461 completion_tokens:  537748 total_tokens:  617209



```


## Qwen1.5

### Qwen1.5-72B-8TP


```
locust -f 910b4-qwen.py --users 100 --spawn-rate 100 -H http://192.xxx.16.211:1025 --run-time 10m 


list_token: 1808
prompt_tokens:  33718 completion_tokens:  218677 total_tokens:  252395
```



### Qwen1.5-7B-1TP

```

```


### Qwen1.5-7B-2TP

```

locust -f 910b4-qwen.py --users 100 --spawn-rate 100 -H http://192.xxx.16.211:1025 --run-time 10m 



list_token: 5022
prompt_tokens:  93291 completion_tokens:  661053 total_tokens:  754344
```


### Qwen1.5-7B-4TP

```
locust -f 910b4-qwen.py --users 100 --spawn-rate 100 -H http://192.xxx.16.211:1025 --run-time 10m 


list_token: 4589
prompt_tokens:  85327 completion_tokens:  596528 total_tokens:  681855

```


## Qwen1-72B-8TP


```
locust -f llm-910b4-qwen-72b-8tp.py --users 100 --spawn-rate 100 -H http://192.xxx.16.211:1025 --run-time 10m 

list_token: 1888
prompt_tokens:  18585 completion_tokens:  253279 total_tokens:  271864
```



## Baichuan2
### Baichuan2-7B-2TP


```
locust -f llm-910b4-baichuan2-7b-2tp.py --users 100 --spawn-rate 100 -H http://192.xxx.16.211:1025 --run-time 10m 



locust -f llm-910b4-baichuan2-7b-2tp.py --users 1 --spawn-rate 1 -H http://192.xxx.16.211:1025 --run-time 10m 

list_token: 144
prompt_tokens:  1540 completion_tokens:  22404 total_tokens:  23944
```




### Baichuan2-7B-4TP


```
locust -f 910b4-qwen.py --users 100 --spawn-rate 100 -H http://192.xxx.16.211:1025 --run-time 10m 
```

### Baichuan2-14B-4TP


```
locust -f llm-910b4-baichuan2-7b-2tp.py --users 100 --spawn-rate 100 -H http://192.xxx.16.211:1025 --run-time 10m 



list_token: 4435
prompt_tokens:  49285 completion_tokens:  626468 total_tokens:  675753
```

### Baichuan2-14B-2TP


```
locust -f llm-910b4-baichuan2-7b-2tp.py --users 100 --spawn-rate 100 -H http://192.xxx.16.211:1025 --run-time 10m 

list_token: 2739
prompt_tokens:  30579 completion_tokens:  383276 total_tokens:  413855

```



## Chatglm3


### Chatglm3-6B-1TP

```
list_token: 10432
prompt_tokens:  183586 completion_tokens:  768828 total_tokens:  952414
```


### Chatglm3-6B-2TP

```
locust -f llm-910b4-chatglm3-6b-2tp.py --users 100 --spawn-rate 100 -H http://192.xxx.16.211:1025 --run-time 10m 

list_token: 6514
prompt_tokens:  114704 completion_tokens:  930099 total_tokens:  1044803
```




### Chatglm3-6B-4TP


```
locust -f llm-910b4-chatglm3-6b-2tp.py --users 100 --spawn-rate 100 -H http://192.xxx.16.211:1025 --run-time 10m 




list_token: 6436
prompt_tokens:  112556 completion_tokens:  915900 total_tokens:  1028456
```






