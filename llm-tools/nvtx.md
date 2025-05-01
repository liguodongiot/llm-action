
- https://nvtx.readthedocs.io/en/latest/annotate.html


装饰器:

```
@nvtx.annotate(message="my_message", color="blue")
def my_func():
    pass
```


上下文管理器:

```
with nvtx.annotate(message="my_message", color="green"):
    pass
```


范围：

```
rng = nvtx.start_range(message="my_message", color="blue")
# ... do something ... #
nvtx.end_range(rng)

```

与start_range类似，但可以嵌套：

```
nvtx.push_range("batch " + str(i),"blue")

nvtx.pop_range()
```