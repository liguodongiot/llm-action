import psutil
import pynvml 
import time

UNIT = 1024 * 1024

pynvml.nvmlInit() #初始化

ids = [3, 7]

max_mem_dict = {

}

num = 0
while True:
    for i in ids:
        max_mem = max_mem_dict.get(str(i), 0)
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        memoryInfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_mem = memoryInfo.used/UNIT
        if num % 12 == 0:
            num = 0
            print("使用容量：", memoryInfo.used/UNIT, "MB, ", "剩余容量：", memoryInfo.free/UNIT, "MB")
        if used_mem > max_mem:
            max_mem_dict[str(i)] = used_mem
            print(max_mem_dict)
    time.sleep(5)
    num += 1

pynvml.nvmlShutdown()


