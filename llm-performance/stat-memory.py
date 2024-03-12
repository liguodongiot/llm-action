import psutil
import pynvml 
import time


UNIT = 1024 * 1024

pynvml.nvmlInit() #初始化

ids = [3, 7]

max_mem_dict = {

}

while True:
    for i in ids:
        max_mem = max_mem_dict.get(str(i), 0)
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        memoryInfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        
        used_mem = memoryInfo.used/UNIT
        if used_mem > max_mem:
            max_mem_dict[str(i)] = used_mem
            print(max_mem_dict)
    time.sleep(5)

pynvml.nvmlShutdown()