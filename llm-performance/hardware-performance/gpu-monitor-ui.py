
import psutil
import pynvml #导包

UNIT = 1024 * 1024

pynvml.nvmlInit() #初始化
gpuDeriveInfo = pynvml.nvmlSystemGetDriverVersion()
print("Drive版本: ", str(gpuDeriveInfo)) #显示驱动信息


gpuDeviceCount = pynvml.nvmlDeviceGetCount()#获取Nvidia GPU块数
print("GPU个数：", gpuDeviceCount )


for i in range(gpuDeviceCount):
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)#获取GPU i的handle，后续通过handle来处理

    memoryInfo = pynvml.nvmlDeviceGetMemoryInfo(handle)#通过handle获取GPU i的信息

    gpuName = str(pynvml.nvmlDeviceGetName(handle))

    gpuTemperature = pynvml.nvmlDeviceGetTemperature(handle, 0)

    # gpuFanSpeed = pynvml.nvmlDeviceGetFanSpeed(handle)
    gpuPowerState = pynvml.nvmlDeviceGetPowerState(handle)

    gpuUtilRate = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
    gpuMemoryRate = pynvml.nvmlDeviceGetUtilizationRates(handle).memory

    print("第 %d 张卡："%i, "-"*30)
    print("显卡名：", gpuName)
    print("内存总容量：", memoryInfo.total/UNIT, "MB")
    print("使用容量：", memoryInfo.used/UNIT, "MB")
    print("剩余容量：", memoryInfo.free/UNIT, "MB")
    print("显存空闲率：", memoryInfo.free/memoryInfo.total)
    print("温度：", gpuTemperature, "摄氏度")
    # print("风扇速率：", gpuFanSpeed)
    print("供电水平：", gpuPowerState)
    print("gpu计算核心满速使用率：", gpuUtilRate)
    print("gpu内存读写满速使用率：", gpuMemoryRate)
    print("内存占用率：", memoryInfo.used/memoryInfo.total)

    """
    # 设置显卡工作模式
    # 设置完显卡驱动模式后，需要重启才能生效
    # 0 为 WDDM模式，1为TCC 模式
    gpuMode = 0     # WDDM
    gpuMode = 1     # TCC
    pynvml.nvmlDeviceSetDriverModel(handle, gpuMode)
    # 很多显卡不支持设置模式，会报错
    # pynvml.nvml.NVMLError_NotSupported: Not Supported
    """

    # 对pid的gpu消耗进行统计
    pidAllInfo = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)#获取所有GPU上正在运行的进程信息
    for pidInfo in pidAllInfo:
        pidUser = psutil.Process(pidInfo.pid).username()
        print("进程pid：", pidInfo.pid, "用户名：", pidUser, 
            "显存占有：", pidInfo.usedGpuMemory/UNIT, "Mb") # 统计某pid使用的显存


pynvml.nvmlShutdown() #最后关闭管理工具