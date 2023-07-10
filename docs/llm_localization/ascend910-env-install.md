

## 安装驱动与固件

先安装驱动，再安装固件。

```
chmod +x Ascend-hdk-910-npu-driver_23.0.rc1_linux-aarch64.run
./Ascend-hdk-910-npu-driver_23.0.rc1_linux-aarch64.run --full --install-path=/usr/local/Ascend
```

<details><summary>详细输出</summary><p>

```
Verifying archive integrity...  100%   SHA256 checksums are OK. All good.
Uncompressing ASCEND DRIVER RUN PACKAGE  100%  
[Driver] [2023-07-10 22:24:33] [INFO]Start time: 2023-07-10 22:24:33
[Driver] [2023-07-10 22:24:33] [INFO]LogFile: /var/log/ascend_seclog/ascend_install.log
[Driver] [2023-07-10 22:24:33] [INFO]OperationLogFile: /var/log/ascend_seclog/operation.log
[Driver] [2023-07-10 22:24:33] [WARNING]Do not power off or restart the system during the installation/upgrade
[Driver] [2023-07-10 22:24:33] [INFO]set username and usergroup, HwHiAiUser:HwHiAiUser
[Driver] [2023-07-10 22:24:34] [INFO]driver install type: Direct
[Driver] [2023-07-10 22:24:34] [INFO]upgradePercentage:10%
[Driver] [2023-07-10 22:24:38] [INFO]upgradePercentage:30%
[Driver] [2023-07-10 22:24:38] [INFO]upgradePercentage:40%
[Driver] [2023-07-10 22:24:40] [INFO]upgradePercentage:90%
[Driver] [2023-07-10 22:24:40] [INFO]Waiting for device startup...
[Driver] [2023-07-10 22:24:42] [INFO]Device startup success
[Driver] [2023-07-10 22:24:53] [INFO]upgradePercentage:100%
[Driver] [2023-07-10 22:25:04] [INFO]Driver package installed successfully! The new version takes effect immediately. 
[Driver] [2023-07-10 22:25:04] [INFO]End time: 2023-07-10 22:25:04
```

</p></details>


```
chmod +x Ascend-hdk-910-npu-firmware_6.3.0.1.241.run
./Ascend-hdk-910-npu-firmware_6.3.0.1.241.run --check
./Ascend-hdk-910-npu-firmware_6.3.0.1.241.run  --full
```

<details><summary>详细输出</summary><p>

```

Verifying archive integrity...  100%   SHA256 checksums are OK. All good.
Uncompressing ASCEND-HDK-910-NPU FIRMWARE RUN PACKAGE  100%  
[Firmware] [2023-07-10 22:27:29] [INFO]Start time: 2023-07-10 22:27:29
[Firmware] [2023-07-10 22:27:29] [INFO]LogFile: /var/log/ascend_seclog/ascend_install.log
[Firmware] [2023-07-10 22:27:29] [INFO]OperationLogFile: /var/log/ascend_seclog/operation.log
[Firmware] [2023-07-10 22:27:29] [WARNING]Do not power off or restart the system during the installation/upgrade
[Firmware] [2023-07-10 22:27:34] [INFO]upgradePercentage: 0%
[Firmware] [2023-07-10 22:27:42] [INFO]upgradePercentage: 0%
[Firmware] [2023-07-10 22:27:51] [INFO]upgradePercentage: 0%
[Firmware] [2023-07-10 22:27:57] [INFO]upgradePercentage: 100%
[Firmware] [2023-07-10 22:27:57] [INFO]The firmware of [8] chips are successfully upgraded.
[Firmware] [2023-07-10 22:27:57] [INFO]Firmware package installed successfully! Reboot now or after driver installation for the installation/upgrade to take effect.
[Firmware] [2023-07-10 22:27:57] [INFO]End time: 2023-07-10 22:27:57
```

</p></details>


## 卸载固件与驱动

驱动和固件的卸载没有先后顺序要求。

查询安装路径：
```
> cat /etc/ascend_install.info
UserName=HwHiAiUser
UserGroup=HwHiAiUser
Firmware_Install_Type=full
Firmware_Install_Path_Param=/usr/local/Ascend
Driver_Install_Type=full
Driver_Install_Path_Param=/usr/local/Ascend
Driver_Install_For_All=yes
```

卸载固件：
```
# /usr/local/Ascend/firmware/script/uninstall.sh --uninstall
/usr/local/Ascend/firmware/script/uninstall.sh --quiet
```

卸载驱动：
```
# /usr/local/Ascend/driver/script/uninstall.sh  --uninstall
/usr/local/Ascend/driver/script/uninstall.sh  --quiet
```

卸载驱动和固件后，根据系统提示信息决定是否重启服务器，若需要重启系统，请执行以下命令；否则，请跳过此步骤。

```
reboot
```



## 安装CANN

```
wget -c https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%206.3.RC1/Ascend-cann-kernels-910_6.3.RC1_linux.run
wget -c https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%206.3.RC1/Ascend-cann-toolkit_6.3.RC1_linux-aarch64.run
```




## 安装bypy，下载百度网盘数据

```
安装模块
pip install bypy -i https://pypi.douban.com/simple

下载数据

1.获取网页

执行下面的命令，用浏览器打开please visit下面的网页

bypy info

2.登录账号获取秘钥

3.返回linux终端输入秘钥

4. 文件转移

授权之后，你的百度云盘的我的应用数据目录下后多出一个bypy文件夹，将你要下载的数据移动到里面即可

5. 下载数据
bypy downdir -v
```


## 安装 Dokcer

```
# https://www.hiascend.com/document/detail/zh/quick-installation/23.0.RC1/quickinstg/800_3010/quickinstg_800_3010_0028.html
# https://www.hiascend.com/document/detail/zh/quick-installation/23.0.RC1/quickinstg/800_3010/quickinstg_800_3010_0012.html

yum makecache
yum install -y docker
systemctl start docker
```



## 安装 git-lfs

EulerOS和openEuler使用以下命令安装：
```
# 根据系统架构选择相应的版本下载。
curl -OL https://github.com/git-lfs/git-lfs/releases/download/v3.1.2/git-lfs-linux-arm64-v3.1.2.tar.gz

# 解压并安装。
mkdir git-lfs
tar xf git-lfs-linux-*-v3.1.2.tar.gz -C git-lfs
cd git-lfs
sudo bash install.sh
```



## SSH免密登录

```
# 本机免密
ssh-keygen -t rsa
cp ~/.ssh/id_rsa.pub ~/.ssh/authorized_keys

# 多机免密
scp -r /root/.ssh/ 183.66.251.xxx:/root
ssh 183.66.251.xxx
```

