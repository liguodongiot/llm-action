## 查看操作系统
```
>cat /etc/os-release
PRETTY_NAME="Ubuntu 22.04.3 LTS"
NAME="Ubuntu"
VERSION_ID="22.04"
VERSION="22.04.3 LTS (Jammy Jellyfish)"
VERSION_CODENAME=jammy
ID=ubuntu
ID_LIKE=debian
HOME_URL="https://www.ubuntu.com/"
SUPPORT_URL="https://help.ubuntu.com/"
BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
UBUNTU_CODENAME=jammy

> uname -a
Linux nodo-1 5.15.0-78-generic #85-Ubuntu SMP Fri Jul 7 15:29:30 UTC 2023 aarch64 aarch64 aarch64 GNU/Linux
```

## 查看服务器的型号信息
```

# yum install dmidecode

> apt-get install dmidecode

> dmidecode -t system
# dmidecode 3.3
Getting SMBIOS data from sysfs.
SMBIOS 3.3.0 present.

Handle 0x0001, DMI type 1, 27 bytes
System Information
        Manufacturer: Huawei
        Product Name: A800 9000 A2
        Version: To be filled by O.E.M.
        Serial Number: 2102314RHT10P81000xx
        UUID: a61414fb-b04f-b618-ee11-c0441xx24b6d
        Wake-up Type: Power Switch
        SKU Number: To be filled by O.E.M.
        Family: To be filled by O.E.M.

Handle 0x0005, DMI type 32, 11 bytes
System Boot Information
        Status: No errors detected
```



以下操作如无特别说明，都是针对EulerOS系统。

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


安装驱动和固件后，根据系统提示信息决定是否重启服务器，若需要重启系统，请执行以下命令；否则，请跳过此步骤。

```
reboot
```


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


## 安装开发工具集（CANN）

下载：
```
wget -c https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%206.3.RC1/Ascend-cann-kernels-910_6.3.RC1_linux.run
wget -c https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%206.3.RC1/Ascend-cann-toolkit_6.3.RC1_linux-aarch64.run
```

安装：
```
chmod +x Ascend-cann-toolkit_6.3.RC1_linux-aarch64.run
./Ascend-cann-toolkit_6.3.RC1_linux-aarch64.run --install --install-for-all
```

<details><summary>详细输出</summary><p>


```
Verifying archive integrity...  100%   SHA256 checksums are OK. All good.
Uncompressing ASCEND_RUN_PACKAGE  100%  
[Toolkit] [20230710-22:44:15] [INFO] LogFile:/var/log/ascend_seclog/ascend_toolkit_install.log
[Toolkit] [20230710-22:44:15] [INFO] install start
[Toolkit] [20230710-22:44:15] [INFO] The installation path is /usr/local/Ascend.
[Toolkit] [20230710-22:44:15] [INFO] install package CANN-runtime-6.3.0.1.241-linux.aarch64.run start
[Toolkit] [20230710-22:44:21] [INFO] CANN-runtime-6.3.0.1.241-linux.aarch64.run --full --quiet --nox11 --install-for-all install success
[Toolkit] [20230710-22:44:21] [INFO] install package CANN-compiler-6.3.0.1.241-linux.aarch64.run start
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv
[Toolkit] [20230710-22:44:56] [INFO] CANN-compiler-6.3.0.1.241-linux.aarch64.run --full --pylocal --quiet --nox11 --install-for-all install success
[Toolkit] [20230710-22:44:56] [INFO] install package CANN-opp-6.3.0.1.241-linux.aarch64.run start
[Toolkit] [20230710-22:45:39] [INFO] CANN-opp-6.3.0.1.241-linux.aarch64.run --full --quiet --nox11 --install-for-all install success
[Toolkit] [20230710-22:45:39] [INFO] install package CANN-toolkit-6.3.0.1.241-linux.aarch64.run start
[Toolkit] [20230710-22:46:20] [INFO] CANN-toolkit-6.3.0.1.241-linux.aarch64.run --full --pylocal --quiet --nox11 --install-for-all install success
[Toolkit] [20230710-22:46:20] [INFO] install package CANN-aoe-6.3.0.1.241-linux.aarch64.run start
[Toolkit] [20230710-22:46:23] [INFO] CANN-aoe-6.3.0.1.241-linux.aarch64.run --full --quiet --nox11 --install-for-all install success
[Toolkit] [20230710-22:46:23] [INFO] install package Ascend-mindstudio-toolkit_6.0.RC1_linux-aarch64.run start
[Toolkit] [20230710-22:46:30] [INFO] Ascend-mindstudio-toolkit_6.0.RC1_linux-aarch64.run --full --quiet --nox11 --install-for-all install success
[Toolkit] [20230710-22:46:30] [INFO] install package Ascend-test-ops_6.3.RC1_linux.run start
[Toolkit] [20230710-22:46:30] [INFO] Ascend-test-ops_6.3.RC1_linux.run --full --quiet --nox11 --install-for-all install success
[Toolkit] [20230710-22:46:30] [INFO] install package Ascend-pyACL_6.3.RC1_linux-aarch64.run start
[Toolkit] [20230710-22:46:30] [INFO] Ascend-pyACL_6.3.RC1_linux-aarch64.run --full --quiet --nox11 --install-for-all install success
[Toolkit] [20230710-22:46:30] [INFO] install package CANN-ncs-6.3.0.1.241-linux.aarch64.run start
[Toolkit] [20230710-22:46:33] [INFO] CANN-ncs-6.3.0.1.241-linux.aarch64.run --full --quiet --nox11 --install-for-all install success


===========
= Summary =
===========

Driver:   Installed in /usr/local/Ascend/driver.
Toolkit:  Ascend-cann-toolkit_6.3.RC1_linux-aarch64 install success, installed in /usr/local/Ascend.

Please make sure that the environment variables have been configured.
-  To take effect for all users, you can add "source /usr/local/Ascend/ascend-toolkit/set_env.sh" to /etc/profile.
-  To take effect for current user, you can exec command below: source /usr/local/Ascend/ascend-toolkit/set_env.sh or add "source /usr/local/Ascend/ascend-toolkit/set_env.sh" to ~/.bashrc.
```

</p></details>



需要注意的是，NPU驱动、固件以及CANN软件安装完成之后，需**设置环境变量**：

如果昇腾AI处理器配套软件包安装在默认路径，则直接使用如下命名直接设置即可。

```
# 建议配置在~/.bashrc中
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
```

如果昇腾AI处理器配套软件包**没有安装在默认路径**，安装好 MindSpore 之后，需要导出Runtime相关环境变量，下述命令中`LOCAL_ASCEND=/usr/local/Ascend`的`/usr/local/Ascend`表示配套软件包的安装路径，**需将其改为配套软件包的实际安装路径**。

```
# control log level. 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR, 4-CRITICAL, default level is WARNING.
export GLOG_v=2

# Conda environmental options
LOCAL_ASCEND=/usr/local/Ascend # the root directory of run package

# lib libraries that the run package depends on
export LD_LIBRARY_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/lib64:${LOCAL_ASCEND}/driver/lib64:${LOCAL_ASCEND}/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/op_tiling:${LD_LIBRARY_PATH}

# Environment variables that must be configured
## TBE operator implementation tool path
export TBE_IMPL_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe
## OPP path
export ASCEND_OPP_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp
## AICPU path
export ASCEND_AICPU_PATH=${ASCEND_OPP_PATH}/..
## TBE operator compilation tool path
export PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/compiler/ccec_compiler/bin/:${PATH}
## Python library that TBE implementation depends on
export PYTHONPATH=${TBE_IMPL_PATH}:${PYTHONPATH}
```

## 升级GCC以及安装Cmake

```
# 安装GCC
sudo yum install gcc -y


# 安装Cmake， aarch64使用
curl -O https://cmake.org/files/v3.19/cmake-3.19.8-Linux-aarch64.sh

sudo mkdir /usr/local/cmake-3.19.8
sudo bash cmake-3.19.8-Linux-*.sh --prefix=/usr/local/cmake-3.19.8 --exclude-subdir


echo -e "export PATH=/usr/local/cmake-3.19.8/bin:\$PATH" >> ~/.bashrc
source ~/.bashrc
```


## 安装 bypy，下载百度网盘数据

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



## 安装 tree

```
yum install -y tree
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

