

## 安装驱动与固件

先按照驱动，在安装固件。

```

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

