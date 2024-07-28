




docker: Error response from daemon: failed to create shim task: OCI runtime create failed: unable to retrieve OCI runtime error (open /var/run/docker/containerd/daemon/io.containerd.runtime.v2.task/moby/579418211a825ef5c7fcf5becdbe90804f0ed7862d9c59663995f9dd463937b4/log.json: no such file or directory): /usr/local/Ascend/Ascend-Docker-Runtime/ascend-docker-runtime did not terminate successfully: exit status 1: 2024/07/24 09:59:29 owner not right /usr/bin/runc 1000




错误信息表明/usr/bin/runc这个文件的所有权不正确，即它不是由root用户拥有或者它的所属用户不是1000。Docker在创建并运行容器时需要runc这个二进制文件，如果权限设置不当，Docker将无法正确执行。


解决办法：


查看权限

ls -lah /usr/bin/runc 


修改权限

sudo chown root:root /usr/bin/runc

