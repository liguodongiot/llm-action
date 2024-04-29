
FROM ascendhub.huawei.com/public-ascendhub/mindie:1.0.RC1-800I-A2-aarch64

USER root

ENV APP_DIR=/workspace

RUN mkdir -p $APP_DIR

RUN cd /opt/package && ls -al && source ./install_and_enable_cann.sh

RUN pip install transformers==4.37.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
