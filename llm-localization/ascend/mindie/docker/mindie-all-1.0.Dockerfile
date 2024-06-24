
FROM ascendhub.huawei.com/public-ascendhub/mindie:1.0.RC1-800I-A2-aarch64

# USER root

COPY driver /usr/local/Ascend/driver

RUN ls -al /usr/local/Ascend/driver

ENV APP_DIR=/workspace

RUN mkdir -p $APP_DIR

COPY install_and_enable_cann.sh /opt/package/install_and_enable_cann.sh

RUN cd /opt/package && ls -al && cat /opt/package/install_and_enable_cann.sh && source ./install_and_enable_cann.sh

RUN pip install transformers==4.37.2 -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY qwen1.5-14b.json /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json

COPY llm-server.sh $APP_DIR

RUN  chmod -R 777 $APP_DIR/llm-server.sh

ENTRYPOINT ["$APP_DIR/llm-server.sh"]
