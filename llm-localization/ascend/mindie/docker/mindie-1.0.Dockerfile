#FROM ascendhub.huawei.com/public-ascendhub/mindie-service-env:1.0.RC1-800I-A2-aarch64
FROM ascendhub.huawei.com/public-ascendhub/mindie-service-env:v2

ENV APP_DIR=/workspace

RUN mkdir -p $APP_DIR

# COPY qwen1.5-14b.json /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
COPY baichuan2-7b.json /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json

COPY llm-server.sh $APP_DIR

RUN  chmod -R 777 $APP_DIR/llm-server.sh

ENTRYPOINT $APP_DIR/llm-server.sh

# docker build --network=host  -f mindie-1.0.Dockerfile -t ascendhub.huawei.com/public-ascendhub/mindie-service-online:v1.0 .
# docker build --network=host  -f mindie-1.0.Dockerfile -t ascendhub.huawei.com/public-ascendhub/mindie-service-online:v1.1 .