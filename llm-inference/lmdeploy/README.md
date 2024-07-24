


- https://hub.docker.com/r/openmmlab/lmdeploy-builder/tags
- https://hub.docker.com/r/openmmlab/lmdeploy/tags


- https://github.com/InternLM/lmdeploy/blob/main/docs/zh_cn/serving/api_server.md
- https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/cli/utils.py#L64





请求队列
- 推理请求首先先加入到请求队列中
Persistent线程
1. 若批处理中有空闲槽位，从队列拉取请求，尽量填满空闲槽位。若无，继续对当前批处理中的请求进行Forward
2. 批次每Forward完一次
- 判断是否有请求推理结束。结束的请求，发送结果，释放槽位
- 转步骤1