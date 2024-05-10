FROM ascendhub.huawei.com/public-ascendhub/mindie:1.0.RC1-800I-A2-aarch64

RUN cd /opt/package && source install_and_enable_cann.sh \
&& source /usr/local/Ascend/ascend-toolkit/set_env.sh \
&& source /usr/local/Ascend/mindie/set_env.sh \
&& source /usr/local/Ascend/llm_model/set_env.sh






