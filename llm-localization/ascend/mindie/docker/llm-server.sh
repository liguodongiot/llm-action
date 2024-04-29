
#!/bin/bash

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/mindie/set_env.sh 
source /usr/local/Ascend/llm_model/set_env.sh


export PYTHONPATH=/usr/local/Ascend/llm_model:$PYTHONPATH
cd /usr/local/Ascend/mindie/latest/mindie-service/bin

./mindieservice_daemon




