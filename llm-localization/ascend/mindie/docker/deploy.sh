#!/bin/bash

echo "入参:" $@

for a in "$@"; do
    #echo $a
    if [[ `echo $a | grep "^--model_name="` ]]; then
            model_name=`echo $a | grep "^--model_name=" | awk -F '=' '{print $2}'`
    fi
    if [[ `echo $a | grep "^--model_weight_path="` ]]; then
            model_weight_path=`echo $a | grep "^--model_weight_path=" | awk -F '=' '{print $2}'`
    fi
    if [[ `echo $a | grep "^--world_size="`  ]]; then
            world_size=`echo $a | grep "^--world_size=" | awk -F '=' '{print $2}'`
    fi
    if [[ `echo $a | grep "^--npu_mem_size="`  ]]; then
            npu_mem_size=`echo $a | grep "^--npu_mem_size=" | awk -F '=' '{print $2}'`
    fi
done

if [ -z "$model_name" ]; then
    model_name="default"
fi

if [ -z "$model_weight_path" ]; then
    model_weight_path="/workspace/models"
fi

if [ -z "$world_size" ]; then
    world_size=4
fi

if [ -z "$npu_mem_size" ]; then
    npu_mem_size=8
fi

echo "平台入参： model_name: $model_name, model_weight_path: $model_weight_path ， world_size: $world_size ， npu_mem_size: $npu_mem_size"


npuids=""
card_num=$(($world_size - 1))
for i in `seq 0 $card_num`
    do
        if [[ $i  == $card_num ]] ;
        then
            npuids=$npuids$i
        else
            npuids=$npuids$i","
        fi    
    done


echo $npuids 


DEPLOYMENT_CONF_PATH="/home/guodong.li/workspace/config.json"

# DEPLOYMENT_CONF_PATH="/usr/local/Ascend/mindie/latest/mindie-service/conf/config.json"

cat <<EOF > $DEPLOYMENT_CONF_PATH
{
    "OtherParam":
    {
        "ResourceParam" :
        {
            "cacheBlockSize" : 128,
            "preAllocBlocks" : 4
        },
        "LogParam" :
        {
            "logLevel" : "Info",
            "logPath" : "/logs/mindservice.log"
        },
        "ServeParam" :
        {
            "ipAddress" : "0.0.0.0",
            "port" : 1025,
            "maxLinkNum" : 300,
            "httpsEnabled" : false,
            "tlsCaPath" : "security/ca/",
            "tlsCaFile" : ["ca.pem"],
            "tlsCert" : "security/certs/server.pem",
            "tlsPk" : "security/keys/server.key.pem",
            "tlsPkPwd" : "security/pass/mindie_server_key_pwd.txt",
            "kmcKsfMaster" : "tools/pmt/master/ksfa",
            "kmcKsfStandby" : "tools/pmt/standby/ksfb",
            "tlsCrl" : "security/certs/server_crl.pem"
        }
    },
    "WorkFlowParam":
    {
        "TemplateParam" :
        {
            "templateType": "Standard",
            "templateName" : "Standard_llama",
            "pipelineNumber" : 1
        }
    },
    "ModelDeployParam":
    {
        "maxSeqLen" : 2560,
        "npuDeviceIds" : [[$npuids]],
        "ModelParam" : [
            {
                "modelInstanceType": "Standard",
                "modelName" : "$model_name",
                "modelWeightPath" : "$model_weight_path",
                "worldSize" : $world_size,
                "cpuMemSize" : 5,
                "npuMemSize" : $npu_mem_size,
                "backendType": "atb"
            }
        ]
    },
    "ScheduleParam":
    {
        "maxPrefillBatchSize" : 50,
        "maxPrefillTokens" : 8192,
        "prefillTimeMsPerReq" : 150,
        "prefillPolicyType" : 0,
        "decodeTimeMsPerReq" : 50,
        "decodePolicyType" : 0,
        "maxBatchSize" : 200,
        "maxIterTimes" : 512,
        "maxPreemptCount" : 200,
        "supportSelectBatch" : false,
        "maxQueueDelayMicroseconds" : 5000
    }
}
EOF

echo "部署参数，$DEPLOYMENT_CONF_PATH"
cat $DEPLOYMENT_CONF_PATH

# source /usr/local/Ascend/ascend-toolkit/set_env.sh
# source /usr/local/Ascend/mindie/set_env.sh 
# source /usr/local/Ascend/llm_model/set_env.sh

# export PYTHONPATH=/usr/local/Ascend/llm_model:$PYTHONPATH
# cd /usr/local/Ascend/mindie/latest/mindie-service/bin

# ./mindieservice_daemon
