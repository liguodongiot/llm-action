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
    model_weight_path="/workspace/model"
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


ip=`hostname -I`

echo "docker ip: [$ip]"
ip=$(echo "$ip" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
echo "docker handle ip: [$ip]"

# DEPLOYMENT_CONF_PATH="/home/guodong.li/workspace/config.json"

DEPLOYMENT_CONF_PATH="/usr/local/Ascend/mindie/latest/mindie-service/conf/config.json"

cat <<EOF > $DEPLOYMENT_CONF_PATH
{
    "OtherParam" :
    {
        "ResourceParam" :
        {
            "cacheBlockSize" : 128
        },
        "LogParam" :
        {
            "logLevel" : "Info",
            "logPath" : "logs/mindservice.log"
        },
        "ServeParam" :
        {
            "ipAddress" : "$ip",
            "managementIpAddress" : "127.0.0.2",
            "port" : 1025,
            "managementPort" : 1026,
            "maxLinkNum" : 1000,
            "httpsEnabled" : false,
            "tlsCaPath" : "security/ca/",
            "tlsCaFile" : ["ca.pem"],
            "tlsCert" : "security/certs/server.pem",
            "tlsPk" : "security/keys/server.key.pem",
            "tlsPkPwd" : "security/pass/mindie_server_key_pwd.txt",
            "tlsCrl" : "security/certs/server_crl.pem",
            "managementTlsCaFile" : ["management_ca.pem"],
            "managementTlsCert" : "security/certs/management_server.pem",
            "managementTlsPk" : "security/keys/management_server.key.pem",
            "managementTlsPkPwd" : "security/pass/management_mindie_server_key_pwd.txt",
            "managementTlsCrl" : "security/certs/management_server_crl.pem",
            "kmcKsfMaster" : "tools/pmt/master/ksfa",
            "kmcKsfStandby" : "tools/pmt/standby/ksfb",
            "multiNodesInferPort" : 1120,
            "interNodeTLSEnabled" : true,
            "interNodeTlsCaFile" : "security/ca/ca.pem",
            "interNodeTlsCert" : "security/certs/server.pem",
            "interNodeTlsPk" : "security/keys/server.key.pem",
            "interNodeTlsPkPwd" : "security/pass/mindie_server_key_pwd.txt",
            "interNodeKmcKsfMaster" : "tools/pmt/master/ksfa",
            "interNodeKmcKsfStandby" : "tools/pmt/standby/ksfb"
        }
    },
    "WorkFlowParam" :
    {
        "TemplateParam" :
        {
            "templateType" : "Standard",
            "templateName" : "Standard_llama"
        }
    },
    "ModelDeployParam" :
    {
        "engineName" : "mindieservice_llm_engine",
        "modelInstanceNumber" : 1,
        "tokenizerProcessNumber" : 8,
        "maxSeqLen" : 2560,
        "npuDeviceIds" : [[$npuids]],
        "multiNodesInferEnabled" : false,
        "ModelParam" : [
            {
                "modelName" : "$model_name",
                "modelWeightPath" : "$model_weight_path",
                "worldSize" : $world_size,
                "cpuMemSize" : 5,
                "npuMemSize" : $npu_mem_size,
                "backendType": "atb",
                "pluginParams" : ""
            }
        ]
    },
    "ScheduleParam" :
    {
        "maxPrefillBatchSize" : 50,
        "maxPrefillTokens" : 8192,
        "prefillTimeMsPerReq" : 150,
        "prefillPolicyType" : 0,

        "decodeTimeMsPerReq" : 50,
        "decodePolicyType" : 0,

        "maxBatchSize" : 200,
        "maxIterTimes" : 512,
        "maxPreemptCount" : 0,
        "supportSelectBatch" : true,
        "maxQueueDelayMicroseconds" : 5000
    }
}
EOF

echo "部署参数，$DEPLOYMENT_CONF_PATH"
cat $DEPLOYMENT_CONF_PATH

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/mindie/set_env.sh
source /usr/local/Ascend/llm_model/set_env.sh

export MIES_PYTHON_LOG_TO_FILE=1
export MIES_PYTHON_LOG_TO_STDOUT=1
export PYTHONPATH=/usr/local/Ascend/llm_model:$PYTHONPATH
cd /usr/local/Ascend/mindie/latest/mindie-service/bin

./mindieservice_daemon
