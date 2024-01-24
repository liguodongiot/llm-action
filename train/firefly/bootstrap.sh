
#!/bin/bash
 
func() {
    echo "Usage:"
    echo "test.sh [-d TRAIN_DATASET_PATH] [-p PRE_MODEL_PATH] [-o MODEL_OUTPUT_PATH] [-m MODEL_METRICS_PATH]"
    echo "Description:"
    echo "TRAIN_DATASET_PATH, 训练数据集路径."
    echo "PRE_MODEL_PATH, 预训练模型路径."
    echo "MODEL_OUTPUT_PATH, 模型输出模型."
    echo "MODEL_METRICS_PATH, 模型指标路径."
    exit -1
}

 
while getopts 'd:p:o:m' OPT; do	
    case $OPT in
        d) TRAIN_DATASET_PATH="$OPTARG";;
        p) PRE_MODEL_PATH="$OPTARG";;
        o) MODEL_OUTPUT_PATH="$OPTARG";;
        m) MODEL_METRICS_PATH="$OPTARG";;
		h) func;;
		?) func;;
    esac
done
 
echo "TRAIN_DATASET_PATH: $TRAIN_DATASET_PATH ，PRE_MODEL_PATH：$PRE_MODEL_PATH ， MODEL_OUTPUT_PATH：$MODEL_OUTPUT_PATH ，MODEL_METRICS_PATH：$MODEL_METRICS_PATH"



