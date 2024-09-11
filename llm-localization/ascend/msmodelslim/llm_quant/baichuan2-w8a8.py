import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig, AntiOutlier
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
import json
import argparse

def init_tokenizer(input_model_path:str):
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=input_model_path,
        # use_fast=False,
        # padding_side='left',
        trust_remote_code=True)
    return  tokenizer

def init_model(input_model_path:str):
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=input_model_path,
        trust_remote_code=True).float().cpu()
    model = model.half().npu() # 如果需要使用npu进行量化
    return model

# 获取校准数据函数定义
def get_calib_dataset(
        auto_tokenizer,
        calib_list,
        device="cpu"):  # 如果需要使用npu进行量化, device="npu:0"。使用cpu,device="cpu"
    calib_dataset = []
    for calib_data in calib_list:
        inputs = auto_tokenizer(calib_data, return_tensors='pt')
        calib_dataset.append([
            inputs.data['input_ids'].to(device),
            inputs.data['attention_mask'].to(device)
        ])
    return calib_dataset

def load_dataset(calib_set_path = "./calib_set.json"):
    calib_set = json.load(open(calib_set_path, "r"))
    return calib_set

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_model_path', type=str, help="input model and tokenizer path")
    parser.add_argument('--output_model_path', type=str, help="output model and tokenizer path")
    parser.add_argument('--calib_set_path', type=str, default="./calib_set.json", help="calib set path")
    
    return parser.parse_args()

def disable_quant_module(input_model_path:str):
    config = json.load(open(input_model_path+"/config.json", "r"))
    num_hidden_layers = config.get("num_hidden_layers", 0)
    disable_names = []

    disable_idx_lst = list(range(num_hidden_layers))
    for layer_index in disable_idx_lst:
        down_proj_name = "model.layers.{}.mlp.down_proj".format(layer_index)
        disable_names.append(down_proj_name)

    # 仅7B禁用
    # BAICHUAN_LAYERS = 32
    if int(num_hidden_layers) == 32:
        disable_last_linear = False
    else:
        # 13B
        disable_last_linear = True

    print("disable_last_linear: ", disable_last_linear, "disable_names: \n", disable_names)
    return disable_names, disable_last_linear

if __name__ == '__main__':
    args = parse_arguments()
    
    tokenizer = init_tokenizer(input_model_path=args.input_model_path)
    model = init_model(input_model_path=args.input_model_path)
    calib_set = load_dataset(calib_set_path=args.calib_set_path)
    
    dataset_calib = get_calib_dataset(tokenizer, calib_set, device="npu:0")

    # 对于linear算子中的激活值如果有表示范围过大，或者“尖刺”的异常值过多，
    # 需要使用anti outleir功能，使用方法如下

    logging.info("===============start AntiOutlier==============")
    
    anti_config = AntiOutlierConfig(
        w_bit=8, 
        a_bit=8, 
        anti_method="m2",
        #dev_type="cpu",
        dev_type="npu", dev_id=0
        )  # dev_type="npu", dev_id=0  如果需要使用npu进行量化。

    anti_outlier = AntiOutlier(model,
                            calib_data=dataset_calib,
                            cfg=anti_config,
                            norm_class_name="RMSNorm")
    anti_outlier.process()

    #下面是回退层的设置，因为w8a8的对激活值也进行了量化，会有部分网络层对激活值的表示
    #范围较为敏感所以需要回退这些网络层使用浮点权重进行计算。

    logging.info("===============end AntiOutlier==============")

    disable_names, disable_last_linear = disable_quant_module(args.input_model_path)

    quant_config = QuantConfig(
        a_bit=8,
        w_bit=8,
        disable_names=disable_names,
        disable_last_linear=disable_last_linear,
        #dev_type='cpu',  # dev_type="npu", dev_id=0,  如果需要使用npu进行量化
        dev_type="npu", dev_id=0,
        act_method=3,
        pr=1.0,
        w_sym=True,
        mm_tensor=False)

    logging.info("===============start Calibrator==============")
    calibrator = Calibrator(model,
                            quant_config,
                            calib_data=dataset_calib,
                            disable_level='L0')
    calibrator.run()  # 执行PTQ量化校准

    calibrator.save(args.output_model_path, 
        save_type=[ "safe_tensor"]
    )  # "safe_tensor"对应safetensors格式权重，"numpy"对应npy格式权重
    logging.info("===============end Calibrator==============")
