
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
import argparse
import json

def load_tokenizer_and_model(fp16_path):
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=fp16_path,
        pad_token='<|extra_0|>',
        eos_token='<|endoftext|>',
        padding_side='left',
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=fp16_path,
        torch_dtype=torch.float32, trust_remote_code=True
    ).cpu()
    return tokenizer, model


def main(fp16_path, quant_save_path, calib_set_path):
    tokenizer, model = load_tokenizer_and_model(fp16_path)

    disable_names = ['lm_head']

    quant_config = QuantConfig(
        w_bit=8,                        # 权重量化位数
        a_bit=16,                       # 激活值量化位数
        disable_names=disable_names,    # 不做量化的层
        dev_type='cpu',
        pr=1.0,                         # 量化正则百分比
        w_sym=True,                     # 对称/非对称量化，True为对称量化，False为非对称量化
        mm_tensor=False                 # 权重量化粒度，True为per-tensor量化，False为per-channel量化（大模型场景建议False）
    )

    calibrator = Calibrator(
        model,
        quant_config,
        calib_data=None,    # W8A16量化无需校准
        disable_level='L0'  # 自动回退等级，根据精度损失程度增加不量化的层（L0~L5，L0为不回退，精度损失明显时可适当提升等级）
    )

    calibrator.run()  # 执行PTQ量化校准

    calibrator.save(quant_save_path, save_type=["safe_tensor"])

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_model_path', type=str, help="input model and tokenizer path")
    parser.add_argument('--output_model_path', type=str, help="output model and tokenizer path")
    parser.add_argument('--calib_set_path', type=str, default="./calib_set_72b.json", help="calib set path")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    model_path = args.input_model_path
    quant_weight_save_path = args.output_model_path
    calib_set_path = args.calib_set_path
    main(model_path, quant_weight_save_path, calib_set_path)
