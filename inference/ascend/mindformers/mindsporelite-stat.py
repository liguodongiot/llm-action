"""lite infer main."""

import sys
import argparse
from threading import Thread
import time


# Avoid bugs when mslite and mindspore are not built from same commit, which may cause running error.
# pylint: disable=W0611
import mindspore_lite as mslite

from mindformers.models.base_tokenizer import Tokenizer
from mindformers.models import BloomTokenizer, LlamaTokenizer
from mindformers.models import ChatGLMTokenizer, ChatGLM2Tokenizer, GPT2Tokenizer, ChatGLM3Tokenizer
from mindformers.pipeline import pipeline
from mindformers.generation import TextIteratorStreamer
from mindformers.tools.utils import str2bool
from mindformers.inference import InferConfig, InferTask
from research.baichuan2.baichuan2_tokenizer import Baichuan2Tokenizer
from research.internlm.internlm_tokenizer import InternLMTokenizer
from research.qwen.qwen_tokenizer import QwenTokenizer
import json
import numpy as np

#input_path = "/root/workspace/data/alpaca_gpt4_data_input_2k.json"
input_path = "/root/workspace/data/alpaca_10.json"


list_str = json.load(open(input_path, "r"))


def pipeline_from_model_paths(args_, tokenizer):
    """build infer pipeline for model paths."""
    lite_pipeline = pipeline(
        task="text_generation",
        model=(args_.prefill_model_path, args_.increment_model_path),
        tokenizer=tokenizer,
        backend="mslite",
        model_name=args_.model_name,
        ge_config_path=args_.config_path,
        device_id=args_.device_id,
        infer_seq_length=args_.seq_length,
        dynamic=args_.dynamic,
        rank_id=args_.rank_id,
    )
    return lite_pipeline


def pipeline_from_model_name(args_, tokenizer):
    """build infer pipeline for model name."""
    lite_pipeline = pipeline(
        task="text_generation",
        model=args_.model_name,
        tokenizer=tokenizer,
        backend="mslite",
        ge_config_path=args_.config_path,
        device_id=args_.device_id,
        infer_seq_length=args_.seq_length,
        dynamic=args_.dynamic,
        rank_id=args_.rank_id,
    )
    return lite_pipeline


def pipeline_from_model_dir(args_, tokenizer):
    """build infer pipeline for model dir."""
    lite_pipeline = pipeline(
        task="text_generation",
        model=args_.model_dir,
        tokenizer=tokenizer,
        backend="mslite",
        model_name=args_.model_name,
        ge_config_path=args_.config_path,
        device_id=args_.device_id,
        infer_seq_length=args_.seq_length,
        dynamic=args_.dynamic,
        rank_id=args_.rank_id,
    )
    return lite_pipeline


def pipeline_from_infer_config(args_, tokenizer):
    """build infer pipeline for infer config."""
    lite_config = InferConfig(
        prefill_model_path=args_.prefill_model_path,
        increment_model_path=args_.increment_model_path,
        model_type="mindir",
        model_name=args_.model_name,
        ge_config_path=args_.config_path,
        device_id=args_.device_id,
        infer_seq_length=args_.seq_length,
        dynamic=args_.dynamic,
        rank_id=args_.rank_id,
    )
    lite_pipeline = InferTask.get_infer_task("text_generation", lite_config, tokenizer=tokenizer)
    return lite_pipeline


# the model name list that mslite inference has supported.
LITE_SUPPORT_MODELS = {
    'bloom': BloomTokenizer,
    'glm': ChatGLMTokenizer,
    'glm2': ChatGLM2Tokenizer,
    'glm3': ChatGLM3Tokenizer,
    'gpt2': GPT2Tokenizer,
    'codegeex2': ChatGLM2Tokenizer,
    'qwen': QwenTokenizer,
    'llama': LlamaTokenizer,
    'llama2': LlamaTokenizer,
    'codellama': LlamaTokenizer,
    'baichuan2': Baichuan2Tokenizer,
    'internlm': InternLMTokenizer
}


def get_tokenizer(model_name: str, tokenizer_path: str) -> Tokenizer:
    """get tokenizer with model name."""
    tokenizer = None
    lite_support_model = model_name.split('_')[0]
    if lite_support_model in LITE_SUPPORT_MODELS:
        if tokenizer_path is not None:
            tokenizer = LITE_SUPPORT_MODELS[lite_support_model](vocab_file=tokenizer_path)
        else:
            tokenizer = LITE_SUPPORT_MODELS[lite_support_model].from_pretrained(model_name)
    else:
        lite_support_list = tuple(LITE_SUPPORT_MODELS.keys())
        raise ValueError(
            f"model must be in {lite_support_list} when getting tokenizer, but got input {model_name}.")
    return tokenizer


def build_prompt(inputs, model_name, prompt):
    """build prompt for inputs"""
    if model_name.startswith('baichuan2'):
        if not prompt:
            prompt = "<reserved_106>{}<reserved_107>"
        else:
            prompt = "<reserved_106>" + prompt + "<reserved_107>"
    elif model_name.startswith('internlm'):
        if not prompt:
            prompt = "<s><s><|User|>:{}<eoh>\n<|Bot|>:"
        else:
            prompt = "<s><s><|User|>:" + prompt + "<eoh>\n<|Bot|>:"
    if not prompt:
        return inputs
    if prompt.find("{}") != -1:
        return prompt.format(inputs)
    raise ValueError(
        "The prompt is invalid! Please make sure your prompt contains placeholder '{}' to replace user input.")



def inference_stat(first_token_time_list, total_token_time_list, new_token_lens_list):
    print(len(first_token_time_list), len(total_token_time_list), len(new_token_lens_list))


    avg_first_token_time = sum(first_token_time_list) / len(first_token_time_list)

    avg_total_token_time = sum(total_token_time_list) / len(total_token_time_list)

    avg_new_token_lens =  sum(new_token_lens_list) / len(new_token_lens_list)

    avg_token_time_list = []

    for i in range(len(list_str)):
        if (new_token_lens_list[i] <= 1):
            continue
        token_time = (total_token_time_list[i] - first_token_time_list[i]) / (new_token_lens_list[i]-1)
        avg_token_time_list.append(token_time)

    avg_token_time = sum(avg_token_time_list) / len(avg_token_time_list)


    print(" avg_first_token_time: ", avg_first_token_time,
    " avg_token_time: ",avg_token_time,
    " avg_total_token_time: ", avg_total_token_time,
    " avg_new_token_lens: ", avg_new_token_lens)


    print("首Token时延---------------------")
    print("最小值：", round(min(first_token_time_list), 2))
    print("最大值：", round(max(first_token_time_list), 2))
    print("TP50：", np.percentile(np.array(first_token_time_list), 50))
    print("TP90：", np.percentile(np.array(first_token_time_list), 90))
    print("TP99：", np.percentile(np.array(first_token_time_list), 99))


    print("端到端时延---------------------")
    print("最小值：", round(min(total_token_time_list), 2))
    print("最大值：", round(max(total_token_time_list), 2))
    print("TP50：", np.percentile(np.array(total_token_time_list), 50))
    print("TP90：", np.percentile(np.array(total_token_time_list), 90))
    print("TP99：", np.percentile(np.array(total_token_time_list), 99))


    print("生成Token长度---------------------")
    print("最小值：", round(min(new_token_lens_list), 2))
    print("最大值：", round(max(new_token_lens_list), 2))
    print("TP50：", np.percentile(np.array(new_token_lens_list), 50))
    print("TP90：", np.percentile(np.array(new_token_lens_list), 90))
    print("TP99：", np.percentile(np.array(new_token_lens_list), 99))


def infer_main(args_):
    """lite infer main."""
    tokenizer = get_tokenizer(args_.model_name.lower(), args_.tokenizer_path)
    lite_pipeline = pipeline_from_infer_config(
        args_, tokenizer
    )

    user_input = "可以帮我做一份旅游攻略吗？"
    start_time = time.perf_counter()
    output, gen_time, new_token_lens = lite_pipeline.infer(user_input,
                                     do_sample=args_.do_sample,
                                     top_k=args_.top_k,
                                     top_p=args_.top_p,
                                     repetition_penalty=args_.repetition_penalty,
                                     temperature=args_.temperature,
                                     max_length=args_.max_length,
                                     max_new_tokens=5,
                                     is_sample_acceleration=args_.is_sample_acceleration,
                                     add_special_tokens=args_.add_special_tokens)
    end_time = time.perf_counter()
    first_gen_time = end_time - start_time
    print("第一次生成时间：", first_gen_time)
    print(output)


    first_token_time_list = []
    total_token_time_list = []
    new_token_lens_list = []


    for i, line in enumerate(list_str):
        start_time = time.perf_counter()
        output, gen_time, new_token_lens = lite_pipeline.infer(line,
                                    do_sample=args_.do_sample,
                                    top_k=args_.top_k,
                                    top_p=args_.top_p,
                                    repetition_penalty=args_.repetition_penalty,
                                    temperature=args_.temperature,
                                    max_length=args_.max_length,
                                    max_new_tokens=1,
                                    is_sample_acceleration=args_.is_sample_acceleration,
                                    add_special_tokens=args_.add_special_tokens)
        end_time = time.perf_counter()
        infer_time = end_time - start_time
        print("infer time: ", infer_time)
        print(output)
        first_token_time_list.append(gen_time)
        print("\n-------------------\n")



    for i, line in enumerate(list_str):
        # start_time = time.perf_counter()
        output, gen_time, new_token_lens = lite_pipeline.infer(line,
                                    do_sample=args_.do_sample,
                                    top_k=args_.top_k,
                                    top_p=args_.top_p,
                                    repetition_penalty=args_.repetition_penalty,
                                    temperature=args_.temperature,
                                    max_length=args_.max_length,
                                    max_new_tokens=args_.max_output_length,
                                    is_sample_acceleration=args_.is_sample_acceleration,
                                    add_special_tokens=args_.add_special_tokens)
        # end_time = time.perf_counter()
        print(output)
        # infer_time = end_time - start_time
        # print("infer time: ", infer_time)
        total_token_time_list.append(gen_time)
        new_token_lens_list.append(new_token_lens)
        print("\n-------------------\n")

    inference_stat(first_token_time_list, total_token_time_list, new_token_lens_list)
    


def infer_stream_main(args_):
    """main entry for infer stream."""
    tokenizer = get_tokenizer(args_.model_name.lower(), args_.tokenizer_path)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
    lite_pipeline = pipeline_from_model_paths(
        args_, tokenizer
    )

    
    text = "可以帮我做一份旅游攻略吗？"
    user_input = build_prompt(text, args_.model_name.lower(), args_.prompt)
    generation_kwargs = dict(inputs=user_input,
                                streamer=streamer,
                                is_sample_acceleration=args_.is_sample_acceleration,
                                add_special_tokens=args_.add_special_tokens)
    thread = Thread(target=lite_pipeline, kwargs=generation_kwargs)
    thread.start()
    output = ""
    for new_text in streamer:
        print(new_text)
        output += new_text
    print(output)

    first_token_time_list = []
    total_token_time_list = []
    new_token_lens_list = []

    for i, line in enumerate(list_str):

        user_input = build_prompt(line, args_.model_name.lower(), args_.prompt)
        generation_kwargs = dict(inputs=user_input,
                                 streamer=streamer,
                                 is_sample_acceleration=args_.is_sample_acceleration,
                                 add_special_tokens=args_.add_special_tokens)
        thread = Thread(target=lite_pipeline, kwargs=generation_kwargs)
        thread.start()
        output = ""

        token_num = 0
        start_time = time.perf_counter()
        for new_text in streamer:
            if token_num == 0:
                first_time = time.perf_counter()
                first_token_time = first_time - start_time
                first_token_time_list.append(first_token_time)
            print(new_text)
            output += new_text
            token_num += 1
        end_time = time.perf_counter()
        gen_time = end_time - start_time
        total_token_time_list.append(gen_time)
        new_token_lens_list.append(token_num)
        print(output)
        print("\n-------------------\n")

    inference_stat(first_token_time_list, total_token_time_list, new_token_lens_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--device_id', default=0, type=int,
        help='ID of the target device, the value must be in [0, device_num_per_host-1],'
             'while device_num_per_host should be no more than 4096. Default: None')
    parser.add_argument(
        '--rank_id', default=0, type=int,
        help='ID of the target device, the value must be in [0, device_num_per_host-1],'
             'while device_num_per_host should be no more than 4096. Default: None')
    parser.add_argument(
        '--model_dir', default=None, type=str,
        help="This model dir path."
             "Default: None")
    parser.add_argument(
        '--model_name', default="common", type=str,
        help=f"The model name, only supports name in {LITE_SUPPORT_MODELS}."
             "Default: None")
    parser.add_argument(
        '--seq_length', default=2048, type=int,
        help="This model dir path."
             "Default: None")
    parser.add_argument(
        '--tokenizer_path', default=None, type=str,
        help="Tokenizer model to load."
             "Default: None")
    parser.add_argument(
        '--prefill_model_path', default=None, type=str,
        help="This full model path."
             "Default: None")
    parser.add_argument(
        '--increment_model_path', default=None, type=str,
        help="When use kv-cache, this is cache mode path."
             "Default: None")
    parser.add_argument(
        '--config_path', default=None, type=str,
        help="ge config file path."
             "Default: None")
    parser.add_argument(
        '--do_sample', default=False, type=str2bool,
        help="Whether postprocess in graph or not."
             "Default: False")
    parser.add_argument(
        '--top_k', default=1, type=int,
        help="top k."
             "Default: 1")
    parser.add_argument(
        '--top_p', default=1.0, type=float,
        help="top p."
             "Default: 1.0")
    parser.add_argument(
        '--repetition_penalty', default=1.0, type=float,
        help="repetition penalty."
             "Default: 1.0")
    parser.add_argument(
        '--temperature', default=1.0, type=float,
        help="The value used to modulate the next token probabilities."
             "Default: 1.0")
    parser.add_argument(
        '--max_length', default=512, type=int,
        help="The maximum word length that can be generated."
             "Default: 512")
    parser.add_argument(
        '--max_output_length', default=128, type=int,
        help="The maximum output length that can be generated."
             "Default: 128")
    parser.add_argument(
        '--is_sample_acceleration', default=False, type=str2bool,
        help="Whether postprocess in graph or not."
             "Default: False")
    parser.add_argument(
        '--add_special_tokens', default=False, type=str2bool,
        help="Whether preprocess add special tokens or not."
             "Default: False")
    parser.add_argument(
        '--stream', default=False, type=str2bool,
        help="Whether decode in stream or not."
             "Default: False")
    parser.add_argument(
        '--prompt', default=None, type=str,
        help="The content of prompt."
             "Default: None")
    parser.add_argument(
        '--dynamic', default=False, type=str2bool,
        help="Whether use dynamic inference."
             "Default: False")

    args = parser.parse_args()
    if len(args.config_path.split(',')) > 1:
        args.config_path = args.config_path.split(',')
    if args.stream:
        infer_stream_main(args)
    else:
        infer_main(args)



