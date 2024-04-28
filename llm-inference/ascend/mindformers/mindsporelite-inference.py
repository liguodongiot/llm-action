
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
from datetime import datetime

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


def infer_main(args_):
    """lite infer main."""
    tokenizer = get_tokenizer(args_.model_name.lower(), args_.tokenizer_path)
    lite_pipeline = pipeline_from_infer_config(
        args_, tokenizer
    )

    while True:
        user_input = input("Please enter your predict data: \n")
        if user_input == "exit":
            print("Task is over.")
            sys.exit()
        user_input = build_prompt(user_input, args_.model_name.lower(), args_.prompt)

        start_time = time.perf_counter()
        output, gen_time, new_token_lens = lite_pipeline.infer(user_input,
                                     do_sample=args_.do_sample,
                                     top_k=args_.top_k,
                                     top_p=args_.top_p,
                                     repetition_penalty=args_.repetition_penalty,
                                     temperature=args_.temperature,
                                     max_length=args_.max_length,
                                     max_new_tokens=args_.max_output_length,
                                     is_sample_acceleration=args_.is_sample_acceleration,
                                     add_special_tokens=args_.add_special_tokens)
        end_time = time.perf_counter()
        infer_time = end_time - start_time
        print("infer time: ", infer_time)
        print(output)


def infer_stream_main(args_):
    """main entry for infer stream."""
    tokenizer = get_tokenizer(args_.model_name.lower(), args_.tokenizer_path)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
    lite_pipeline = pipeline_from_model_paths(
        args_, tokenizer
    )

    while True:
        user_input = input("Please enter your predict data(input 'exit' to quit): \n")
        if user_input == "exit":
            print("Quit now, this may take a while.")
            sys.exit()
        user_input = build_prompt(user_input, args_.model_name.lower(), args_.prompt)
        generation_kwargs = dict(inputs=user_input,
                                 streamer=streamer,
                                 is_sample_acceleration=args_.is_sample_acceleration,
                                 add_special_tokens=args_.add_special_tokens)
        thread = Thread(target=lite_pipeline, kwargs=generation_kwargs)
        thread.start()
        output = ""
        start_time = time.perf_counter()
        for new_text in streamer:
            current_time = time.perf_counter()
            current_fmt_time = datetime.now()
            current_token_time = current_time - start_time
            start_time = current_time
            print(current_fmt_time, current_token_time,  new_text)
            output += new_text
        print(output)


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


