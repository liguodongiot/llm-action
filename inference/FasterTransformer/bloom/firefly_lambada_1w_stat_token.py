from __future__ import annotations

import argparse
import configparser
import dataclasses
import json
import pathlib
import time
from typing import Dict, List
from random import choice
import torch
import tqdm
import transformers

from utils import bloom
from statistics import mean
import numpy as np
import os




class TensorEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        return super().default(obj)


class LambadaDataset(torch.utils.data.Dataset):
    """ LAMBADA dataset class. """

    def __init__(self,
                 path: str | pathlib.Path,
                 tokenizer: transformers.PreTrainedTokenizerBase):
        self.tokenizer = tokenizer
        with open(path, 'r') as f:
            inputs, targets = zip(*[
                json.loads(line)["text"].strip('\n').rsplit(' ', 1)
                for line in f.readlines()])
            # This whitespace preprocessing (additional space to the target)
            # is required.
            targets = [' ' + tgt for tgt in targets]
            self.encodings = self.tokenizer(list(inputs),
                                            targets,
                                            padding=True,
                                            return_token_type_ids=True,
                                            return_tensors='pt')

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return dict(
            input_ids=self.encodings['input_ids'][idx],
            attention_mask=self.encodings['attention_mask'][idx],
            token_type_ids=self.encodings['token_type_ids'][idx]
        )


@dataclasses.dataclass
class Metric:
    acc: float


@dataclasses.dataclass
class RequestAndResult:
    prompt: str
    model_answer: str
    target: str
    input_ids: List[int]
    input_len: int
    output_len: int
    model_params: bloom.BloomParam
    infer_params: bloom.BloomInferParam
    output_ids: List[int]
    metrics: Metric

    def asdict(self):
        return dataclasses.asdict(self)


class Timer:

    def __init__(self):
        self._start_times = {}
        self._total_elapsed_times = {}

    def start(self, tag='__default'):
        self._start_times[tag] = time.time()

    def stop(self, tag='__default'):
        elapsed_time = time.time() - self._start_times[tag]
        if tag not in self._total_elapsed_times:
            self._total_elapsed_times[tag] = 0
        self._total_elapsed_times[tag] += elapsed_time
        return elapsed_time

    def elapsed_time_in_sec(self, tag='__default'):
        if tag not in self._total_elapsed_times:
            return None
        return self._total_elapsed_times[tag]

    def reset(self):
        self._start_times.clear()
        self._total_elapsed_times.clear()


def get_args():
    parser = argparse.ArgumentParser(
        'Evaluation: LAMBADA Task',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    bloom.BloomParam.add_args_group(parser)
    bloom.BloomInferParam.add_args_group(parser)

    group = parser.add_argument_group('LAMBADA Task Parameters')
    group.add_argument(
        '--checkpoint-path', type=str, metavar='DIR', default=None,
        help='A directory of a converted pretrained checkpoint and model config '
             'If None, a model will inference by random weights.')
    group.add_argument(
        '--dataset-path', type=str, metavar='PATH', required=True,
        help="A file path to LAMBADA task dataset.")
    group.add_argument(
        '--output-path', type=str, metavar='PATH', default=None,
        help="Path to sample output file.")
    group.add_argument(
        "--tokenizer-path", type=str, metavar='DIR_OR_PATH', default=None,
        help='A file path of a pretrained tokenizer or a checkpoint directory '
             'of HF pretrained model.')
    group.add_argument(
        '--lib-path', type=str, metavar='PATH', default='./lib/libth_transformer.so',
        help='A FT library path to load `FasterTransformer.ParallelGptOp`')
    group.add_argument(
        '--test-hf', action='store_true',
        help='Run a huggingface model instead of an FT model. The checkpoint '
             'of the huggingface model is assumed to be at --tokenizer-path.')
    group.add_argument(
        '--acc-threshold', type=float, metavar='M', default=None,
        help='The minimum value of the expected accuracy of the LAMBADA '
             'evaluation for a test. If the achieved accuracy is less '
             'than given value, a value error will occurs.')
    group.add_argument(
        '--show-progress', action='store_true',
        help='Show evaluation progress')
    group.add_argument(
        '--inference-data-type', '--data-type', type=str, metavar='TYPE', default=None,
        choices=[None, 'fp32', 'fp16', 'bf16'],
        help='The data type to inference. If None, the data type follows the '
             'checkpoint data type.')
    group.add_argument(
        '--weights-data-type', type=str, metavar='TYPE', default=None,
        choices=[None, 'fp32', 'fp16'],
        help='The data type of FT checkpoint. If None, it will be retrieved '
             'from the config file in the checkpoint directory.')
    group.add_argument(
        '--int8_mode', type=int, default=0, choices=[0, 1],
        help='The level of quantization to perform.'
             ' 0: No quantization. All computation in data_type'
             ' 1: Quantize weights to int8, all compute occurs in fp16/bf16. Not supported when data_type is fp32')
    group.add_argument('--input-token-len', type=int, default=128)
    group.add_argument('--output-token-len', type=int, default=128)
    group.add_argument('--dianxiao-path-stat', type=str, default="/workspace/data/random_sample_1w_stat.json")

    args = parser.parse_args()

    print('\n=================== Arguments ===================')
    for k, v in vars(args).items():
        print(f' - {k.ljust(25, ".")}: {v}')
    print('=================================================')

    return args


def get_model_and_tokenizer(args: argparse.Namespace):
    tokenizer_path = pathlib.Path(args.tokenizer_path)
    # HF requires left padding for a decoder-only model.
    padding_side = 'left' if args.test_hf else 'right'
    if tokenizer_path.is_dir():
        # Load from the HF's pretrained model directory.
        tokenizer = transformers.BloomTokenizerFast.from_pretrained(
            args.tokenizer_path, padding_side=padding_side)
    else:
        # Directly load from a tokenizer json file.
        tokenizer = transformers.BloomTokenizerFast(
            tokenizer_file=tokenizer_path, padding_side=padding_side)
    # For open-ended generation, the pad token is sometimes replaced by the
    # eos token but the Bloom of HF requires as it is to correctly generate.

    if args.test_hf:
        # Load HF's pretrained model for testing.
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.tokenizer_path, torch_dtype=torch.float16).cuda()
        return model, tokenizer

    checkpoint_path = pathlib.Path(args.checkpoint_path)
    config_path = checkpoint_path / 'config.ini'

    if config_path.exists():
        # Read model params from config.
        cfg = configparser.ConfigParser()
        cfg.read(config_path)
        model_name = 'gpt'
        inference_data_type = args.inference_data_type
        if inference_data_type == None:
            inference_data_type = cfg.get(model_name, "weight_data_type")
        model_args = dict(
            head_num=cfg.getint(model_name, 'head_num'),
            size_per_head=cfg.getint(model_name, "size_per_head"),
            layer_num=cfg.getint(model_name, "num_layer"),
            tensor_para_size=cfg.getint(model_name, "tensor_para_size"),
            vocab_size=cfg.getint(model_name, "vocab_size"),
            start_id=cfg.getint(model_name, "start_id"),
            end_id=cfg.getint(model_name, "end_id"),
            weights_data_type=cfg.get(model_name, "weight_data_type"),
            layernorm_eps=cfg.getfloat(model_name, 'layernorm_eps'),
            inference_data_type=inference_data_type)
    else:
        inference_data_type = args.inference_data_type
        if inference_data_type == None:
            inference_data_type = args.weights_data_type
        model_args = dict(head_num=args.num_heads,
                          size_per_head=args.size_per_head,
                          vocab_size=args.vocab_size,
                          start_id=args.start_id or tokenizer.bos_token_id,
                          end_id=args.end_id or tokenizer.eos_token_id,
                          layer_num=args.num_layers,
                          tensor_para_size=args.tensor_para_size,
                          weights_data_type=args.weights_data_type,
                          inference_data_type=inference_data_type)

    # update common parameters
    model_args.update(dict(
        lib_path=args.lib_path,
        pipeline_para_size=args.pipeline_para_size,
        shared_contexts_ratio=args.shared_contexts_ratio,
        int8_mode=args.int8_mode
    ))

    print('[FT][INFO] Load BLOOM model')
    for k, v in model_args.items():
        print(f' - {k.ljust(25, ".")}: {v}')

    # Check sanity and consistency between the model and tokenizer.
    checklist = ['head_num', 'size_per_head', 'vocab_size', 'layer_num',
                 'tensor_para_size', 'tensor_para_size', 'weights_data_type']
    if None in [model_args[k] for k in checklist]:
        none_params = [p for p in checklist if model_args[p] is None]
        print(f'[FT][WARNING] Found None parameters {none_params}. They must '
              f'be provided either by config file or CLI arguments.')
    if model_args['start_id'] != tokenizer.bos_token_id:
        print('[FT][WARNING] Given start_id is not matched with the bos token '
              'id of the pretrained tokenizer.')
    if model_args['end_id'] not in (tokenizer.pad_token_id, tokenizer.eos_token_id):
        print('[FT][WARNING] Given end_id is not matched with neither pad '
              'token id nor eos token id of the pretrained tokenizer.')
    model = bloom.Bloom(**model_args)
    if not model.load(ckpt_path=args.checkpoint_path):
        print('[FT][WARNING] Skip model loading since no checkpoints are found')

    return model, tokenizer


def split_inputs_and_targets(entries: Dict[str, torch.LongTensor],
                             pad_token_id: int,
                             pad_to_left=False):
    input_ids = entries['input_ids']
    attn_mask = entries['attention_mask']
    token_type_ids = entries['token_type_ids']

    # Split inputs and labels by token_type_ids.
    input_token_ids = [
        ids[(mask == 1) & (type_ids == 0)]
        for ids, mask, type_ids in zip(input_ids, attn_mask, token_type_ids)]
    # FT allows int32 tensors.
    input_lengths = torch.tensor(
        [len(input_tokens) for input_tokens in input_token_ids]).int()
    max_length = input_lengths.max()
    input_token_ids = torch.stack([
        torch.nn.functional.pad(
            token_ids,
            pad=[max_length - len(token_ids), 0]
            if pad_to_left else [0, max_length - len(token_ids)],
            mode='constant',
            value=pad_token_id
        ) for token_ids in input_token_ids]).int()
    target_token_ids = [
        ids[(mask == 1) & (type_ids == 1)]
        for ids, mask, type_ids in zip(input_ids, attn_mask, token_type_ids)]
    return input_token_ids, input_lengths, target_token_ids



time_list = []
pre_toekn_time_list = []
gen_token_len_list = []
error_input_list = []

# dianxiao_path = "/workspace/data/random_sample_1w_format.json"
dianxiao_path = "/workspace/data/actor_v3_21w_sampling_1w.json"

# dianxiao_path_stat = "/workspace/data/random_sample_1w_stat.json"

stat_result = []

@torch.no_grad()
def main():

    args = get_args()
    input_token_len = args.input_token_len
    output_token_len = args.output_token_len
    dianxiao_path_stat = args.dianxiao_path_stat

    print("------------------------------")
    print("bello-bloom-7b fp16")
    print("input_token_len:", input_token_len, "output_token_len:", output_token_len,"dianxiao_path_stat:",dianxiao_path_stat)
    print("------------------------------")

    nums = 0

    model, tokenizer = get_model_and_tokenizer(args)
    model.eval()

    with open(dianxiao_path, encoding='utf-8') as json_str:
        result = json.load(json_str)

    for temp in result:

        input_str = temp["input"]


        input_encoder = tokenizer(input_str, padding=True, return_token_type_ids=True, return_tensors='pt')

        input_encoder_dict = dict(
            input_ids=input_encoder['input_ids'],
            attention_mask=input_encoder['attention_mask'],
            token_type_ids=input_encoder['token_type_ids']
        )

        input_token_ids, input_lengths, target_token_ids = split_inputs_and_targets(input_encoder_dict,
                                                                                    tokenizer.pad_token_id, args.test_hf)
        output_length = args.output_token_len

        params = bloom.BloomInferParam.from_args(args, 1)

        input_token_lens = input_encoder.input_ids.size(dim=1)
        # input_token_lens2 = input_token_ids.size(dim=1)
        # print("input_token_lens:", input_token_lens, "input_token_lens2:", input_token_lens2)

        if args.test_hf:
            # Outputs (batch_size, seq_length)
            start = time.perf_counter()
            outputs = model.generate(inputs=input_token_ids.cuda(),
                                     max_new_tokens=output_length,
                                     num_beams=args.beam_width,
                                     temperature=args.temperature,
                                     top_k=args.top_k,
                                     top_p=args.top_p,
                                     repetition_penalty=args.repetition_penalty,
                                     length_penalty=args.len_penalty)

                                     # ,use_cache=False)
            end = time.perf_counter()
            # output_token_ids: input/padding/output
            output_token_ids = outputs[:, input_token_ids.shape[1]:]
            output_token_ids = [
                out[:len(tgt)].cpu()
                for out, tgt in zip(output_token_ids, target_token_ids)]
        else:
            param_dict = params.asdict()
            start = time.perf_counter()
            outputs = model(start_ids=input_token_ids,
                            start_lengths=input_lengths,
                            output_len=output_length,
                            **param_dict)
            end = time.perf_counter()

            outputs = outputs[0]
            # if params.return_cum_log_probs or params.return_cum_log_probs > 0:
            #     outputs = outputs[0]  # output_token_ids.

        rets = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        runTime = end - start
        runTime_ms = runTime * 1000
        print("id", str(temp["id"]), "运行时间：", round(runTime_ms, 2), "毫秒")
        time_list.append(round(runTime_ms, 2))

        print("id", str(temp["id"]), "input:", input_str)
        print("id", str(temp["id"]), "output_text:",  rets[0])
        print("------------------")
        input_seq_len = len(input_str.replace("</s>", ""))
        output_seq_len = len(rets[0])
        gen_seq_len = output_seq_len-input_seq_len
        generate_text = rets[0][input_seq_len:]
        print("id", str(temp["id"]), "generate_text:", generate_text)
        gen_token_len = tokenizer(generate_text, return_tensors="pt").input_ids.size(dim=1)
        output_token_lens = tokenizer(rets[0], return_tensors="pt").input_ids.size(dim=1)
        # gen_token_len = output_token_lens - input_token_lens

        if gen_token_len > 0:
            gen_token_len_list.append(gen_token_len)
            pre_toekn_time_list.append(round(runTime_ms / gen_token_len, 2))
        else:
            error_input_list.append(temp["id"])
            continue


        temp["input_seq_len"] = input_seq_len
        temp["input_token_len"] = input_token_lens
        temp["output_seq_len"] = output_seq_len
        temp["output_token_len"] = output_token_lens
        temp["gen_seq_len"] = gen_seq_len
        temp["gen_token_len"] = gen_token_len
        temp["inference_duration_us"] = round(runTime_ms, 2)
        temp["generate_text"] = generate_text
        stat_result.append(temp)

        nums = nums + 1
        #if nums == 1000:
        #   break


    print("错误输入列表：", error_input_list)

    print("推理耗时列表：", time_list)
    result = mean(time_list)
    print("均值：", round(result, 2))
    print("最小值：", round(min(time_list), 2))
    print("最大值：", round(max(time_list), 2))
    print("TP50：", np.percentile(np.array(time_list), 50))
    print("TP90：", np.percentile(np.array(time_list), 90))
    print("TP99：", np.percentile(np.array(time_list), 99))

    print("gen token len:", gen_token_len_list)
    result_gen_token_len = mean(gen_token_len_list)
    print("pre token time:", pre_toekn_time_list)
    result_pre_token_time = mean(pre_toekn_time_list)
    print("每个Token生成的平均耗时：", round(result_pre_token_time, 2))
    print("生成Token长度均值：", round(result_gen_token_len, 2))
    print("每个Token生成的平均耗时(微平均)：", round(sum(time_list)/sum(gen_token_len_list), 2))
    print("TOKEN 最小值：", min(gen_token_len_list))
    print("TOKEN 最大值：", max(gen_token_len_list))


    pids = os.getpid()
    # print("pid:", pids, "rank:", torch.distributed.get_rank(group=None))

    with open(dianxiao_path_stat+"_"+str(pids), 'w', encoding='utf-8') as b:
        json.dump(stat_result, b, ensure_ascii=False, indent=4)

    print("------------------------------")
    print("bello-bloom-7b fp16")
    print("input_token_len:", input_token_len, "output_token_len:", output_token_len,"dianxiao_path_stat:",dianxiao_path_stat)
    print("------------------------------")


if __name__ == "__main__":
    main()

