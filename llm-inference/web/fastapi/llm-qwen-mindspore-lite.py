

import os
import mindspore_lite
import mindspore as ms

from mindformers.pipeline import pipeline
from mindformers.inference import InferConfig, InferTask
from mindformers.tools.utils import str2bool
from qwen_tokenizer import QwenTokenizer
from dataclasses import dataclass

from typing import Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, status, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn, json, datetime

app = FastAPI()

@dataclass
class InferParam:
    device_id:int = -1
    ge_config_path:str = './lite.ini'
    vocab_file:str = './run/qwen.tiktoken'
    mindir_root_dir:str = 'output'
    seq_length:int = 1024
    batch_size:int = 1
    do_sample:bool = True
    predict_data:str = 'hello'
    predict_length:int = 512
    paged_attention:bool = False
    pa_block_size:int = 16
    pa_num_blocks:int = 512
    dynamic:bool = False 
    top_k:int = 3
    top_p:float = 0.85
    repetition_penalty:float = 1.0
    temperature:float = 1.0
    eos_token_id:int = 151643
    pad_token_id:int = 151643


def get_mindir_path(export_path='output', full=True):
    """Return relative path to MINDIR file"""
    if not os.path.isdir(export_path):
        raise FileNotFoundError(export_path)

    rank_id = os.getenv('RANK_ID', '0')

    mindir_path = "%s/mindir_%s_checkpoint/rank_%s_graph.mindir" % \
                (export_path, "full" if full else "inc", rank_id)
    if not os.path.isfile(mindir_path):
        raise FileNotFoundError(mindir_path)

    var_path = "%s/mindir_%s_checkpoint/rank_%s_variables" % \
                (export_path, "full" if full else "inc", rank_id)
    if not os.path.isdir(var_path):
        raise FileNotFoundError(var_path)

    return mindir_path


def create_mslite_pipeline(args):
    """Create MS lite inference pipeline."""
    ms.set_context(mode=ms.GRAPH_MODE, device_target='Ascend')

    if not os.path.isfile(args.vocab_file):
        raise FileNotFound(args.vocab_file)
    tokenizer = QwenTokenizer(pad_token='<|endoftext|>',
                              vocab_file=args.vocab_file)

    prefill_model_path = get_mindir_path(args.mindir_root_dir, full=True)
    inc_model_path = get_mindir_path(args.mindir_root_dir, full=False)

    if len(args.ge_config_path.split(',')) > 1:
        args.ge_config_path = args.ge_config_path.split(',')
        for ini in args.ge_config_path:
            if not os.path.isfile(ini):
                raise FileNotFoundError(ini)
    else:
        if not os.path.isfile(args.ge_config_path):
            raise FileNotFoundError(args.ge_config_path)

    if args.device_id == -1:
        args.device_id = int(os.getenv('DEVICE_ID', '0'))

    rank_id = int(os.getenv('RANK_ID', '0'))

    print("Creating pipeline from (%s, %s)..." % (prefill_model_path, inc_model_path))
    lite_config = InferConfig(
        prefill_model_path=prefill_model_path,
        increment_model_path=inc_model_path,
        model_type="mindir",
        model_name="qwen",
        ge_config_path=args.ge_config_path,
        device_id=args.device_id,
        rank_id=rank_id,
        dynamic=args.dynamic,
        infer_seq_length=args.seq_length,
        paged_attention=args.paged_attention,
        pa_block_size=args.pa_block_size,
        pa_num_blocks=args.pa_num_blocks,
    )
    pipeline_task = InferTask.get_infer_task("text_generation", lite_config, tokenizer=tokenizer)
    return pipeline_task


def expand_input_list(input_list, batch_size):
    """Expand 'input_list' to a list of size 'batch_size'."""
    if len(input_list) < batch_size:
        repeat_time = batch_size // len(input_list) + 1
        input_list = input_list * repeat_time
    input_list = input_list[:batch_size]
    return input_list


def run_mslite_infer(pipeline_task, prompt, args):
    """Run MS lite inference with PROMPT and ARGS."""

    print("request args:", args)

    input_list = prompt
    if not isinstance(prompt, list):
        input_list = [prompt,]
    input_list = expand_input_list(input_list, args.batch_size)

    max_length = args.predict_length
    if not args.dynamic:
        if args.seq_length is None:
            raise ValueError('Argument "--seq_length" is missing.')
        if max_length > args.seq_length:
            max_length = args.seq_length

    return pipeline_task.infer(
        input_list,
        max_length=max_length,
        do_sample=args.do_sample,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        is_sample_acceleration=False,
        add_special_tokens=False,
        eos_token_id=args.eos_token_id,
        pad_token_id=args.pad_token_id)

args = InferParam(
        mindir_root_dir='/workspace/qwen-7b-chat-mindir',
        ge_config_path='/workspace/mindformers/research/qwen/lite.ini',
        vocab_file='/workspace/Qwen-7B-Chat/qwen.tiktoken',
        seq_length=2048,
        batch_size=1,
        predict_data='hello'
    )


pipeline_task = create_mslite_pipeline(args)
# to warm up the model
run_mslite_infer(pipeline_task, "hello", args)

class Item(BaseModel):
    text_input: str 
    top_k: Optional[int] = Field(default=3, description="top_k")
    top_p: Optional[float] = Field(default=0.85, description="top_p")
    temperature: Optional[float] = Field(temperature=1.0, description="temperature")
    repetition_penalty: Optional[float] = Field(default=1.0, description="repetition_penalty")
    predict_length: Optional[int] = Field(default=512, description="max_length")
    eos_token_id: Optional[int] = Field(default=151643, description="eos_token_id")
    pad_token_id: Optional[int] = Field(default=151643, description="pad_token_id")

@app.post('/v2/models/ensemble/generate_stream')
async def generate_stream(data:Item):
    text_input = data.text_input
    args.top_k = data.top_k
    args.top_p = data.top_p
    args.temperature = data.temperature
    args.repetition_penalty = data.repetition_penalty
    args.predict_length = data.predict_length
    args.eos_token_id = data.eos_token_id
    args.pad_token_id = data.pad_token_id

    outputs = run_mslite_infer(pipeline_task, text_input, args)
    for output in outputs:
        print(output)

    result = {'text_output': outputs[0]}
    
    return JSONResponse(content=result, status_code=status.HTTP_200_OK)

@app.post('/predict')
async def predict(request: Request):
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    input_text = json_post_list.get('input_text')  # input_text

    outputs = run_mslite_infer(pipeline_task, input_text, args)
    for output in outputs:
        print(output)
    return {'result': outputs}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=4)

