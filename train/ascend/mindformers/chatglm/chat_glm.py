import time
import mindspore as ms
import numpy as np
from mindformers.models.glm import GLMConfig, GLMChatModel
from mindformers.models.glm.chatglm_6b_tokenizer import ChatGLMTokenizer
from mindformers.models.glm.glm_processor import process_response

config = GLMConfig(
    position_encoding_2d=True,
    use_past=True,
    is_npu_acceleration=True,
)

def chat_glm():
    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=7)
    model = GLMChatModel(config)
    ms.load_checkpoint("./checkpoint_download/glm/glm_6b.ckpt", model)
    tokenizer = ChatGLMTokenizer('./checkpoint_download/glm/ice_text.model')

    prompts = ["你好", "请介绍一下华为"]
    history = []
    for query in prompts:
        if not history:
            prompt = query
        else:
            prompt = ""
            for i, (old_query, response) in enumerate(history):
                prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
            prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
        inputs = tokenizer(prompt)

        start_time = time.time()
        outputs = model.generate(np.expand_dims(np.array(inputs['input_ids']).astype(np.int32), 0),
                                    max_length=config.max_decode_length, do_sample=False, top_p=0.7, top_k=1)
        end_time = time.time()
        print(f'generate speed: {outputs[0].shape[0]/(end_time-start_time):.2f} tokens/s')
        response = tokenizer.decode(outputs)
        response = process_response(response[0])
        history = history + [(query, response)]
        print(response)

if __name__ == "__main__":
    chat_glm()
  
