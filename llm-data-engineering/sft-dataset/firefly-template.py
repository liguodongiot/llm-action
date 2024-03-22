from dataclasses import dataclass
from typing import Dict


@dataclass
class Template:
    template_name:str
    system_format: str
    user_format: str
    assistant_format: str
    system: str
    stop_word: str
    # stop_token_id: int


template_dict: Dict[str, Template] = dict()


def register_template(template_name, system_format, user_format, assistant_format, system, stop_word=None):
    template_dict[template_name] = Template(
        template_name=template_name,
        system_format=system_format,
        user_format=user_format,
        assistant_format=assistant_format,
        system=system,
        stop_word=stop_word,
        # stop_token_id=stop_token_id
    )


# 注册template
register_template(
    template_name='default',
    system_format='System: {content}\n\n',
    user_format='User: {content}\nAssistant: ',
    assistant_format='{content} {stop_token}',
    system=None,
    stop_word=None
)

register_template(
    template_name='internlm',
    system_format="<|System|>:{content}\n",
    user_format='<|User|>:{content}\n<|Bot|>:',
    assistant_format='{content}</s>\n',
    system="You are an AI assistant whose name is InternLM (书生·浦语).\n"
        "- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n"
        "- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.",
    stop_word='</s>'
)

register_template(
    template_name='internlm2',
    system_format='<|im_start|>system\n{content}<|im_end|>\n',
    user_format='<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n',
    assistant_format='{content}<|im_end|>\n',
    system="You are an AI assistant whose name is InternLM (书生·浦语).\n"
        "- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n"
        "- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.",
    stop_word='<|im_end|>'
)

register_template(
    template_name='qwen',
    system_format='<|im_start|>system\n{content}<|im_end|>\n',
    user_format='<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n',
    assistant_format='{content}<|im_end|>\n',
    system="You are a helpful assistant.",
    stop_word='<|im_end|>'
)

register_template(
    template_name='yi',
    system_format='<|im_start|>system\n{content}<|im_end|>\n',
    user_format='<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n',
    assistant_format='{content}<|im_end|>\n',
    system=None,
    stop_word='<|im_end|>'
)

register_template(
    template_name="orion",
    system_format='<s>',
    user_format='Human: {content}\n\nAssistant: </s>',
    assistant_format='{content}</s>',
    system='',
    stop_word='</s>',
)

register_template(
    template_name='deepseek',
    system_format=None,
    user_format='User: {content}\n\nAssistant: ',
    assistant_format='{content}<｜end▁of▁sentence｜>',
    system=None,
    stop_word='<｜end▁of▁sentence｜>'
)

# todo 更优雅的实现方式
register_template(
    template_name='chatglm2',
    system_format=None,
    user_format='[Round {idx}]\n\n问：{content}\n\n答：',
    assistant_format='{content}',
    system=None,
    stop_word='</s>',
)

register_template(
    template_name='chatglm3',
    system_format='{content}',
    user_format='{content}',
    assistant_format='{content}',
    system="You are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown.",
    stop_word='</s>',
)

register_template(
    template_name='ziya2',
    system_format=None,
    user_format='<human>:{content} <bot>:',
    assistant_format='{content}</s>',
    system=None,
    stop_word='</s>',
)

register_template(
    template_name="xverse",
    system_format=None,
    user_format='Human: {content}\n\nAssistant: ',
    assistant_format='{content}<|endoftext|>',
    system=None,
    stop_word='<|endoftext|>',
)

register_template(
    template_name='minicpm',
    system_format=None,
    user_format='<用户>{content}<AI>',
    assistant_format='{content}</s>',
    system=None,
    stop_word='</s>'
)

register_template(
    template_name='zephyr',
    system_format='<|system|>\n{content}</s>',
    user_format='<|user|>\n{content}</s>\n<|assistant|>\n',
    assistant_format='{content}</s>\n',
    system=None,
    stop_word='</s>'
)

register_template(
    template_name='mistral',
    system_format='<s>',
    user_format='[INST]{content}[/INST]',
    assistant_format='{content}</s>',
    system='',
    stop_word='</s>'
)

register_template(
    template_name='mixtral',
    system_format='<s>',
    user_format='[INST]{content}[/INST]',
    assistant_format='{content}</s>',
    system='',
    stop_word='</s>'
)

register_template(
    template_name='baichuan',
    system_format=None,
    user_format='<reserved_102>{content}<reserved_103>',
    assistant_format='{content}</s>',
    system=None,
    stop_word='</s>'
)

register_template(
    template_name='baichuan2',
    system_format=None,
    user_format='<reserved_106>{content}<reserved_107>',
    assistant_format='{content}</s>',
    system=None,
    stop_word='</s>'
)

register_template(
    template_name='vicuna',
    system_format='{content}\n',
    user_format='USER: {content} ASSISTANT:',
    assistant_format='{content}</s>',
    system="A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    stop_word='</s>'
)

register_template(
    template_name='llama2',
    system_format='<<SYS>>\n{content}\n<</SYS>>\n\n',
    user_format='[INST]{content}[/INST]',
    assistant_format='{content} </s>',
    system="You are a helpful, respectful and honest assistant. "
        "Always answer as helpfully as possible, while being safe. "
        "Your answers should not include any harmful, unethical, "
        "racist, sexist, toxic, dangerous, or illegal content. "
        "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        "If a question does not make any sense, or is not factually coherent, "
        "explain why instead of answering something not correct. "
        "If you don't know the answer to a question, please don't share false information.",
    stop_word='</s>'
)

register_template(
    template_name='gemma',
    system_format='<bos>',
    user_format='<start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n',
    assistant_format='{content}<eos>\n',
    system='',
    stop_word='<eos>'
)

# {system_format}{user_format}{assistant_format}{user_format}{assistant_format}




if __name__ == '__main__':
    print("-------------------")
    print(template_dict['default'])
    
    print("-------------------")
    print(template_dict['qwen'])

    print("-------------------")
    print(template_dict['baichuan'])
    
    print("-------------------")
    print(template_dict['baichuan2'])
