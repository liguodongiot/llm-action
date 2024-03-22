
from jinja2 import Template
from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment


system = "system"
user = "user"
assistant = "assistant"

chat_template=(
"{% for message in messages %}"
"{% if message['role'] == 'system' %}"
"{{'<|im_start|>' + system + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
"{% elif message['role'] == 'user' %}"
"{{'<|im_start|>' + user + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
"{% elif message['role'] == 'assistant' %}"
"{{'<|im_start|>' + assistant + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
"{% endif %}"
"{% endfor %}"
"{% if add_generation_prompt %}"
"{{ '<|im_start|>assistant\n' }}"
"{% endif %}"
)




def raise_exception(message):
    raise TemplateError(message)

jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
jinja_env.globals["raise_exception"] = raise_exception
compiled_template = jinja_env.from_string(chat_template)

print(compiled_template)
print("--------------------")


chat = [
        {
            'role': 'system', 
            'content': 'You are a helpful assistant.'
        },
        {
            "role": "user",
            "content": "What is your name?"
        },
        {
            "role": "assistant",
            "content": "My name is Qwen."
        }
    ]

add_generation_prompt = True
rendered_chat = compiled_template.render(
    messages=chat, system=system, user=user, assistant=assistant, add_generation_prompt=add_generation_prompt
)

print(rendered_chat)

