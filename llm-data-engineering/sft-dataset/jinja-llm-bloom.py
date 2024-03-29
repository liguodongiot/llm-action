
from jinja2 import Template
from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment


system = "system"
user = "user"

chat_template=(
"{% if add_prompt %}"
"{{ '<|system|>\n' + prompt + '\n'}}"
"{% endif %}"
"{% for message in messages %}"
"{% if message['role'] == 'system' %}"
"{{'<|user|>\n' + message['content'] + '\n'}}"
"{% elif message['role'] == 'user' %}"
"{{'<|assistant|>\n' + message['content'] + '\n'}}"
"{% endif %}"
"{% endfor %}"
"{% if add_generation_prompt %}"
"{{ '<|assistant|>\n' }}"
"{% endif %}"
)

def raise_exception(message):
    raise TemplateError(message)

jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
jinja_env.globals["raise_exception"] = raise_exception
compiled_template = jinja_env.from_string(chat_template)

print(compiled_template)
print("--------------------")

prompt="You are a helpful assistant."

chat = [
        {
            "role": "system",
            "content": "What is your name?"
        },
        {
            "role": "user",
            "content": "My name is Qwen."
        }
    ]

add_generation_prompt = True
rendered_chat = compiled_template.render(
    messages=chat, system=system, user=user, add_prompt=True, prompt = prompt, add_generation_prompt=add_generation_prompt
)
print(rendered_chat)

print("--------------------train")

add_generation_prompt = False
rendered_chat = compiled_template.render(
    messages=chat, system=system, user=user, add_prompt=True, prompt = prompt, add_generation_prompt=add_generation_prompt
)
print(rendered_chat)