
from jinja2 import Template
from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment


system = "system"
user = "user"

chat_template=(
"{% for message in messages %}"
"{% if message['role'] == 'user' %}"
"{{ '<reserved_102>' + message['content'] }}"
"{% elif message['role'] == 'assistant' %}"
"{{ '<reserved_103>' + message['content'] + '</s>'}}"
"{% endif %}"
"{% endfor %}"
"{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}"
"{{ '<reserved_103>' }}"
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
            "role": "user",
            "content": "What is your name?"
        },
        {
            "role": "assistant",
            "content": "My name is Qwen."
        },
        {
            "role": "user",
            "content": "What is your name?"
        }
    ]

add_generation_prompt = True
rendered_chat = compiled_template.render(
    messages=chat, system=system, user=user, add_generation_prompt=add_generation_prompt
)
print(rendered_chat)

print("--------------------train")

add_generation_prompt = False
rendered_chat = compiled_template.render(
    messages=chat, system=system, user=user, add_generation_prompt=add_generation_prompt
)
print(rendered_chat)