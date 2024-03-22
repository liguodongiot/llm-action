

from typing import List


def build_chat_input(messages: List[dict]):
    def _parse_messages(messages, split_role="user"):
        system, rounds = "", []
        round = []
        for i, message in enumerate(messages):
            if message["role"] == "system":
                assert i == 0
                system = message["content"]
                continue
            if message["role"] == split_role and round:
                rounds.append(round)
                round = []
            round.append(message)
        if round:
            rounds.append(round)
        return system, rounds

    system, rounds = _parse_messages(messages, split_role="user")
    print(system, rounds)
    return system, rounds

    
messages = []
messages.append({"role": "system", "content": "你是一个聪明的助手"})
messages.append({"role": "user", "content": "解释一下“温故而知新”"})
messages.append({"role": "assistant", "content": "好的"})
messages.append({"role": "user", "content": "天气不错哇"})
messages.append({"role": "assistant", "content": "是的哈"})
messages.append({"role": "user", "content": "下雨了"})


system, rounds = build_chat_input(messages)

max_history_tokens = 1024

history_tokens = []
for round in rounds[::-1]:
    round_tokens = []
    for message in round:
        if message["role"] == "user":
            round_tokens.append("user: ")
        else:
            round_tokens.append("assistant: ")
        round_tokens.extend([message["content"]])
    if len(history_tokens) == 0 or len(history_tokens) + len(round_tokens) <= max_history_tokens:
            history_tokens = round_tokens + history_tokens  # concat left
            if len(history_tokens) < max_history_tokens:
                continue
    break

input_tokens = [system] + history_tokens

print(input_tokens)
