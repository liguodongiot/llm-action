


- https://huggingface.co/THUDM/chatglm2-6b/blob/main/modeling_chatglm.py


-更强大的性能：ChatGLM2-6B 使用了 GLM 的混合目标函数，经过了 1.4T 中英标识符的预训练与人类偏好对齐训练。
- 更长的上下文：基于 FlashAttention 技术，我们将基座模型的上下文长度（Context Length）由 ChatGLM-6B 的 2K 扩展到了 32K，并在对话阶段使用 8K 的上下文长度训练。对于更长的上下文，我们发布了 ChatGLM2-6B-32K 模型。
- 更高效的推理：基于 Multi-Query Attention 技术，ChatGLM2-6B 有更高效的推理速度和更低的显存占用。








## 说明


- F.silu：
- RMSNorm





## chatglm 与 chatglm2 不同支持


- 激活函数不同
- RotaryEmbedding 位置不同。