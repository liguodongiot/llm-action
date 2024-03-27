

# chatglm-6b

- https://huggingface.co/THUDM/chatglm-6b/tree/main
- https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/main.py


自回归填空





ChatGLM借助编码器-解码器架构思想，前半部分采用类似于Bert的双向注意力进行掩码，后半部分采用类似于GPT的自回归架构进行预测。







说明：

- gelu
- LayerNorm



- 重新排列了LN和残差连接的顺序，具体来讲就是将Post-LN改成Pre-LN。
- 使用一个线性层来预测输出词；
- 将ReLU激活函数替换为GeLU激活函数。