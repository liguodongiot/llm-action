


- https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file
- https://github.com/ggerganov/llama.cpp





GGUF量化格式

- ctransformers、llama.cpp



```
CMAKE_ARGS="-DGGML_METAL=on" pip install -U llama-cpp-python --no-cache-dir
pip install 'llama-cpp-python[server]'
```


```
export MODEL=/Users/liguodong/model/qwen2/qwen2-0_5b-instruct-q2_k.gguf
python3 -m llama_cpp.server --model $MODEL  --n_gpu_layers 1
```




## 量化

- https://github.com/ggerganov/llama.cpp/pull/1684  值得阅读
- 模型量化技术概述及 GGUF/GGML 文件格式解析，了解 Q4_0、Q4_1、Q4_K 和 Q4_K_M 的区别：https://blog.csdn.net/weixin_42426841/article/details/142706753
- LLM量化大比拼：哪种 LLM 量化最适合你？：https://zhuanlan.zhihu.com/p/8936080946
- llama.cpp中的量化方法简介：https://zhuanlan.zhihu.com/p/12729759086



- https://github.com/ggerganov/llama.cpp/blob/3e693197724c31d53a9b69018c2f1bd0b93ebab2/examples/quantize/quantize.cpp#L18

Q4_K_M, Q5_K_S and Q5_K_M are considered "recommended".



量化方法：
```
static const std::vector<struct quant_option> QUANT_OPTIONS = {
    { "Q4_0",     LLAMA_FTYPE_MOSTLY_Q4_0,     " 4.34G, +0.4685 ppl @ Llama-3-8B",  },
    { "Q4_1",     LLAMA_FTYPE_MOSTLY_Q4_1,     " 4.78G, +0.4511 ppl @ Llama-3-8B",  },
    { "Q5_0",     LLAMA_FTYPE_MOSTLY_Q5_0,     " 5.21G, +0.1316 ppl @ Llama-3-8B",  },
    { "Q5_1",     LLAMA_FTYPE_MOSTLY_Q5_1,     " 5.65G, +0.1062 ppl @ Llama-3-8B",  },
    { "IQ2_XXS",  LLAMA_FTYPE_MOSTLY_IQ2_XXS,  " 2.06 bpw quantization",            },
    { "IQ2_XS",   LLAMA_FTYPE_MOSTLY_IQ2_XS,   " 2.31 bpw quantization",            },
    { "IQ2_S",    LLAMA_FTYPE_MOSTLY_IQ2_S,    " 2.5  bpw quantization",            },
    { "IQ2_M",    LLAMA_FTYPE_MOSTLY_IQ2_M,    " 2.7  bpw quantization",            },
    { "IQ1_S",    LLAMA_FTYPE_MOSTLY_IQ1_S,    " 1.56 bpw quantization",            },
    { "IQ1_M",    LLAMA_FTYPE_MOSTLY_IQ1_M,    " 1.75 bpw quantization",            },
    { "TQ1_0",    LLAMA_FTYPE_MOSTLY_TQ1_0,    " 1.69 bpw ternarization",           },
    { "TQ2_0",    LLAMA_FTYPE_MOSTLY_TQ2_0,    " 2.06 bpw ternarization",           },
    { "Q2_K",     LLAMA_FTYPE_MOSTLY_Q2_K,     " 2.96G, +3.5199 ppl @ Llama-3-8B",  },
    { "Q2_K_S",   LLAMA_FTYPE_MOSTLY_Q2_K_S,   " 2.96G, +3.1836 ppl @ Llama-3-8B",  },
    { "IQ3_XXS",  LLAMA_FTYPE_MOSTLY_IQ3_XXS,  " 3.06 bpw quantization",            },
    { "IQ3_S",    LLAMA_FTYPE_MOSTLY_IQ3_S,    " 3.44 bpw quantization",            },
    { "IQ3_M",    LLAMA_FTYPE_MOSTLY_IQ3_M,    " 3.66 bpw quantization mix",        },
    { "Q3_K",     LLAMA_FTYPE_MOSTLY_Q3_K_M,   "alias for Q3_K_M"                   },
    { "IQ3_XS",   LLAMA_FTYPE_MOSTLY_IQ3_XS,   " 3.3 bpw quantization",             },
    { "Q3_K_S",   LLAMA_FTYPE_MOSTLY_Q3_K_S,   " 3.41G, +1.6321 ppl @ Llama-3-8B",  },
    { "Q3_K_M",   LLAMA_FTYPE_MOSTLY_Q3_K_M,   " 3.74G, +0.6569 ppl @ Llama-3-8B",  },
    { "Q3_K_L",   LLAMA_FTYPE_MOSTLY_Q3_K_L,   " 4.03G, +0.5562 ppl @ Llama-3-8B",  },
    { "IQ4_NL",   LLAMA_FTYPE_MOSTLY_IQ4_NL,   " 4.50 bpw non-linear quantization", },
    { "IQ4_XS",   LLAMA_FTYPE_MOSTLY_IQ4_XS,   " 4.25 bpw non-linear quantization", },
    { "Q4_K",     LLAMA_FTYPE_MOSTLY_Q4_K_M,   "alias for Q4_K_M",                  },
    { "Q4_K_S",   LLAMA_FTYPE_MOSTLY_Q4_K_S,   " 4.37G, +0.2689 ppl @ Llama-3-8B",  },
    { "Q4_K_M",   LLAMA_FTYPE_MOSTLY_Q4_K_M,   " 4.58G, +0.1754 ppl @ Llama-3-8B",  },
    { "Q5_K",     LLAMA_FTYPE_MOSTLY_Q5_K_M,   "alias for Q5_K_M",                  },
    { "Q5_K_S",   LLAMA_FTYPE_MOSTLY_Q5_K_S,   " 5.21G, +0.1049 ppl @ Llama-3-8B",  },
    { "Q5_K_M",   LLAMA_FTYPE_MOSTLY_Q5_K_M,   " 5.33G, +0.0569 ppl @ Llama-3-8B",  },
    { "Q6_K",     LLAMA_FTYPE_MOSTLY_Q6_K,     " 6.14G, +0.0217 ppl @ Llama-3-8B",  },
    { "Q8_0",     LLAMA_FTYPE_MOSTLY_Q8_0,     " 7.96G, +0.0026 ppl @ Llama-3-8B",  },
    { "F16",      LLAMA_FTYPE_MOSTLY_F16,      "14.00G, +0.0020 ppl @ Mistral-7B",  },
    { "BF16",     LLAMA_FTYPE_MOSTLY_BF16,     "14.00G, -0.0050 ppl @ Mistral-7B",  },
    { "F32",      LLAMA_FTYPE_ALL_F32,         "26.00G              @ 7B",          },
    // Note: Ensure COPY comes after F32 to avoid ftype 0 from matching.
    { "COPY",     LLAMA_FTYPE_ALL_F32,         "only copy tensors, no quantizing",  },
};
```









