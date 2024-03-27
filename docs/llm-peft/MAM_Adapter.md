






# ----- MAM adapter -----
attn_mode="prefix"
attn_option="concat"
attn_composition="add"
attn_bn=30  # attn bottleneck dim

ffn_mode="adapter"
ffn_option="parallel"
ffn_adapter_layernorm_option="none"
ffn_adapter_init_option="lora"   # 权重的初始化方式
ffn_adapter_scalar="4"
ffn_bn=512 # ffn bottleneck dim



参数详细说明：

https://github.com/jxhe/unify-parameter-efficient-tuning/blob/25b44ac0e6f70e116af15cb866faa9ddc13b6c77/petl/options.py#L45


ffm_mode 逻辑： 

https://github.com/jxhe/unify-parameter-efficient-tuning/blob/3222ce2c0079566a28043e22380eb4ab6ad14389/src/transformers/models/bart/modeling_bart.py#L401


ffn_adapter_init_option 初始化逻辑：

https://github.com/jxhe/unify-parameter-efficient-tuning/blob/3222ce2c0079566a28043e22380eb4ab6ad14389/petl/petl_factory.py#L427



启动脚本：

https://github.com/jxhe/unify-parameter-efficient-tuning/blob/3222ce2c0079566a28043e22380eb4ab6ad14389/exps/run_xsum.sh








