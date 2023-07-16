## DeepSpeed Configuration JSON

地址：https://www.deepspeed.ai/docs/config-json/



### FP16 训练的 ZeRO 优化

启用和配置 ZeRO 内存优化



- stage3_gather_16bit_weights_on_model_save: [boolean]

> 在通过 save_16bit_model() 保存模型之前合并权重。 由于权重在 GPU 之间进行分区，因此它们不是 state_dict 的一部分，因此启用此选项时该函数会自动收集权重，然后保存 fp16 模型权重。



