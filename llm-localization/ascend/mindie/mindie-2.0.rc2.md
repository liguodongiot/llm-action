




# 量化



```
cd ${ATB_SPEED_HOME_PATH}
python examples/models/llama3/convert_quant_weights.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --w_bit 8 --a_bit 8 --disable_level L0 --device_type cpu --anti_method m1 --act_method 1 --calib_file ${llm_path}/examples/convert/model_slim/boolq.jsonl
```



