

```
sh /task/script/bootstrap-llm.sh --sft_type=full --conda_env=torch1131-venv \
--train_dataset_path=s3://infer-test/data/train_data_20240105_1k.json,s3://infer-test/data/train_data_20240105_100.json  \
--pre_model_path=s3://infer-test/model-bloom-2b6 \
--checkpoint_path=s3://infer-test/model-bloom-2b6 \
--model_output_path=s3://infer-test/model-bloom-2b6 \
--model_metrics_path=s3://infer-test/model-bloom-metrics/processx.json \
--gpu_num=2 \
--epoch=1 --batch_size=8 --learning_rate=1e-5 --max_seq_length=512 --logging_steps=1 --warmup_ratio=0.1 --weight_decay=0
```







## lora


降低显存消耗：

gradient_checkpointing

gradient_accumulation_steps

batch_size

seq_length







## FAQ


### baichuan2报错

- 'BitsAndBytesConfig' object is not subscriptable

https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat/discussions/2



- AttributeError: 'BaichuanTokenizer' object has no attribute 'sp_model'

降低版本到4.34.0及以下 ： pip install transformers==4.34.0






