

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



https://download.pytorch.org/whl/cu118/torch-2.0.1%2Bcu118-cp310-cp310-linux_x86_64.whl#sha256=a7a49d459bf4862f64f7bc1a68beccf8881c2fa9f3e0569608e16ba6f85ebf7b

https://download.pytorch.org/whl/cu118/torchvision-0.15.2%2Bcu118-cp310-cp310-linux_x86_64.whl#sha256=19ca4ab5d6179bbe53cff79df1a855ee6533c2861ddc7389f68349d8b9f8302a



## lora


降低显存消耗：

gradient_checkpointing

gradient_accumulation_steps

batch_size

seq_length










