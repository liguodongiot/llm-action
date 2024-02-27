

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
