


- https://gitee.com/ascend/transformers/



```
pip3 install -U transformers
```

```
git clone https://github.com/huggingface/transformers.git
cd examples/pytorch/text-classification
pip install -r requirements.txt
```


```
export TASK_NAME=mrpc

python run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /tmp/$TASK_NAME/

```