



```
构建数据集
from megatron.data.gpt_dataset import build_train_valid_test_datasets



预训练函数
from megatron.training import pretrain

Main training program.
This function will run the followings in the order provided:
    1) initialize Megatron.
    2) setup model, optimizer and lr schedule using the model_provider.
    3) call train_val_test_data_provider to get train/val/test datasets.
    4) train the modle using the forward_step_func.






```






