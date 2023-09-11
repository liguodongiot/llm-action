



# miniGPT

- https://pytorch.org/tutorials/intermediate/ddp_series_minGPT.html

用于训练的文件：

- trainer.py ：包含 Trainer 类，该类使用提供的数据集在模型上运行分布式训练迭代。
- model.py ：定义模型架构。
- char_dataset.py ：包含字符级数据集的 Dataset 类。
- gpt2_train_cfg.yaml ：包含数据、模型、优化器和训练运行的配置。
- main.py ：训练作业的入口点。 它设置 DDP 进程组，读取所有配置并运行训练作业。








