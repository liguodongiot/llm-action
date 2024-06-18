






global_batch_size = batch_size * data_parallel * micro_batch_num * micro_batch_interleave_num = 16 = 2 * 1 * 8 * 1).



batch_size : 数据批次大小

micro_batch_num：流水线并行的微批次大小。pipeline_satge大于1时，开启流水并行时使用，此处需满足micro_batch_num >= pipeline_satge


micro_batch_interleave_num： batch_size的拆分份数，多副本并行开关，通常在模型并行时使用，用于优化model_parallel时产生的通信损耗，纯流水并行时不建议使用。


# compute throughput   (samples/s/p)  每一步每一卡每一秒能处理的样本数
throughput = self.global_batch_size / self.device_num / (per_step_seconds / 1000) 





deepspeed:

global_train_batch_size =  train_micro_batch_size_per_gpu * gradient_accumulation_steps * number of GPUs