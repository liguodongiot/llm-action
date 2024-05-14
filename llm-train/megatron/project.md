

## 项目结构


```
.
├── CONTRIBUTING.md
├── LICENSE
├── README.md
├── docs  # 文档
├── examples  # 训练样例
├── images
├── megatron  # Megatron 核心
├── pretrain_bert.py
├── pretrain_gpt.py
├── pretrain_gpt_core.py
├── pretrain_ict.py
├── pretrain_retro.py
├── pretrain_t5.py
├── pretrain_vision_classify.py
├── pretrain_vision_dino.py
├── pretrain_vision_inpaint.py
├── setup.py 
├── tasks  # 任务示例
├── tests  # 单元测试
└── tools  # 工具，合并权重、合并数据集等
```


### megatron

```
megatron
├── __init__.py
├── arguments.py
├── checkpointing.py
├── core  # Megatron Core
│   ├── README.md
│   ├── __init__.py
│   ├── enums.py
│   ├── fusions
│   │   ├── __init__.py
│   │   ├── fused_bias_dropout.py
│   │   ├── fused_bias_gelu.py
│   │   ├── fused_layer_norm.py
│   │   └── fused_softmax.py
│   ├── model_parallel_config.py  # Megatron Core 基础配置
│   ├── models
│   │   ├── __init__.py
│   │   ├── common
│   │   │   └── rotary_pos_embedding.py
│   │   └── gpt
│   │       ├── __init__.py
│   │       ├── gpt_embedding.py
│   │       └── gpt_model.py
│   ├── package_info.py
│   ├── parallel_state.py
│   ├── pipeline_parallel  # 流水线并行核心模块
│   │   ├── __init__.py
│   │   ├── p2p_communication.py
│   │   └── schedules.py
│   ├── requirements.txt
│   ├── tensor_parallel  # 张量并行核心模块
│   │   ├── __init__.py
│   │   ├── cross_entropy.py
│   │   ├── data.py
│   │   ├── layers.py
│   │   ├── mappings.py
│   │   ├── random.py
│   │   └── utils.py
│   ├── transformer
│   │   ├── __init__.py
│   │   ├── attention.py
│   │   ├── core_attention.py
│   │   ├── custom_layers
│   │   │   └── transformer_engine.py
│   │   ├── enums.py
│   │   ├── mlp.py
│   │   ├── module.py
│   │   ├── transformer_block.py
│   │   ├── transformer_config.py
│   │   ├── transformer_layer.py
│   │   └── utils.py
│   └── utils.py
├── data
│   ├── Makefile
│   ├── __init__.py
│   ├── autoaugment.py
│   ├── bert_dataset.py
│   ├── biencoder_dataset_utils.py
│   ├── blendable_dataset.py
│   ├── data_samplers.py
│   ├── dataset_utils.py
│   ├── gpt_dataset.py
│   ├── helpers.cpp
│   ├── ict_dataset.py
│   ├── image_folder.py
│   ├── indexed_dataset.py
│   ├── multimodal_dataset.py
│   ├── orqa_wiki_dataset.py
│   ├── realm_dataset_utils.py
│   ├── realm_index.py
│   ├── t5_dataset.py
│   ├── test
│   │   ├── test_indexed_dataset.py
│   │   └── test_preprocess_data.sh
│   └── vit_dataset.py
├── dist_signal_handler.py
├── fp16_deprecated
│   └── loss_scaler.py
├── fused_kernels
│   ├── __init__.py
│   ├── compat.h
│   ├── tests
│   │   ├── __init__.py
│   │   └── test_fused_kernels.py
│   └── type_shim.h
├── global_vars.py
├── indexer.py
├── initialize.py
├── memory.py
├── microbatches.py
├── model
│   ├── __init__.py
│   ├── bert_model.py
│   ├── biencoder_model.py
│   ├── classification.py
│   ├── distributed.py
│   ├── enums.py
│   ├── fused_bias_gelu.py
│   ├── fused_layer_norm.py
│   ├── fused_softmax.py
│   ├── gpt_model.py
│   ├── language_model.py
│   ├── module.py
│   ├── multiple_choice.py
│   ├── realm_model.py
│   ├── t5_model.py
│   ├── transformer.py
│   ├── utils.py
│   └── vision
│       ├── classification.py
│       ├── dino.py
│       ├── esvit_swin_backbone.py
│       ├── inpainting.py
│       ├── knn_monitor.py
│       ├── mit_backbone.py
│       ├── swin_backbone.py
│       ├── utils.py
│       └── vit_backbone.py
├── mpu
│   └── tests
│       ├── __init__.py
│       ├── commons.py
│       ├── test_cross_entropy.py
│       ├── test_data.py
│       ├── test_initialize.py
│       ├── test_layers.py
│       └── test_random.py
├── optimizer
│   ├── __init__.py
│   ├── clip_grads.py
│   ├── distrib_optimizer.py
│   ├── grad_scaler.py
│   └── optimizer.py
├── optimizer_param_scheduler.py
├── static
│   └── index.html
├── text_generation
│   ├── __init__.py
│   ├── api.py
│   ├── beam_utils.py
│   ├── communication.py
│   ├── forward_step.py
│   ├── generation.py
│   ├── sampling.py
│   └── tokenization.py
├── text_generation_server.py
├── timers.py
├── tokenizer
│   ├── __init__.py
│   ├── bert_tokenization.py
│   ├── gpt2_tokenization.py
│   └── tokenizer.py
├── training.py
└── utils.py
```


### tools


```
tools
├── bert_embedding
│   ├── __init__.py
│   ├── dataset.py
│   ├── embed.py
│   ├── external_libs.py
│   ├── huggingface.py
│   └── utils.py
├── checkpoint_loader_megatron.py
├── checkpoint_saver_megatron.py
├── checkpoint_util.py
├── linter.py
├── merge_datasets.py
├── openwebtext
│   ├── README.md
│   ├── add_id.py
│   ├── blacklist_urls.py
│   ├── cleanup_dataset.py
│   ├── cleanup_fix_dataset.py
│   ├── filter_ngrams.py
│   ├── find_duplicates.py
│   ├── group_duplicate_url.py
│   ├── merge_jsons.py
│   └── remove_group_duplicates.py
├── preprocess_data.py
├── preprocess_data_nmt.py
├── preprocess_mmdata.py
├── retro
│   ├── README.md
│   ├── cli
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   └── cli.py
│   ├── db
│   │   ├── __init__.py
│   │   ├── build.py
│   │   ├── dataset.py
│   │   └── utils.py
│   ├── examples
│   │   ├── preprocess_data.sh
│   │   └── pretrain_model.sh
│   ├── external_libs.py
│   ├── index
│   │   ├── __init__.py
│   │   ├── build.py
│   │   ├── factory.py
│   │   ├── index.py
│   │   ├── indexes
│   │   │   ├── __init__.py
│   │   │   ├── faiss_base.py
│   │   │   └── faiss_par_add.py
│   │   └── utils.py
│   ├── main.py
│   ├── query
│   │   ├── __init__.py
│   │   ├── chunk_dataset.py
│   │   ├── query.py
│   │   ├── retro_dataset.py
│   │   └── utils.py
│   └── utils.py
├── run_text_generation_server.py
└── text_generation_cli.py
```


### tasks


```
tasks
├── data_utils.py
├── ensemble_classifier.py
├── eval_utils.py
├── finetune_utils.py
├── glue
│   ├── data.py
│   ├── finetune.py
│   ├── mnli.py
│   └── qqp.py
├── main.py
├── msdp
│   ├── README.md
│   ├── evaluate.py
│   ├── main.py
│   ├── metrics.py
│   ├── preprocessing.py
│   └── prompt.py
├── orqa
│   ├── README.md
│   ├── evaluate_orqa.py
│   ├── evaluate_utils.py
│   ├── supervised
│   │   ├── data.py
│   │   ├── eval_utils.py
│   │   └── finetune.py
│   └── unsupervised
│       ├── nq.py
│       ├── qa_utils.py
│       └── tokenizers.py
├── race
│   ├── data.py
│   └── finetune.py
├── vision
│   ├── classification
│   │   ├── classification.py
│   │   └── eval_utils.py
│   ├── finetune_utils.py
│   ├── main.py
│   └── segmentation
│       ├── cityscapes.py
│       ├── data.py
│       ├── finetune_segformer.py
│       ├── finetune_setr.py
│       ├── metrics.py
│       ├── seg_heads.py
│       ├── seg_models.py
│       ├── transforms.py
│       └── utils.py
└── zeroshot_gpt
    ├── datasets.py
    ├── detokenizer.py
    └── evaluate.py
```





### examples



```
examples
├── detxoify_lm
│   ├── README.md
│   ├── annotations
│   │   ├── filter-selfgeneration.py
│   │   ├── perspective_api_annotate.py
│   │   └── preprocess.sh
│   ├── finetune_gpt.py
│   ├── finetune_gpt_distributed-1.3b.sh
│   ├── generate-1.3b.sh
│   ├── generate_samples_gpt.py
│   ├── perspective_api.py
│   └── self_generation
│       └── selfgenerate-1.3b-unconditional.sh
├── evaluate_retriever_nq.sh
├── evaluate_zeroshot_gpt.sh
├── finetune_mnli_distributed.sh
├── finetune_race_distributed.sh
├── finetune_retriever_distributed.sh
├── merge_mp_bert.sh
├── msdp
│   ├── README.md
│   ├── data_processing.sh
│   ├── eval_knwl_generation.sh
│   ├── eval_resp_generation.sh
│   ├── prep_resp_gen.sh
│   ├── prompt_knwl_gen.sh
│   └── prompt_resp_gen.sh
├── pretrain_bert.sh
├── pretrain_bert_distributed.sh
├── pretrain_bert_distributed_with_mp.sh
├── pretrain_gpt.sh
├── pretrain_gpt3_175B.sh
├── pretrain_gpt_distributed.sh
├── pretrain_gpt_distributed_with_mp.sh
├── pretrain_ict.sh
├── pretrain_t5.sh
├── pretrain_t5_distributed.sh
├── pretrain_t5_distributed_with_mp.sh
├── run_text_generation_server_345M.sh
├── run_text_generation_server_345M_8_tensor_parallel.sh
└── sc21
    ├── CONFIG.sh
    ├── README.md
    ├── SBATCH.sh
    ├── SRUN.sh
    ├── run_figure_11.sh
    ├── run_figure_12.sh
    ├── run_figure_13.sh
    ├── run_figure_14.sh
    ├── run_figure_15.sh
    ├── run_figure_16.sh
    ├── run_figure_17.sh
    ├── run_figure_18.sh
    └── run_table_1.sh
```



### tests-单元测试

```
tests
├── __init__.py
├── functional_tests
│   ├── __init__.py
│   ├── python_test_utils
│   │   ├── __init__.py
│   │   ├── check_slurm_job_completion.py
│   │   ├── get_test_results_from_tensorboard_logs.py
│   │   ├── test_ci_pipeline.py
│   │   └── test_resume_checkpoint_pipeline.py
│   ├── shell_test_utils
│   │   └── jobwait.sh
│   ├── test_results
│   │   ├── bert
│   │   │   ├── bert_tp1_pp2_1nodes_50steps.json
│   │   │   ├── bert_tp1_pp4_1nodes_50steps.json
│   │   │   ├── bert_tp2_pp2_1nodes_50steps.json
│   │   │   └── bert_tp4_pp1_1nodes_50steps.json
│   │   └── gpt3
│   │       ├── gpt3_tp1_pp2_1nodes_50steps.json
│   │       ├── gpt3_tp1_pp4_1nodes_50steps.json
│   │       ├── gpt3_tp2_pp2_1nodes_50steps.json
│   │       └── gpt3_tp4_pp1_1nodes_50steps.json
│   └── test_scripts
│       ├── bert
│       │   ├── pretrain_bert_distributed_resume_checkpoint_test.sh
│       │   ├── pretrain_bert_distributed_test.sh
│       │   ├── sbatch_bert_distributed_resume_checkpoint_test.sh
│       │   └── sbatch_bert_distributed_test.sh
│       └── gpt3
│           ├── pretrain_gpt3_distributed_resume_checkpoint_test.sh
│           ├── pretrain_gpt3_distributed_test.sh
│           ├── sbatch_gpt3_distributed_resume_checkpoint_test.sh
│           └── sbatch_gpt3_distributed_test.sh
├── models
│   ├── __init__.py
│   ├── test_gpt_embedding.py
│   └── test_gpt_model.py
├── pipeline_parallel
│   ├── __init__.py
│   └── test_schedules.py
├── tensor_parallel
│   └── __int__.py
├── transformer
│   ├── __init__.py
│   ├── test_core_attention.py
│   ├── test_module.py
│   ├── test_parallel_attention.py
│   ├── test_parallel_mlp.py
│   ├── test_parallel_transformer_block.py
│   ├── test_parallel_transformer_layer.py
│   └── test_transformer_config.py
└── unit_tests
    ├── __init__.py
    ├── conftest.py
    ├── tensor_parallel
    │   ├── test_cross_entropy.py
    │   ├── test_data.py
    │   ├── test_mappings.py
    │   ├── test_random.py
    │   └── test_tensor_parallel_utils.py
    ├── test_basic.py
    ├── test_parallel_state.py
    ├── test_utilities.py
    └── test_utils.py
```



