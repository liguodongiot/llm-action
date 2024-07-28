


## 摩尔线程


2022年，摩尔线程就推出了GPU统一系统架构MUSA，发布并量产“苏堤”和“春晓”两颗全功能GPU芯片，这也是国内采用现代GPU架构





## 主流 AI 芯片配置



| 厂商  | 型号                          | 图形处理器        | 架构           | 显存           | FP16 算力                               | BF16 算力                      | INT8 算力                   | FP32算力       | TF32 算力                     | FP8算力                        | CUDA Core | Tensor Core |
| --- | --------------------------- | ------------ | ------------ | ------------ | ------------------------------------- | ---------------------------- | ------------------------- | ------------ | --------------------------- | ---------------------------- | --------- | ----------- |
| 英伟达 | RTX 3090                    | GA102-300-A1 | Ampere       | 24GB（GDDR6X） | 35.58 TFLOPS                          | -                           | -                        | 35.58 TFLOPS | -                          | 不支持                          | 10496     | 328         |
| 英伟达 | RTX 3090 Ti                 | GA102-350-A1 | Ampere       | 24GB（GDDR6X） | 40.00 TFLOPS                          | -                           | -                        | 40.00 TFLOPS | -                          | 不支持                          | 10752     | 336         |
| 英伟达 | RTX 4090                    | AD102-300-A1 | Ada Lovelace | 24GB（GDDR6X） | 369.7 TFLOPS（Tensor Core）82.58 TFLOPS | 369.7 TFLOPS（Tensor Core）    | 739.4 TFLOPS（Tensor Core） | 82.58 TFLOPS | -                          | -                           | 16384     | 512         |
| 英伟达 | RTX 4090 Ti                 | AD102-400-A1 | Ada Lovelace | 24GB（GDDR6X） | 93.24 TFLOPS                          | -                           | -                        | 93.24 TFLOPS | -                          | -                           | 18176     | 568         |
| 英伟达 | RTX 4090D-特供-消费级            | AD102-250-A1 | Ada Lovelace | 24GB（GDDR6X） | 329.3 TFLOPS（Tensor Core）73.54 TFLOPS | 329.3 TFLOPS（Tensor Core）    | 658.6 TFLOPS（Tensor Core） | 73.54 TFLOPS | -                          | -                           | 14592     | 456         |
| 英伟达 | L20（PCIe）-特供-推理（PCIe）       | AD102        | Ada Lovelace | 48GB（GDDR6）  | 119.5 TFLOPS（Tensor Core）             | 119.5 TFLOPS（Tensor Core）    | 239 TOPS（Tensor Core）     | 59.8 TFLOPS  | 59.8 TFLOPS（Tensor Core）    | 239 TFOPS（Tensor Core）       | 11776     | 368         |
| 英伟达 | H20-特供-训练（PCIe、Nvlink）      | -           | Hopper       | 96GB（HBM3）   | 148 TFLOPS（Tensor Core）               | 148 TFLOPS（Tensor Core）      | 296 TOPS（Tensor Core）     | 44 TFLOPS    | 74 TFLOPS（Tensor Core）      | 296 TFOPS（Tensor Core）       | -        | -          |
| 英伟达 | A800（PCIe）                  | GA100        | Ampere       | 80GB（HBM2e）  | 312 TFLOPS（Tensor Core）77.97 TFLOPS   | 312 TFLOPS（Tensor Core）      | 624 TOPS（Tensor Core）     | 19.5 TFLOPS  | 156 TFLOPS（Tensor Core）     | 不支持                          | 6912      | 432         |
| 英伟达 | H800（ SXM）                  | GH100        | Hopper       | 80GB（HBM3）   | 1,979 TFLOPS（Tensor Core）             | 1,979 teraFLOPS（Tensor Core） | 3,958 TOPS（Tensor Core）   | 67 teraFLOPS | 989 teraFLOPS （Tensor Core） | 3,958 teraFLOPS（Tensor Core） | 18,432    | 640         |
| 昇腾  | Atlas 800T A2训练（910B3-HCCS） | -           | 达芬奇          | 64GB（HBM2e）  | 313 TFLOPS                            | 313 TFLOPS                   | 640 TOPS                  | 75 TFLOPS    | 141 TFLOPS（HF）              | 不支持                          | -        | -          |
| 昇腾  | Atlas 800I 推理（910B4）        | -           | 达芬奇          | 32GB（HBM2e）  | 280 TFLOPS                            | 280 TFLOPS                   | 550 TOPS                  | 75 TFLOPS    | 141 TFLOPS（HF）              | 不支持                          | -        | - |

