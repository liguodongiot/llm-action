


# 代码

## swe-bench


```
modelscope download --dataset evalscope/cmmlu --local_dir ./data/cmmlu
modelscope download --dataset evalscope/ceval --local_dir ./data/ceval



modelscope download --dataset allenai/ai2_arc --local_dir ./data/arc
modelscope download --dataset evalscope/ceval --local_dir ./data/ceval





modelscope download --dataset princeton-nlp/SWE-bench_Lite --local_dir ./data/swe_bench_lite
modelscope download --dataset princeton-nlp/SWE-bench_Verified --local_dir ./data/swe_bench_verified
modelscope download --dataset evalscope/swe-bench-verified-mini --local_dir ./data/swe_bench_verified_mini

```






```
modelscope download --dataset princeton-nlp/SWE-bench_bm25_13K --local_dir ./data/swe_bench_bm25_13k
modelscope download --dataset princeton-nlp/SWE-bench_bm25_27K --local_dir ./data/swe_bench_bm25_27k
modelscope download --dataset princeton-nlp/SWE-bench_bm25_40K --local_dir ./data/swe_bench_bm25_40k
modelscope download --dataset princeton-nlp/SWE-bench_oracle --local_dir ./data/swe_bench_oracle
```


```
docker pull swebench/sweb.eval.x86_64.django_1776_django-10914:latest
docker pull swebench/sweb.eval.x86_64.django_1776_django-11001:latest

docker pull swebench/sweb.eval.x86_64.django_1776_django-10924:latest
docker pull swebench/sweb.eval.x86_64.django_1776_django-11019:latest


docker pull swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest

docker pull swebench/sweb.eval.x86_64.astropy_1776_astropy-14182:latest
docker pull swebench/sweb.eval.x86_64.astropy_1776_astropy-14365:latest
docker pull swebench/sweb.eval.x86_64.astropy_1776_astropy-14995:latest
docker pull swebench/sweb.eval.x86_64.astropy_1776_astropy-7746:latest
docker pull swebench/sweb.eval.x86_64.astropy_1776_astropy-6938:latest

```





```
docker save -o swe-11001.tar swebench/sweb.eval.x86_64.django_1776_django-11001:latest

docker save -o swe-11019.tar swebench/sweb.eval.x86_64.django_1776_django-11019:latest


docker save -o swe-14182.tar swebench/sweb.eval.x86_64.astropy_1776_astropy-14182:latest


docker save -o swe-14365.tar swebench/sweb.eval.x86_64.astropy_1776_astropy-14365:latest
docker save -o swe-14995.tar swebench/sweb.eval.x86_64.astropy_1776_astropy-14995:latest



docker save -o swe-7746.tar swebench/sweb.eval.x86_64.astropy_1776_astropy-7746:latest
docker save -o swe-6938.tar swebench/sweb.eval.x86_64.astropy_1776_astropy-6938:latest


docker save -o swe-12907.tar swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest

docker save -o swe-10924.tar swebench/sweb.eval.x86_64.django_1776_django-10924:latest

```


## mbpp/humaneval

```
modelscope download --dataset google-research-datasets/mbpp --local_dir ./data/mbpp
modelscope download --dataset opencompass/humaneval --local_dir ./data/humaneval
modelscope download --dataset AI-ModelScope/code_generation_lite --local_dir ./data/code_generation_lite
modelscope download --dataset evalscope/MultiPL-E --local_dir ./data/multiple_humaneval_and_mbpp


docker save -o sandbox-fusion.tar volcengine/sandbox-fusion:server-20250609

docker pull volcengine/sandbox-fusion:server-20250609

	
docker pull volcengine/sandbox-fusion:server-20250609


```


# 推理

## hle



```
modelscope download --dataset cais/hle --local_dir ./data/hle

```


```
modelscope download --dataset evalscope/aime26 --local_dir ./data/aime26

```

```
modelscope download --dataset allenai/ai2_arc --local_dir ./data/arc


modelscope download --dataset AI-ModelScope/gsm8k --local_dir ./data/gsm8k
```


