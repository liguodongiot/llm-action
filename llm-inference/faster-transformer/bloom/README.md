






## 统计性能指标

### 数据格式

```
[
    {
        "id":0,
        "input":"#姓名：何#性别：女士#剩余额度：一万七千#当前额度：一万七千#绑定银行：无#活动：D#初始额度：nan#手机尾号：七八八#注册时间：二零二三年一月三日#借款日期：nan#优惠券数：0.0#电销时间：二零二三年四月二十三日#登录类型：无#提额时间：二零二三年四月二十二日#offer类型：提额#cust_type：授信T91-180\n\nHuman: 你好。\nAssistant: ",
        "answer":"Assistant: 好，请问一下是何女士对吧？<\/s>"
    },
    {
        "id":1,
        "input":"#姓名：岳#性别：先生#剩余额度：五万#当前额度：零#绑定银行：农业银行#活动：E#初始额度：一万#手机尾号：六四七三#注册时间：二零一九年十一月七日#借款日期：nan#优惠券数：1.0#电销时间：二零二三年二月二十五日#登录类型：无#提额时间：nan#offer类型：nan#cust_type：授信T31-90\n\nHuman: 喂。\nAssistant: 您好，请问就是岳先生对吧？<\/s>Human: 行。对呀。\nAssistant: 诶你好，岳先生，不好意思打扰到你，我这边是xxx的回访专员哈，来电话给你做个提醒，关注到你这边在我们ttt里面申请的这个额度还打算用吗？诶已经过了六天了，ttt。<\/s>Human: 嗯。我用，但是我我用是可以用，但是我不我我上次问他，他不都不跟我讲，我这个钱，假如我这个一万块钱，我用一天大概要多少钱利息知道吧？\nAssistant: 啊就是我这边给你说一下哈，可能之前之前可能是有同事就是就是没有给你说，对吧？<\/s>Human: 对呀，说我不敢用呀，你。\nAssistant: 啊跟你说一下。<\/s>Human: 嗯。\nAssistant: 一万块钱一万块钱，你要分一年，他总共的利息下来是在一千三左右哈，你平均到每个月的话，你下来的一百块钱左右，每天每个月在一百块钱左右的利息，平均到每天就在在三块钱左右哈。<\/s>Human: 就是五万块钱。\nAssistant: 我一万块钱哈。<\/s>Human: 呃一一天三块钱左右应该就是一百块钱，但是的话说就是。\nAssistant: 一百块钱左右嘛。<\/s>Human: 一你我一百块钱就是一就是假的话，就是利大概是六千多块钱。\nAssistant: 我看一下你。<\/s>Human: 一年。\nAssistant: 分的话差不多六千块钱左右哈。<\/s>Human: 对对对，因为对我所以说你你给我讲一百块钱，我说大概六千多块，我就大概六千块钱左右了，一年。\nAssistant: 对。<\/s>Human: 啊啊\nAssistant: 嗯嗯差这一五万块钱一年就是六千块钱左右哈。<\/s>Human: 哦对对对，因为因为因为我这我这这这个钱还有个问题，就是假的话的。就是我我说的我拿的是我我的就是分十二期还，但是我提前还就是提前还就没有违约，没有什么利息的啦。\nAssistant: 嗯。<\/s>Human: 是\nAssistant: 诶你提前还款的话，它就是就是你你就是我们就是满足条件没有逾期的情况下就可以申请提前还款了，提前还款，您用几个月收取你几个月的利息哈，就是满足条件还满三期之后就可以申请提前还款了哈，就比如说哈，因为先生比如说你分了一年啊，就你只用了半年哈，你提前还款了，你嗯。<\/s>Human: 嗯。对。对对对。对。\nAssistant: ",
        "answer":"Assistant: 就是后你提前还款了之后，后面是不会收取这利息的哈。<\/s>"
    },
    {
        "id":2,
        "input":"#姓名：李#性别：先生#剩余额度：两万元#当前额度：零#绑定银行：无#活动：C#初始额度：nan#手机尾号：五零零七#注册时间：nan#借款日期：nan#优惠券数：nan#电销时间：nan#登录类型：nan#提额时间：nan#offer类型：nan#cust_type：nan\n\nAssistant: 好，请问是李先生是吧？<\/s>Human: 啊行。\nAssistant: 诶李先生您好，我是xxx的工作人员。<\/s>Human: 哦\nAssistant: 来电呢是给您确认一下，您之前在ttt还有个剩余额度，两万元是您本人在自己使用的，对吧？<\/s>Human: 啊？\nAssistant: 诶是这样的先生，因为呢我们系统显示您之前还款记录保持得比较好，现在公司在回馈优质客户专门来电给您确认一下，<\/s>Human: 好\nAssistant: 如果您打算以后长期使用ttt，还需要提高您的额度和降低利息的话。<\/s>Human: 嗯\nAssistant: ",
        "answer":"Assistant: 您现在呢只需要将您的剩余额度两万元按去暂时借出来。<\/s>"
    }
]
```


### 推理


推理耗时，平均每个token生成时长等。

单卡：
```
CUDA_VISIBLE_DEVICES=1 python examples/pytorch/gpt/firefly_lambada_dianxiao_1w_stat_token.py \
--checkpoint-path /workspace/model/firefly-2b6-dx-1tp/belle7b/1/1-gpu \
--tokenizer-path /workspace/model/firefly-2b6-dx \
--dataset-path /workspace/data/lambada_test.jsonl \
--lib-path  /workspace/lib/libth_transformer.so \
--inference-data-type fp16 --show-progress --input-token-len 64 --output-token-len 256 \
--dianxiao-path-stat /workspace/output/firefly_random_sample_1w_256_stat_ft.json
```

双卡张量并行：
```
CUDA_VISIBLE_DEVICES=2,3  mpirun -n 2 python examples/pytorch/gpt/firefly_lambada_dianxiao_1w_stat_token.py \
--checkpoint-path /workspace/model/firefly-2b6-dx-2tp/belle7b/1/2-gpu \
--tokenizer-path /workspace/model/firefly-2b6-dx \
--dataset-path /workspace/data/lambada_test.jsonl \
--lib-path  /workspace/lib/libth_transformer.so \
--inference-data-type fp16 \
--tensor-para-size 2 \
--pipeline-para-size 1 \
--show-progress \
--input-token-len 64 \
--output-token-len 256 \
--dianxiao-path-stat  /workspace/output/firefly_random_sample_1w_256_stat_ft_tp2.json
```

