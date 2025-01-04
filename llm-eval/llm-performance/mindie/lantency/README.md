




## qwen1.5

```
nohup python performance.py  > qwen1.5-7b-2tp.log 2>&1 &
nohup python performance.py  > qwen1.5-7b-4tp.log 2>&1 &
nohup python performance.py  > qwen1.5-7b-1tp.log 2>&1 &

```

### 7b
```
nohup python performance-qwen1.5.py  > qwen1.5-7b-1tp.log 2>&1 &


首Token时延---------------------
最小值： 0.0362
最大值： 0.0822
TP50： 0.03910830750828609
TP90： 0.041983614274067806
TP99： 0.0693812740611611
平均： 0.0402
平均Token间时延-------宏平均--------------
最小值： 0.0276
最大值： 0.029
TP50： 0.0278010167882747
TP90： 0.0279021279463161
TP99： 0.028085540810020855
平均： 0.0278
Token间时延----------微平均-----------
最小值： 0.0255
最大值： 0.0769
TP50： 0.027775271475547925
TP90： 0.027990428486373276
TP99： 0.02832354519458022
平均： 0.0278
端到端时延---------------------
最小值： 0.1774
最大值： 7.2057
TP50： 5.678898271493381
TP90： 7.1518600779993
TP99： 7.181869987533427
平均： 4.5789
```


### 14b


```
nohup python performance-qwen1.5.py  > qwen1.5-14b-2tp.log 2>&1 &



首Token时延---------------------
最小值： 0.0485
最大值： 0.1147
TP50： 0.05076179598108865
TP90： 0.06449698104988784
TP99： 0.08938701630046125
平均： 0.054
平均Token间时延---------------------
最小值： 0.0291
最大值： 0.0313
TP50： 0.02938067449457426
TP90： 0.02956166821740162
TP99： 0.03005740493887995
平均： 0.0294
Token间时延---------------------
最小值： 0.0276
最大值： 0.1254
TP50： 0.02931902397540398
TP90： 0.0297731505590491
TP99： 0.030323233010713012
平均： 0.0294
端到端时延---------------------
最小值： 0.1389
最大值： 7.6702
TP50： 5.302297868474852
TP90： 7.572541862208164
TP99： 7.6225727947789705
平均： 4.5433

```


```
nohup python performance-qwen1.5-opt.py  > qwen1.5-14b-4tp.log 2>&1 &

首Token时延---------------------
最小值： 0.0342
最大值： 0.106
TP50： 0.04471042950171977
TP90： 0.06295614595292136
TP99： 0.08384593328344635
平均： 0.0491
平均Token间时延-宏平均---------------------
最小值： 0.0199
最大值： 0.0252
TP50： 0.020568159196124064
TP90： 0.0210901382120879
TP99： 0.022589506589673493
平均： 0.0206
-------------生成的Token长度
最小值： 3
最大值： 255
TP50： 163.5
TP90： 255.0
TP99： 255.0
平均： 151.149
Token间时延-微平均---------------------
最小值： 0.0168
最大值： 0.0801
TP50： 0.02059264504350722
TP90： 0.021456072013825177
TP99： 0.022933204732835267
平均： 0.0207
端到端时延---------------------
最小值： 0.127
最大值： 5.834
TP50： 3.3732377134729177
TP90： 5.3804829642409455
TP99： 5.546778464077506
平均： 3.181




--------输入的token Qwen1.5-14B
最小值： 9
最大值： 107
TP50： 17.0
TP90： 25.0
TP99： 46.02999999999997
平均： 18.463

```


输入变长：

```

20-50
nohup python performance-qwen1.5-opt.py  > qwen1.5-14b-4tp.log 2>&1 &

首Token时延---------------------
最小值： 0.0344
最大值： 0.1042
TP50： 0.044600390014238656
TP90： 0.0656163000036031
TP99： 0.08991615840815935
平均： 0.0486
平均Token间时延-宏平均---------------------
最小值： 0.0199
最大值： 0.0236
TP50： 0.020652142043143293
TP90： 0.02121227477502163
TP99： 0.022700149870105903
平均： 0.0207
最小值： 3
最大值： 255
TP50： 140.0
TP90： 255.0
TP99： 255.0
平均： 145.564
Token间时延-微平均---------------------
最小值： 0.0186
最大值： 0.1405
TP50： 0.020674455969128758
TP90： 0.0216125491540879
TP99： 0.023144232148770243
平均： 0.0208
端到端时延---------------------
最小值： 0.1068
最大值： 5.8191
TP50： 2.9713201579870656
TP90： 5.404564902291168
TP99： 5.592448376263492
平均： 3.0775



50-80
nohup python performance-qwen1.5-opt.py  > qwen1.5-14b-4tp.log 2>&1 &

首Token时延---------------------
最小值： 0.0369
最大值： 0.0841
TP50： 0.044670506031252444
TP90： 0.05576291526667773
TP99： 0.08015473581384873
平均： 0.0473
平均Token间时延-宏平均---------------------
最小值： 0.0199
最大值： 0.0273
TP50： 0.02071355515784633
TP90： 0.02131128107264086
TP99： 0.022842199074673587
平均： 0.0208
最小值： 3
最大值： 255
TP50： 93.0
TP90： 255.0
TP99： 255.0
平均： 121.6323
Token间时延-微平均---------------------
最小值： 0.0185
最大值： 0.0837
TP50： 0.02079473255435005
TP90： 0.02174522407585755
TP99： 0.02275449804146774
平均： 0.0209
端到端时延---------------------
最小值： 0.1066
最大值： 5.6484
TP50： 1.9413445619866252
TP90： 5.388641358423047
TP99： 5.5315114035015
平均： 2.5893




>80
nohup python performance-qwen1.5-opt.py  > qwen1.5-14b-4tp.log 2>&1 &

首Token时延---------------------
最小值： 0.0386
最大值： 0.0878
TP50： 0.04492756101535633
TP90： 0.05734664782648908
TP99： 0.08296710541937502
平均： 0.0484
平均Token间时延-宏平均---------------------
最小值： 0.0202
最大值： 0.0223
TP50： 0.0208923420236519
TP90： 0.02139808727210862
TP99： 0.02194536470390972
平均： 0.0209
最小值： 7
最大值： 255
TP50： 99.0
TP90： 255.0
TP99： 255.0
平均： 127.7532
Token间时延-微平均---------------------
最小值： 0.0176
最大值： 0.0838
TP50： 0.02089356805663556
TP90： 0.021700534643605354
TP99： 0.022964883805252613
平均： 0.021
端到端时延---------------------
最小值： 0.1985
最大值： 5.6518
TP50： 2.117296829004772
TP90： 5.418090309249237
TP99： 5.5139858593721875
平均： 2.7303


> 100

Token时延---------------------
最小值： 0.0427
最大值： 0.0885
TP50： 0.045819810940884054
TP90： 0.057311376091092825
TP99： 0.08563760861288758
平均： 0.0493
平均Token间时延-宏平均---------------------
最小值： 0.0202
最大值： 0.0235
TP50： 0.021073095572435044
TP90： 0.02178727035023659
TP99： 0.022659467397324773
平均： 0.0211
最小值： 12
最大值： 255
TP50： 106.0
TP90： 255.0
TP99： 255.0
平均： 131.9383
Token间时延-微平均---------------------
最小值： 0.0175
最大值： 0.0834
TP50： 0.02109450800344348
TP90： 0.022011490445584057
TP99： 0.023271926655434066
平均： 0.0212
端到端时延---------------------
最小值： 0.2921
最大值： 5.689
TP50： 2.2772261140635237
TP90： 5.426153923035599
TP99： 5.684156133793294
平均： 2.8497
```


### 72b

```
nohup python performance-qwen1.5.py  > qwen1.5-72b-8tp.log 2>&1 &


首Token时延---------------------
最小值： 0.0534
最大值： 0.2076
TP50： 0.1079369744984433
TP90： 0.1705317942192778
TP99： 0.1919716515240725
平均： 0.1158
平均Token间时延-----宏平均----------------
最小值： 0.0387
最大值： 0.0495
TP50： 0.040660905435692465
TP90： 0.04177531845047009
TP99： 0.0446021025582795
平均： 0.0407
Token间时延------微平均---------------
最小值： 0.0363
最大值： 0.2612
TP50： 0.040213780011981726
TP90： 0.04158656799700111
TP99： 0.06988605986116452
平均： 0.0409
端到端时延---------------------
最小值： 0.2416
最大值： 11.059
TP50： 6.681836381496396
TP90： 10.643284943437902
TP99： 10.850449634447578
平均： 6.2931

```




## baichuan2


### 7b

```
nohup python performance-stream-baichuan2.py > baichuan2-7b-1tp.log 2>&1 &



首Token时延---------------------
最小值： 0.0346
最大值： 0.0746
TP50： 0.037620097515173256
TP90： 0.03930058096302673
TP99： 0.0629892204713542
平均： 0.0383
Token间时延---------------------
最小值： 0.0266
最大值： 0.0282
TP50： 0.02690558272830032
TP90： 0.027013076266135427
TP99： 0.02722421399674232
平均： 0.0269
端到端时延---------------------
最小值： 0.0912
最大值： 6.9819
TP50： 3.798740677011665
TP90： 6.923744286387228
TP99： 6.9560187938367015
平均： 3.845

```


### 13b


```
nohup python performance-stream-baichuan2.py > baichuan2-13b-8tp.log 2>&1 &



Token间时延： 0.0205
端到端时延： 5.2605
首Token时延---------------------
最小值： 0.0262
最大值： 0.095
TP50： 0.04364173547946848
TP90： 0.07536821577814408
TP99： 0.08855077477928716
平均： 0.0504
Token间时延---------------------
最小值： 0.0175
最大值： 0.0233
TP50： 0.02025962653195685
TP90： 0.020924803277969476
TP99： 0.021784018275444395
平均： 0.0202
端到端时延---------------------
最小值： 0.1063
最大值： 5.5972
TP50： 3.0839127039944287
TP90： 5.346198681468377
TP99： 5.478170610390371
平均： 3.0057
```



```
nohup python performance-stream-baichuan2.py > baichuan2-13b-4tp.log 2>&1 &



首Token时延---------------------
最小值： 0.033
最大值： 0.0926
TP50： 0.040996209951117635
TP90： 0.04627442727796734
TP99： 0.08442730915965511
平均： 0.0443
Token间时延---------------------
最小值： 0.019
最大值： 0.0253
TP50： 0.01949691754800923
TP90： 0.020152707407890144
TP99： 0.02044556271550976
平均： 0.0196
生成token长度---------------------
最小值： 1
最大值： 255
TP50： 150.0
TP90： 255.0
TP99： 255.0
平均： 145.617
端到端时延---------------------
最小值： 0.0819
最大值： 5.3597
TP50： 2.9611026739585213
TP90： 5.179639161087107
TP99： 5.2417431398411285
平均： 2.933


Baichuan2-13B
最小值： 3
最大值： 89
TP50： 10.0
TP90： 18.0
TP99： 37.00999999999999
平均： 11.652
```

输入长度变长：

```
20-50 输入长度：

nohup python performance-stream-baichuan2.py > baichuan2-13b-4tp.log 2>&1 &


首Token时延---------------------
最小值： 0.0337
最大值： 0.0927
TP50： 0.041033171000890434
TP90： 0.04589057200355456
TP99： 0.08637305982760153
平均： 0.0445
Token间时延---------------------
最小值： 0.0189
最大值： 0.0225
TP50： 0.019536821587244464
TP90： 0.020144198685473597
TP99： 0.02044627025890091
平均： 0.0196
生成token长度---------------------
最小值： 1
最大值： 255
TP50： 141.5
TP90： 255.0
TP99： 255.0
平均： 142.422
端到端时延---------------------
最小值： 0.0652
最大值： 5.3101
TP50： 2.7811530429753475
TP90： 5.179711006244179
TP99： 5.248473131870851
平均： 2.8742



50-80:

nohup python performance-stream-baichuan2.py > baichuan2-13b-4tp.log 2>&1 &


首Token时延---------------------
最小值： 0.0353
最大值： 0.0902
TP50： 0.04135144897736609
TP90： 0.04567298900801688
TP99： 0.08496335084317252
平均： 0.0446
Token间时延---------------------
最小值： 0.019
最大值： 0.0237
TP50： 0.019521759752283936
TP90： 0.020296478207752693
TP99： 0.021035376845630727
平均： 0.0197
生成token长度---------------------
最小值： 1
最大值： 255
TP50： 81.0
TP90： 255.0
TP99： 255.0
平均： 120.4401
端到端时延---------------------
最小值： 0.0604
最大值： 5.4079
TP50： 1.5992199269821867
TP90： 5.206169682811014
TP99： 5.313588380513248
平均： 2.4501



>80

首Token时延---------------------
最小值： 0.0363
最大值： 0.0872
TP50： 0.041622460004873574
TP90： 0.04473990759579465
TP99： 0.08524959351751024
平均： 0.0444
Token间时延---------------------
最小值： 0.0191
最大值： 0.0216
TP50： 0.019501857135144518
TP90： 0.020219680912400066
TP99： 0.020678448217103768
平均： 0.0196
生成token长度---------------------
最小值： 8
最大值： 255
TP50： 66.0
TP90： 255.0
TP99： 255.0
平均： 106.5063
端到端时延---------------------
最小值： 0.2141
最大值： 5.3175
TP50： 1.344075930013787
TP90： 5.194844978442416
TP99： 5.291031377102481
平均： 2.1657



```




## 输入token统计

```

chatglm3:
"padded_vocab_size": 65024,  

baichuan2:
vocab_size 125696

Qwen-72B-Chat
"vocab_size": 152064


Qwen1.5-7B-Chat
"vocab_size": 151936

Qwen1.5-72B-Chat
"vocab_size": 152064
```


```
-------- Qwen1.5-7B
最小值： 9
最大值： 107
TP50： 17.0
TP90： 25.0
TP99： 46.02999999999997
平均： 18.463
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
-------- Qwen1.5-14B
最小值： 9
最大值： 107
TP50： 17.0
TP90： 25.0
TP99： 46.02999999999997
平均： 18.463
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
-------- Qwen1.5-72B
最小值： 9
最大值： 107
TP50： 17.0
TP90： 25.0
TP99： 46.02999999999997
平均： 18.463

-------- Qwen-72B
最小值： 1
最大值： 99
TP50： 9.0
TP90： 17.0
TP99： 38.02999999999997
平均： 10.463

-------- Baichuan2-7B
最小值： 3
最大值： 89
TP50： 10.0
TP90： 18.0
TP99： 37.00999999999999
平均： 11.652
-------- Baichuan2-13B
最小值： 3
最大值： 89
TP50： 10.0
TP90： 18.0
TP99： 37.00999999999999
平均： 11.652
Setting eos_token is not supported, use the default one.
Setting pad_token is not supported, use the default one.
Setting unk_token is not supported, use the default one.
-------- chatglm3-6b
最小值： 9
最大值： 101
TP50： 17.0
TP90： 24.0
TP99： 44.02999999999997
平均： 18.173

```




```
20-50:
stat_token.py 
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
-------- Qwen1.5-7B
最小值： 14
最大值： 47
TP50： 23.0
TP90： 29.0
TP99： 43.0
平均： 23.65
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
-------- Qwen1.5-14B
最小值： 14
最大值： 47
TP50： 23.0
TP90： 29.0
TP99： 43.0
平均： 23.65
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
-------- Qwen1.5-72B
最小值： 14
最大值： 47
TP50： 23.0
TP90： 29.0
TP99： 43.0
平均： 23.65
-------- Qwen-72B
最小值： 6
最大值： 39
TP50： 15.0
TP90： 21.0
TP99： 35.0
平均： 15.65
-------- Baichuan2-7B
最小值： 8
最大值： 38
TP50： 16.0
TP90： 22.0
TP99： 33.00999999999999
平均： 16.596
-------- Baichuan2-13B
最小值： 8
最大值： 38
TP50： 16.0
TP90： 22.0
TP99： 33.00999999999999
平均： 16.596
Setting eos_token is not supported, use the default one.
Setting pad_token is not supported, use the default one.
Setting unk_token is not supported, use the default one.
-------- chatglm3-6b
最小值： 14
最大值： 47
TP50： 22.0
TP90： 29.0
TP99： 40.0
平均： 23.242

```



```

50 - 80:


-------- Qwen1.5-7B
最小值： 22
最大值： 72
TP50： 40.0
TP90： 51.0
TP99： 58.420000000000016
平均： 39.9972
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
-------- Qwen1.5-14B
最小值： 22
最大值： 72
TP50： 40.0
TP90： 51.0
TP99： 58.420000000000016
平均： 39.9972
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
-------- Qwen1.5-72B
最小值： 22
最大值： 72
TP50： 40.0
TP90： 51.0
TP99： 58.420000000000016
平均： 39.9972
-------- Qwen-72B
最小值： 14
最大值： 64
TP50： 32.0
TP90： 43.0
TP99： 50.420000000000016
平均： 31.9972
-------- Baichuan2-7B
最小值： 16
最大值： 64
TP50： 32.0
TP90： 42.0
TP99： 48.84000000000003
平均： 32.3398
-------- Baichuan2-13B
最小值： 16
最大值： 64
TP50： 32.0
TP90： 42.0
TP99： 48.84000000000003
平均： 32.3398
Setting eos_token is not supported, use the default one.
Setting pad_token is not supported, use the default one.
Setting unk_token is not supported, use the default one.
-------- chatglm3-6b
最小值： 22
最大值： 71
TP50： 40.0
TP90： 50.0
TP99： 57.420000000000016
平均： 39.6462



```



```
大于80个字符

<100

Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
-------- Qwen1.5-7B
最小值： 29
最大值： 143
TP50： 65.0
TP90： 100.60000000000002
TP99： 137.44000000000005
平均： 70.7089
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
-------- Qwen1.5-14B
最小值： 29
最大值： 143
TP50： 65.0
TP90： 100.60000000000002
TP99： 137.44000000000005
平均： 70.7089
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
-------- Qwen1.5-72B
最小值： 29
最大值： 143
TP50： 65.0
TP90： 100.60000000000002
TP99： 137.44000000000005
平均： 70.7089
-------- Qwen-72B
最小值： 21
最大值： 135
TP50： 57.0
TP90： 92.60000000000002
TP99： 129.44000000000005
平均： 62.7089
-------- Baichuan2-7B
最小值： 23
最大值： 129
TP50： 56.0
TP90： 89.0
TP99： 119.86000000000001
平均： 61.1772
-------- Baichuan2-13B
最小值： 23
最大值： 129
TP50： 56.0
TP90： 89.0
TP99： 119.86000000000001
平均： 61.1772
Setting eos_token is not supported, use the default one.
Setting pad_token is not supported, use the default one.
Setting unk_token is not supported, use the default one.
-------- chatglm3-6b
最小值： 29
最大值： 138
TP50： 65.0
TP90： 98.0
TP99： 130.72000000000003
平均： 69.9873
```



```
>100 

python stat_token.py 
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
-------- Qwen1.5-7B
最小值： 40
最大值： 143
TP50： 81.0
TP90： 112.0
TP99： 142.2
平均： 83.3333
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
-------- Qwen1.5-14B
最小值： 40
最大值： 143
TP50： 81.0
TP90： 112.0
TP99： 142.2
平均： 83.3333
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
-------- Qwen1.5-72B
最小值： 40
最大值： 143
TP50： 81.0
TP90： 112.0
TP99： 142.2
平均： 83.3333
-------- Qwen-72B
最小值： 32
最大值： 135
TP50： 73.0
TP90： 104.0
TP99： 134.2
平均： 75.3333
-------- Baichuan2-7B
最小值： 34
最大值： 129
TP50： 70.0
TP90： 99.0
TP99： 122.60000000000002
平均： 72.9753
-------- Baichuan2-13B
最小值： 34
最大值： 129
TP50： 70.0
TP90： 99.0
TP99： 122.60000000000002
平均： 72.9753
Setting eos_token is not supported, use the default one.
Setting pad_token is not supported, use the default one.
Setting unk_token is not supported, use the default one.
-------- chatglm3-6b
最小值： 39
最大值： 138
TP50： 79.0
TP90： 110.0
TP99： 134.0
平均： 82.284
```
