

```
docker rm -f transformer_engine

nvidia-docker run -dti --name transformer_engine \
--restart=always --gpus all --network=host \
--shm-size 5g \
-v /home/h800/h800-work/h800-workspace:/workspace \
-w /workspace \
nvcr.io/nvidia/pytorch:23.05-py3 \
bash


sudo docker exec -it transformer_engine bash



git clone https://github.com/NVIDIA/TransformerEngine.git


```




```
> python main.py
Train Epoch: 1 [0/60000 (0%)]   Loss: 2.310981
Train Epoch: 1 [640/60000 (1%)] Loss: 2.075975
Train Epoch: 1 [1280/60000 (2%)]        Loss: 0.894793
Train Epoch: 1 [1920/60000 (3%)]        Loss: 0.733659
Train Epoch: 1 [2560/60000 (4%)]        Loss: 0.607289
Train Epoch: 1 [3200/60000 (5%)]        Loss: 0.512597
Train Epoch: 1 [3840/60000 (6%)]        Loss: 0.467567
Train Epoch: 1 [4480/60000 (7%)]        Loss: 0.453665
Train Epoch: 1 [5120/60000 (9%)]        Loss: 0.358288
Train Epoch: 1 [5760/60000 (10%)]       Loss: 0.286079
Train Epoch: 1 [6400/60000 (11%)]       Loss: 0.506863
Train Epoch: 1 [7040/60000 (12%)]       Loss: 0.457581
Train Epoch: 1 [7680/60000 (13%)]       Loss: 0.174460
Train Epoch: 1 [8320/60000 (14%)]       Loss: 0.207888
Train Epoch: 1 [8960/60000 (15%)]       Loss: 0.109221
Train Epoch: 1 [9600/60000 (16%)]       Loss: 0.115240
Train Epoch: 1 [10240/60000 (17%)]      Loss: 0.376108
Train Epoch: 1 [10880/60000 (18%)]      Loss: 0.190710
Train Epoch: 1 [11520/60000 (19%)]      Loss: 0.135155
Train Epoch: 1 [12160/60000 (20%)]      Loss: 0.128043
Train Epoch: 1 [12800/60000 (21%)]      Loss: 0.128150
Train Epoch: 1 [13440/60000 (22%)]      Loss: 0.288093
Train Epoch: 1 [14080/60000 (23%)]      Loss: 0.319398
Train Epoch: 1 [14720/60000 (25%)]      Loss: 0.110766
Train Epoch: 1 [15360/60000 (26%)]      Loss: 0.141596
Train Epoch: 1 [16000/60000 (27%)]      Loss: 0.274476
Train Epoch: 1 [16640/60000 (28%)]      Loss: 0.196867
Train Epoch: 1 [17280/60000 (29%)]      Loss: 0.320345
Train Epoch: 1 [17920/60000 (30%)]      Loss: 0.281575
Train Epoch: 1 [18560/60000 (31%)]      Loss: 0.142241
Train Epoch: 1 [19200/60000 (32%)]      Loss: 0.175851
Train Epoch: 1 [19840/60000 (33%)]      Loss: 0.110049
Train Epoch: 1 [20480/60000 (34%)]      Loss: 0.219275
Train Epoch: 1 [21120/60000 (35%)]      Loss: 0.058848
Train Epoch: 1 [21760/60000 (36%)]      Loss: 0.281849
Train Epoch: 1 [22400/60000 (37%)]      Loss: 0.029589
Train Epoch: 1 [23040/60000 (38%)]      Loss: 0.067243
Train Epoch: 1 [23680/60000 (39%)]      Loss: 0.063093
Train Epoch: 1 [24320/60000 (41%)]      Loss: 0.430286
Train Epoch: 1 [24960/60000 (42%)]      Loss: 0.076447
Train Epoch: 1 [25600/60000 (43%)]      Loss: 0.059815
Train Epoch: 1 [26240/60000 (44%)]      Loss: 0.453320
Train Epoch: 1 [26880/60000 (45%)]      Loss: 0.354056
Train Epoch: 1 [27520/60000 (46%)]      Loss: 0.109220
Train Epoch: 1 [28160/60000 (47%)]      Loss: 0.359484
Train Epoch: 1 [28800/60000 (48%)]      Loss: 0.148566
Train Epoch: 1 [29440/60000 (49%)]      Loss: 0.105767
Train Epoch: 1 [30080/60000 (50%)]      Loss: 0.099680
Train Epoch: 1 [30720/60000 (51%)]      Loss: 0.024636
Train Epoch: 1 [31360/60000 (52%)]      Loss: 0.023627
Train Epoch: 1 [32000/60000 (53%)]      Loss: 0.117661
Train Epoch: 1 [32640/60000 (54%)]      Loss: 0.140753
Train Epoch: 1 [33280/60000 (55%)]      Loss: 0.062557
Train Epoch: 1 [33920/60000 (57%)]      Loss: 0.127518
Train Epoch: 1 [34560/60000 (58%)]      Loss: 0.148100
Train Epoch: 1 [35200/60000 (59%)]      Loss: 0.116498
Train Epoch: 1 [35840/60000 (60%)]      Loss: 0.190453
Train Epoch: 1 [36480/60000 (61%)]      Loss: 0.095069
Train Epoch: 1 [37120/60000 (62%)]      Loss: 0.100426
Train Epoch: 1 [37760/60000 (63%)]      Loss: 0.067289
Train Epoch: 1 [38400/60000 (64%)]      Loss: 0.075233
Train Epoch: 1 [39040/60000 (65%)]      Loss: 0.132019
Train Epoch: 1 [39680/60000 (66%)]      Loss: 0.153820
Train Epoch: 1 [40320/60000 (67%)]      Loss: 0.121976
Train Epoch: 1 [40960/60000 (68%)]      Loss: 0.029021
Train Epoch: 1 [41600/60000 (69%)]      Loss: 0.038629
Train Epoch: 1 [42240/60000 (70%)]      Loss: 0.144574
Train Epoch: 1 [42880/60000 (71%)]      Loss: 0.121906
Train Epoch: 1 [43520/60000 (72%)]      Loss: 0.101441
Train Epoch: 1 [44160/60000 (74%)]      Loss: 0.040515
Train Epoch: 1 [44800/60000 (75%)]      Loss: 0.147051
Train Epoch: 1 [45440/60000 (76%)]      Loss: 0.090034
Train Epoch: 1 [46080/60000 (77%)]      Loss: 0.261030
Train Epoch: 1 [46720/60000 (78%)]      Loss: 0.115955
Train Epoch: 1 [47360/60000 (79%)]      Loss: 0.111859
Train Epoch: 1 [48000/60000 (80%)]      Loss: 0.073982
Train Epoch: 1 [48640/60000 (81%)]      Loss: 0.237517
Train Epoch: 1 [49280/60000 (82%)]      Loss: 0.030576
Train Epoch: 1 [49920/60000 (83%)]      Loss: 0.118248
Train Epoch: 1 [50560/60000 (84%)]      Loss: 0.092839
Train Epoch: 1 [51200/60000 (85%)]      Loss: 0.053416
Train Epoch: 1 [51840/60000 (86%)]      Loss: 0.287856
Train Epoch: 1 [52480/60000 (87%)]      Loss: 0.120020
Train Epoch: 1 [53120/60000 (88%)]      Loss: 0.244491
Train Epoch: 1 [53760/60000 (90%)]      Loss: 0.187061
Train Epoch: 1 [54400/60000 (91%)]      Loss: 0.045214
Train Epoch: 1 [55040/60000 (92%)]      Loss: 0.115131
Train Epoch: 1 [55680/60000 (93%)]      Loss: 0.010176
Train Epoch: 1 [56320/60000 (94%)]      Loss: 0.030591
Train Epoch: 1 [56960/60000 (95%)]      Loss: 0.196713
Train Epoch: 1 [57600/60000 (96%)]      Loss: 0.075411
Train Epoch: 1 [58240/60000 (97%)]      Loss: 0.215062
Train Epoch: 1 [58880/60000 (98%)]      Loss: 0.177358
Train Epoch: 1 [59520/60000 (99%)]      Loss: 0.084014

Test set: Average loss: 0.0501, Accuracy: 9836/10000 (98%)

Train Epoch: 2 [0/60000 (0%)]   Loss: 0.064148
Train Epoch: 2 [640/60000 (1%)] Loss: 0.061523
...
Train Epoch: 2 [58240/60000 (97%)]      Loss: 0.023575
Train Epoch: 2 [58880/60000 (98%)]      Loss: 0.113758
Train Epoch: 2 [59520/60000 (99%)]      Loss: 0.007068

Test set: Average loss: 0.0380, Accuracy: 9872/10000 (99%)

Train Epoch: 3 [0/60000 (0%)]   Loss: 0.173563
Train Epoch: 3 [640/60000 (1%)] Loss: 0.018026
Train Epoch: 3 [1280/60000 (2%)]        Loss: 0.022182
Train Epoch: 3 [1920/60000 (3%)]        Loss: 0.030231
...
Train Epoch: 3 [58240/60000 (97%)]      Loss: 0.008527
Train Epoch: 3 [58880/60000 (98%)]      Loss: 0.027886
Train Epoch: 3 [59520/60000 (99%)]      Loss: 0.038440

Test set: Average loss: 0.0350, Accuracy: 9888/10000 (99%)

Train Epoch: 4 [0/60000 (0%)]   Loss: 0.035976
Train Epoch: 4 [640/60000 (1%)] Loss: 0.027167
Train Epoch: 4 [1280/60000 (2%)]        Loss: 0.038828
Train Epoch: 4 [1920/60000 (3%)]        Loss: 0.075179
Train Epoch: 4 [2560/60000 (4%)]        Loss: 0.069215
Train Epoch: 4 [3200/60000 (5%)]        Loss: 0.037071
...
Train Epoch: 4 [51200/60000 (85%)]      Loss: 0.084609
Train Epoch: 4 [51840/60000 (86%)]      Loss: 0.021072
Train Epoch: 4 [52480/60000 (87%)]      Loss: 0.042509
Train Epoch: 4 [53120/60000 (88%)]      Loss: 0.059584
Train Epoch: 4 [53760/60000 (90%)]      Loss: 0.013509
Train Epoch: 4 [54400/60000 (91%)]      Loss: 0.059899
Train Epoch: 4 [55040/60000 (92%)]      Loss: 0.072503
Train Epoch: 4 [55680/60000 (93%)]      Loss: 0.011318
Train Epoch: 4 [56320/60000 (94%)]      Loss: 0.012140
Train Epoch: 4 [56960/60000 (95%)]      Loss: 0.015528
Train Epoch: 4 [57600/60000 (96%)]      Loss: 0.153004
Train Epoch: 4 [58240/60000 (97%)]      Loss: 0.013440
Train Epoch: 4 [58880/60000 (98%)]      Loss: 0.050854
Train Epoch: 4 [59520/60000 (99%)]      Loss: 0.037339

Test set: Average loss: 0.0324, Accuracy: 9903/10000 (99%)

Train Epoch: 5 [0/60000 (0%)]   Loss: 0.074965
...
Train Epoch: 5 [58240/60000 (97%)]      Loss: 0.007150
Train Epoch: 5 [58880/60000 (98%)]      Loss: 0.005998
Train Epoch: 5 [59520/60000 (99%)]      Loss: 0.020781

Test set: Average loss: 0.0290, Accuracy: 9911/10000 (99%)

...

Train Epoch: 14 [0/60000 (0%)]  Loss: 0.002542
Train Epoch: 14 [640/60000 (1%)]        Loss: 0.034651
Train Epoch: 14 [1280/60000 (2%)]       Loss: 0.006098
Train Epoch: 14 [1920/60000 (3%)]       Loss: 0.008892
Train Epoch: 14 [2560/60000 (4%)]       Loss: 0.000881
Train Epoch: 14 [3200/60000 (5%)]       Loss: 0.005196
Train Epoch: 14 [3840/60000 (6%)]       Loss: 0.116072
Train Epoch: 14 [4480/60000 (7%)]       Loss: 0.004457
Train Epoch: 14 [5120/60000 (9%)]       Loss: 0.075053
Train Epoch: 14 [5760/60000 (10%)]      Loss: 0.002541
Train Epoch: 14 [6400/60000 (11%)]      Loss: 0.008487
Train Epoch: 14 [7040/60000 (12%)]      Loss: 0.010142
Train Epoch: 14 [7680/60000 (13%)]      Loss: 0.018750
Train Epoch: 14 [8320/60000 (14%)]      Loss: 0.022964
Train Epoch: 14 [8960/60000 (15%)]      Loss: 0.018425
Train Epoch: 14 [9600/60000 (16%)]      Loss: 0.083772
Train Epoch: 14 [10240/60000 (17%)]     Loss: 0.005031
Train Epoch: 14 [10880/60000 (18%)]     Loss: 0.006118
Train Epoch: 14 [11520/60000 (19%)]     Loss: 0.012194
Train Epoch: 14 [12160/60000 (20%)]     Loss: 0.026096
Train Epoch: 14 [12800/60000 (21%)]     Loss: 0.005872
Train Epoch: 14 [13440/60000 (22%)]     Loss: 0.002419
Train Epoch: 14 [14080/60000 (23%)]     Loss: 0.004439
Train Epoch: 14 [14720/60000 (25%)]     Loss: 0.000334
Train Epoch: 14 [15360/60000 (26%)]     Loss: 0.001956
Train Epoch: 14 [16000/60000 (27%)]     Loss: 0.023584
Train Epoch: 14 [16640/60000 (28%)]     Loss: 0.032739
Train Epoch: 14 [17280/60000 (29%)]     Loss: 0.001395
Train Epoch: 14 [17920/60000 (30%)]     Loss: 0.052542
Train Epoch: 14 [18560/60000 (31%)]     Loss: 0.003781
Train Epoch: 14 [19200/60000 (32%)]     Loss: 0.002781
Train Epoch: 14 [19840/60000 (33%)]     Loss: 0.040734
Train Epoch: 14 [20480/60000 (34%)]     Loss: 0.002403
Train Epoch: 14 [21120/60000 (35%)]     Loss: 0.009741
Train Epoch: 14 [21760/60000 (36%)]     Loss: 0.003009
Train Epoch: 14 [22400/60000 (37%)]     Loss: 0.001433
Train Epoch: 14 [23040/60000 (38%)]     Loss: 0.017763
Train Epoch: 14 [23680/60000 (39%)]     Loss: 0.032766
Train Epoch: 14 [24320/60000 (41%)]     Loss: 0.055031
Train Epoch: 14 [24960/60000 (42%)]     Loss: 0.014967
Train Epoch: 14 [25600/60000 (43%)]     Loss: 0.097655
Train Epoch: 14 [26240/60000 (44%)]     Loss: 0.011274
Train Epoch: 14 [26880/60000 (45%)]     Loss: 0.045538
Train Epoch: 14 [27520/60000 (46%)]     Loss: 0.073794
Train Epoch: 14 [28160/60000 (47%)]     Loss: 0.002871
Train Epoch: 14 [28800/60000 (48%)]     Loss: 0.104455
Train Epoch: 14 [29440/60000 (49%)]     Loss: 0.046128
Train Epoch: 14 [30080/60000 (50%)]     Loss: 0.027590
Train Epoch: 14 [30720/60000 (51%)]     Loss: 0.023746
Train Epoch: 14 [31360/60000 (52%)]     Loss: 0.029652
Train Epoch: 14 [32000/60000 (53%)]     Loss: 0.014070
Train Epoch: 14 [32640/60000 (54%)]     Loss: 0.003088
Train Epoch: 14 [33280/60000 (55%)]     Loss: 0.001508
Train Epoch: 14 [33920/60000 (57%)]     Loss: 0.008475
Train Epoch: 14 [34560/60000 (58%)]     Loss: 0.013877
Train Epoch: 14 [35200/60000 (59%)]     Loss: 0.011066
Train Epoch: 14 [35840/60000 (60%)]     Loss: 0.047547
Train Epoch: 14 [36480/60000 (61%)]     Loss: 0.068665
Train Epoch: 14 [37120/60000 (62%)]     Loss: 0.006788
Train Epoch: 14 [37760/60000 (63%)]     Loss: 0.013033
Train Epoch: 14 [38400/60000 (64%)]     Loss: 0.002599
Train Epoch: 14 [39040/60000 (65%)]     Loss: 0.036043
Train Epoch: 14 [39680/60000 (66%)]     Loss: 0.028148
Train Epoch: 14 [40320/60000 (67%)]     Loss: 0.157269
Train Epoch: 14 [40960/60000 (68%)]     Loss: 0.054018
Train Epoch: 14 [41600/60000 (69%)]     Loss: 0.061800
Train Epoch: 14 [42240/60000 (70%)]     Loss: 0.013732
Train Epoch: 14 [42880/60000 (71%)]     Loss: 0.003842
Train Epoch: 14 [43520/60000 (72%)]     Loss: 0.003533
Train Epoch: 14 [44160/60000 (74%)]     Loss: 0.101497
Train Epoch: 14 [44800/60000 (75%)]     Loss: 0.004142
Train Epoch: 14 [45440/60000 (76%)]     Loss: 0.026316
Train Epoch: 14 [46080/60000 (77%)]     Loss: 0.006387
Train Epoch: 14 [46720/60000 (78%)]     Loss: 0.130743
Train Epoch: 14 [47360/60000 (79%)]     Loss: 0.005992
Train Epoch: 14 [48000/60000 (80%)]     Loss: 0.002442
Train Epoch: 14 [48640/60000 (81%)]     Loss: 0.069629
Train Epoch: 14 [49280/60000 (82%)]     Loss: 0.002736
Train Epoch: 14 [49920/60000 (83%)]     Loss: 0.003623
Train Epoch: 14 [50560/60000 (84%)]     Loss: 0.002044
Train Epoch: 14 [51200/60000 (85%)]     Loss: 0.017906
Train Epoch: 14 [51840/60000 (86%)]     Loss: 0.067535
Train Epoch: 14 [52480/60000 (87%)]     Loss: 0.078820
Train Epoch: 14 [53120/60000 (88%)]     Loss: 0.004781
Train Epoch: 14 [53760/60000 (90%)]     Loss: 0.111643
Train Epoch: 14 [54400/60000 (91%)]     Loss: 0.003517
Train Epoch: 14 [55040/60000 (92%)]     Loss: 0.048870
Train Epoch: 14 [55680/60000 (93%)]     Loss: 0.001008
Train Epoch: 14 [56320/60000 (94%)]     Loss: 0.003493
Train Epoch: 14 [56960/60000 (95%)]     Loss: 0.007140
Train Epoch: 14 [57600/60000 (96%)]     Loss: 0.019242
Train Epoch: 14 [58240/60000 (97%)]     Loss: 0.000630
Train Epoch: 14 [58880/60000 (98%)]     Loss: 0.014589
Train Epoch: 14 [59520/60000 (99%)]     Loss: 0.079487

Test set: Average loss: 0.0294, Accuracy: 9912/10000 (99%)

train time list: [7.174770716985222, 6.396849530981854, 6.700996204977855, 6.140903544030152, 6.5446675819694065, 6.775199636991601, 6.91574348002905, 6.429244890983682, 6.813953557983041, 6.112987361033447, 6.425478958000895, 6.353286882978864, 7.112901913002133, 7.050599466951098]
inference time list: [1.042455518967472, 1.0501488799927756, 1.039717328036204, 1.0230632729944773, 1.0144481860334054, 0.9574590580305085, 0.9778538069804199, 1.0533951570396312, 1.0227251390460879, 0.9527524459990673, 1.0071292700013146, 1.041757729020901, 1.0557105189654976, 1.050325177027844]
train time: 92.9475837268983
inference time: 14.288941488135606
```



```
> python main.py --use-te
...
Train Epoch: 14 [58240/60000 (97%)]     Loss: 0.001275
Train Epoch: 14 [58880/60000 (98%)]     Loss: 0.018255
Train Epoch: 14 [59520/60000 (99%)]     Loss: 0.190788

Test set: Average loss: 0.0258, Accuracy: 9921/10000 (99%)

train time list: [6.391690019983798, 6.792517438996583, 6.072094923991244, 6.099119610036723, 6.473783936991822, 6.634461442998145, 6.545820511004422, 6.504633364034817, 5.760160473990254, 7.329169453005306, 6.647246962995268, 6.197783594019711, 6.261224630987272, 6.351588386984076]
inference time list: [1.010965807014145, 0.9527024410199374, 1.0541439579683356, 0.9858089959598146, 1.0347943140077405, 1.0267179679940455, 0.9497789539746009, 0.9827587549807504, 0.9495672340271994, 1.0524603630183265, 1.040817208995577, 1.0367748150019906, 0.9209445589804091, 1.0245853600208648]
train time: 90.06129475001944
inference time: 14.022820732963737
```




```
>  python main.py --use-fp8
...
Train Epoch: 14 [55040/60000 (92%)]     Loss: 0.000463
Train Epoch: 14 [55680/60000 (93%)]     Loss: 0.002357
Train Epoch: 14 [56320/60000 (94%)]     Loss: 0.079992
Train Epoch: 14 [56960/60000 (95%)]     Loss: 0.010031
Train Epoch: 14 [57600/60000 (96%)]     Loss: 0.012265
Train Epoch: 14 [58240/60000 (97%)]     Loss: 0.002443
Train Epoch: 14 [58880/60000 (98%)]     Loss: 0.054622
Train Epoch: 14 [59520/60000 (99%)]     Loss: 0.126032

Test set: Average loss: 0.0250, Accuracy: 9925/10000 (99%)

train time list: [7.210564187029377, 6.6171875660074875, 6.091605413996149, 5.305714985996019, 5.808578662981745, 6.6427561830496415, 6.510479367978405, 6.239163941005245, 6.416430394980125, 6.473377400950994, 6.813092800031882, 6.601396489015315, 6.405762499955017, 5.858293619006872]
inference time list: [1.0294796290108934, 0.9135813339962624, 0.9606331089744344, 1.005576271971222, 1.0113011699868366, 1.0482170059694909, 0.9300815680180676, 1.048414018994663, 0.9391937680193223, 1.0117242050473578, 0.9825000909622759, 0.95032776996959, 0.934907581016887, 0.9126828419975936]
train time: 88.99440351198427
inference time: 13.678620363934897

```




```
> python main_stat.py
train time list:  [7.087993178982288, 7.079371533007361, 7.364974033029284, 7.1193796820007265, 7.02183426899137, 5.9516880650189705, 6.674289498012513, 6.816586718952749, 6.995105146022979, 7.027234941022471, 6.648724443977699, 6.042013334052172, 6.5562196180108, 5.603573656990193]
inference time list:  [0.9502711769891903, 1.0733181409887038, 0.9475853049661964, 0.973363564000465, 0.9129067260073498, 1.0311341150081716, 1.0374468020163476, 1.0528108790167607, 1.05149551195791, 1.0637726149871014, 0.9370641469722614, 0.9356176999863237, 1.0465723529923707, 1.0388881860417314]
train time:  93.98898811807157
inference time:  14.052247221930884
sum_train_time_list:  [0.9921075316960923, 0.979746263648849, 1.295295728952624, 0.8635914049809799, 0.6020516679272987, 0.5717894238187, 0.8129167315200903, 0.7123352510388941, 0.9578135043848306, 0.8523455414106138, 0.4895935410168022, 0.6240919533884153, 0.6424971928354353, 0.699452716158703]
sum_test_time_list:  [0.022764541092328727, 0.010617257852572948, 0.011623128957580775, 0.008137622033245862, 0.008895464998204261, 0.005871215951628983, 0.010772994079161435, 0.00680145708611235, 0.006934473058208823, 0.013688619888853282, 0.00445437798043713, 0.006612447032239288, 0.01153424585936591, 0.006899168132804334]
sum_train_time_list:  11.095628452778328
sum_test_time_list:  0.1356070140027441
```

```
> python main_stat.py  --use-te
train time list:  [6.577109163044952, 6.456322034006007, 6.206114369968418, 6.317878360976465, 6.13362843496725, 6.5731987790204585, 6.681127022020519, 6.562263609026559, 5.871911644004285, 6.824332197022159, 6.278518081991933, 5.295758633001242, 6.421999779995531, 5.8961328599834815]
inference time list:  [1.0320385239901952, 1.0516188929905184, 1.03763318201527, 0.9469043670105748, 0.9094336919952184, 1.029361633991357, 0.9276270279660821, 0.9503992049722001, 0.9650351019809023, 1.0215440799947828, 1.0220137339783832, 0.9601732759620063, 1.0666903390083462, 0.9581517709884793]
train time:  88.09629496902926
inference time:  13.878624826844316
sum_train_time_list:  [1.2195107479346916, 0.7842218662262894, 0.620298806228675, 0.6155243891989812, 0.678798888984602, 1.4767131984699517, 0.6328003642847762, 0.7442239616648294, 0.8118451639311388, 0.9232772113755345, 0.6797919715172611, 1.3140820758999325, 0.493738787365146, 0.694927082862705]
sum_test_time_list:  [0.019329448929056525, 0.008496595080941916, 0.006880718923639506, 0.006158717966172844, 0.009659261035267264, 0.011841151979751885, 0.006156609044410288, 0.010696726909372956, 0.018550584965851158, 0.007702818897087127, 0.009280916943680495, 0.006388182984665036, 0.008835386077407748, 0.007065327954478562]
sum_train_time_list:  11.689754515944514
sum_test_time_list:  0.1370424476917833
```


```
> python main_stat.py --use-fp8
```





