


下表列出了不同 GPU 产品上支持的功能。

| Feature Group | Tesla | Titan | Quadro | GeForce |
| --- | --- | --- | --- | --- |
| Field Value Watches (GPU metrics) | X | X | X | X |
| Configuration Management | X | X | X | X |
| Active Health Checks (GPU subsystems) | X | X | X | X |
| Job Statistics | X | X | X | X |
| Topology | X | X | X | X |
| Introspection | X | X | X | X |
| Policy Notification | X | | | |
| GPU Diagnostics (Diagnostic Levels - 1, 2, 3) | All Levels | Level 1 | Level 1 | Level 1 |

## dcgmi discovery

验证是否能够找到 GPU 设备

```
> dcgmi discovery -l
8 GPUs found.
+--------+----------------------------------------------------------------------+
| GPU ID | Device Information                                                   |
+--------+----------------------------------------------------------------------+
| 0      | Name: NVIDIA H800                                                    |
|        | PCI Bus ID: 00000000:18:00.0                                         |
|        | Device UUID: GPU-34bf77d1-c686-6821-79a8-32d326c5039c                |
+--------+----------------------------------------------------------------------+
| 1      | Name: NVIDIA H800                                                    |
|        | PCI Bus ID: 00000000:3E:00.0                                         |
|        | Device UUID: GPU-f5046fa5-3db4-45e8-870a-dc1376becaa5                |
+--------+----------------------------------------------------------------------+
| 2      | Name: NVIDIA H800                                                    |
|        | PCI Bus ID: 00000000:51:00.0                                         |
|        | Device UUID: GPU-9de407ad-ba9c-af12-ce09-65828829a67c                |
+--------+----------------------------------------------------------------------+
| 3      | Name: NVIDIA H800                                                    |
|        | PCI Bus ID: 00000000:65:00.0                                         |
|        | Device UUID: GPU-b54d703a-dee5-a9da-aeb9-465003acdd4b                |
+--------+----------------------------------------------------------------------+
| 4      | Name: NVIDIA H800                                                    |
|        | PCI Bus ID: 00000000:98:00.0                                         |
|        | Device UUID: GPU-09c6e33a-ffcf-b330-e68b-e1e9f745eae6                |
+--------+----------------------------------------------------------------------+
| 5      | Name: NVIDIA H800                                                    |
|        | PCI Bus ID: 00000000:BD:00.0                                         |
|        | Device UUID: GPU-9a8ef0b8-9816-459d-fa13-cda74cf19d37                |
+--------+----------------------------------------------------------------------+
| 6      | Name: NVIDIA H800                                                    |
|        | PCI Bus ID: 00000000:CF:00.0                                         |
|        | Device UUID: GPU-70c5b9a8-82a3-4199-d7f5-adb9186459eb                |
+--------+----------------------------------------------------------------------+
| 7      | Name: NVIDIA H800                                                    |
|        | PCI Bus ID: 00000000:E2:00.0                                         |
|        | Device UUID: GPU-474d838c-171f-d249-4f45-bbc01a8eb74a                |
+--------+----------------------------------------------------------------------+
0 NvSwitches found.
+-----------+
| Switch ID |
+-----------+
+-----------+
```


## dcgmi group

```
dcgmi group -l
+-------------------+----------------------------------------------------------+
| GROUPS                                                                       |
| 2 groups found.                                                              |
+===================+==========================================================+
| Groups            |                                                          |
| -> 0              |                                                          |
|    -> Group ID    | 0                                                        |
|    -> Group Name  | DCGM_ALL_SUPPORTED_GPUS                                  |
|    -> Entities    | GPU 0, GPU 1, GPU 2, GPU 3, GPU 4, GPU 5, GPU 6, GPU 7   |
| -> 1              |                                                          |
|    -> Group ID    | 1                                                        |
|    -> Group Name  | DCGM_ALL_SUPPORTED_NVSWITCHES                            |
|    -> Entities    | None                                                     |
+-------------------+----------------------------------------------------------+

```



## dcgmi dmon

用于监控 GPU 及其统计数据
```
> dcgmi dmon --help

 dmon -- Used to monitor GPUs and their stats.

Usage: dcgmi dmon
   dcgmi dmon -i <gpuId> -g <groupId> -f <fieldGroupId> -e <fieldId> -d
        <delay> -c <count> -l

Flags:
      --host       IP/FQDN    Connects to specified IP or fully-qualified domain
                               name. To connect to a host engine that was
                               started with -d (unix socket), prefix the unix
                               socket filename with 'unix://'. [default =
                               localhost]
  -f  --field-group-idfieldGroupId  The field group to query on the specified
                               host.
  -e  --field-id   fieldId     Field identifier to view/inject.（要查看的字段ID）
  -l  --list                  List to look up the long names, short names and
                               field ids.（用于查找长名称、短名称和字段 ID 的列表。）
  -h  --help                  Displays usage information and exits.
  -i  --gpu-id     gpuId       The comma separated list of GPU/GPU-I/GPU-CI IDs
                               to run the dmon on. Default is -1 which runs for
                               all supported GPU. Run dcgmi discovery -c to
                               check list of available GPU entities （用于运行守护程序的 GPU/GPU-I/GPU-CI ID 的逗号分隔列表。 默认值为 -1，适用于所有支持的 GPU。 运行 dcgmi discovery -c 以检查可用 GPU 实体列表）
  -g  --group-id   groupId     The group to query on the specified host.
  -d  --delay      delay       In milliseconds. Integer representing how often
                               to query results from DCGM and print them for all
                               of the entities. [default = 1000 msec,  Minimum
                               value = 1 msec.](以毫秒为单位。 表示从 DCGM 查询结果并为所有实体打印结果的频率。 [默认 = 1000 毫秒，最小值 = 1 毫秒)
  -c  --count      count       Integer representing How many times to loop
                               before exiting. [default- runs forever.]（表示退出前循环次数。[默认值-永远运行]）
  --  --ignore_rest           Ignores the rest of the labeled arguments
                               following this flag.

NVIDIA Datacenter GPU Management Interface
```



```
> dcgmi dmon -l
___________________________________________________________________________________________________________
Long Name                                                                    Short Name          Field ID
___________________________________________________________________________________________________________
driver_version                                                                DRVER               1
nvml_version                                                                  NVVER               2
process_name                                                                  PRNAM               3
device_count                                                                  DVCNT               4
cuda_driver_version                                                           CDVER               5
name                                                                          DVNAM               50
brand                                                                         DVBRN               51
nvml_index                                                                    NVIDX               52
serial_number                                                                 SRNUM               53
uuid                                                                          UUID#               54
minor_number                                                                  MNNUM               55
oem_inforom_version                                                           OEMVR               56
pci_busid                                                                     PCBID               57
pci_combined_id                                                               PCCID               58
pci_subsys_id                                                                 PCSID               59
system_topology_pci                                                           STVCI               60
system_topology_nvlink                                                        STNVL               61
system_affinity                                                               SYSAF               62
cuda_compute_capability                                                       DVCCC               63
compute_mode                                                                  CMMOD               65
persistance_mode                                                              PMMOD               66
mig_mode                                                                      MGMOD               67
cuda_visible_devices                                                          CUVID               68
mig_max_slices                                                                MIGMS               69
cpu_affinity_0                                                                CAFF0               70
cpu_affinity_1                                                                CAFF1               71
cpu_affinity_2                                                                CAFF2               72
cpu_affinity_3                                                                CAFF3               73
cc_mode                                                                       CCMOD               74
mig_attributes                                                                MIGATT              75
mig_gi_info                                                                   MIGGIINFO           76
mig_ci_info                                                                   MIGCIINFO           77
ecc_inforom_version                                                           EIVER               80
power_inforom_version                                                         PIVER               81
inforom_image_version                                                         IIVER               82
inforom_config_checksum                                                       CCSUM               83
inforom_config_valid                                                          ICVLD               84
vbios_version                                                                 VBVER               85
bar1_total                                                                    B1TTL               90
sync_boost                                                                    SYBST               91
bar1_used                                                                     B1USE               92
bar1_free                                                                     B1FRE               93
sm_clock                                                                      SMCLK               100
memory_clock                                                                  MMCLK               101
video_clock                                                                   VICLK               102
sm_app_clock                                                                  SACLK               110
mem_app_clock                                                                 MACLK               111
current_clock_throttle_reasons                                                DVCCTR              112
sm_max_clock                                                                  SMMAX               113
memory_max_clock                                                              MMMAX               114
video_max_clock                                                               VIMAX               115
autoboost                                                                     ATBST               120
supported_clocks                                                              SPCLK               130
memory_temp                                                                   MMTMP               140
gpu_temp                                                                      TMPTR               150
gpu_mem_max_op_temp                                                           GMMOT               151
gpu_max_op_temp                                                               GGMOT               152
power_usage                                                                   POWER               155
total_energy_consumption                                                      TOTEC               156
slowdown_temp                                                                 SDTMP               158
shutdown_temp                                                                 SHTMP               159
power_management_limit                                                        PMLMT               160
power_management_limit_min                                                    PMMIN               161
power_management_limit_max                                                    PMMAX               162
power_management_limit_default                                                PMDEF               163
enforced_power_limit                                                          EPLMT               164
pstate                                                                        PSTAT               190
fan_speed                                                                     FANSP               191
pcie_tx_throughput                                                            TXTPT               200
pcie_rx_throughput                                                            RXTPT               201
pcie_replay_counter                                                           RPCTR               202
gpu_utilization                                                               GPUTL               203
mem_copy_utilization                                                          MCUTL               204
accounting_data                                                               ACCDT               205
enc_utilization                                                               ECUTL               206
dec_utilization                                                               DCUTL               207
mem_util_samples                                                              MUSAM               210
gpu_util_samples                                                              GUSAM               211
graphics_pids                                                                 GPIDS               220
compute_pids                                                                  CMPID               221
xid_errors                                                                    XIDER               230
pcie_max_link_gen                                                             PCIMG               235
pcie_max_link_width                                                           PCIMW               236
pcie_link_gen                                                                 PCILG               237
pcie_link_width                                                               PCILW               238
power_violation                                                               PVIOL               240
thermal_violation                                                             TVIOL               241
sync_boost_violation                                                          SBVIO               242
board_limit_violation                                                         BLVIO               243
low_util_violation                                                            LUVIO               244
reliability_violation                                                         RVIOL               245
app_clock_violation                                                           TAPCV               246
base_clock_violation                                                          TAPBC               247
fb_total                                                                      FBTTL               250
fb_free                                                                       FBFRE               251
fb_used                                                                       FBUSD               252
fb_resv                                                                       FBRSV               253
fb_USDP                                                                       FBUSP               254
ecc                                                                           ECCUR               300
ecc_pending                                                                   ECPEN               301
ecc_sbe_volatile_total                                                        ESVTL               310
ecc_dbe_volatile_total                                                        EDVTL               311
ecc_sbe_aggregate_total                                                       ESATL               312
ecc_dbe_aggregate_total                                                       EDATL               313
ecc_sbe_volatile_l1                                                           ESVL1               314
ecc_dbe_volatile_l1                                                           EDVL1               315
ecc_sbe_volatile_l2                                                           ESVL2               316
ecc_dbe_volatile_l2                                                           EDVL2               317
ecc_sbe_volatile_device                                                       ESVDV               318
ecc_dbe_volatile_device                                                       EDVDV               319
ecc_sbe_volatile_register                                                     ESVRG               320
ecc_dbe_volatile_register                                                     EDVRG               321
ecc_sbe_volatile_texture                                                      ESVTX               322
ecc_dbe_volatile_texture                                                      EDVTX               323
ecc_sbe_aggregate_l1                                                          ESAL1               324
ecc_dbe_aggregate_l1                                                          EDAL1               325
ecc_sbe_aggregate_l2                                                          ESAL2               326
ecc_dbe_aggregate_l2                                                          EDAL2               327
ecc_sbe_aggregate_device                                                      ESADV               328
ecc_dbe_aggregate_device                                                      EDADV               329
ecc_sbe_aggregate_register                                                    ESARG               330
ecc_dbe_aggregate_register                                                    EDARG               331
ecc_sbe_aggregate_texture                                                     ESATX               332
ecc_dbe_aggregate_texture                                                     EDATX               333
retired_pages_sbe                                                             RPSBE               390
retired_pages_dbe                                                             RPDBE               391
retired_pages_pending                                                         RPPEN               392
uncorrectable_remapped_rows                                                   URMPS               393
correctable_remapped_rows                                                     CRMPS               394
row_remap_failure                                                             RRF                 395
row_remap_pending                                                             RRP                 396
nvlink_flit_crc_error_count_l0                                                NFEL0               400
nvlink_flit_crc_error_count_l1                                                NFEL1               401
nvlink_flit_crc_error_count_l2                                                NFEL2               402
nvlink_flit_crc_error_count_l3                                                NFEL3               403
nvlink_flit_crc_error_count_l4                                                NFEL4               404
nvlink_flit_crc_error_count_l5                                                NFEL5               405
nvlink_flit_crc_error_count_l12                                               NFEL12              406
nvlink_flit_crc_error_count_l13                                               NFEL13              407
nvlink_flit_crc_error_count_l14                                               NFEL14              408
nvlink_flit_crc_error_count_total                                             NFELT               409
nvlink_data_crc_error_count_l0                                                NDEL0               410
nvlink_data_crc_error_count_l1                                                NDEL1               411
nvlink_data_crc_error_count_l2                                                NDEL2               412
nvlink_data_crc_error_count_l3                                                NDEL3               413
nvlink_data_crc_error_count_l4                                                NDEL4               414
nvlink_data_crc_error_count_l5                                                NDEL5               415
nvlink_data_crc_error_count_l12                                               NDEL12              416
nvlink_data_crc_error_count_l13                                               NDEL13              417
nvlink_data_crc_error_count_l14                                               NDEL14              418
nvlink_data_crc_error_count_total                                             NDELT               419
nvlink_replay_error_count_l0                                                  NREL0               420
nvlink_replay_error_count_l1                                                  NREL1               421
nvlink_replay_error_count_l2                                                  NREL2               422
nvlink_replay_error_count_l3                                                  NREL3               423
nvlink_replay_error_count_l4                                                  NREL4               424
nvlink_replay_error_count_l5                                                  NREL5               425
nvlink_replay_error_count_l12                                                 NREL12              426
nvlink_replay_error_count_l13                                                 NREL13              427
nvlink_replay_error_count_l14                                                 NREL14              428
nvlink_replay_error_count_total                                               NRELT               429
nvlink_recovery_error_count_l0                                                NRCL0               430
nvlink_recovery_error_count_l1                                                NRCL1               431
nvlink_recovery_error_count_l2                                                NRCL2               432
nvlink_recovery_error_count_l3                                                NRCL3               433
nvlink_recovery_error_count_l4                                                NRCL4               434
nvlink_recovery_error_count_l5                                                NRCL5               435
nvlink_recovery_error_count_l12                                               NRCL12              436
nvlink_recovery_error_count_l13                                               NRCL13              437
nvlink_recovery_error_count_l14                                               NRCL14              438
nvlink_recovery_error_count_total                                             NRCLT               439
nvlink_bandwidth_l0                                                           NBWL0               440
nvlink_bandwidth_l1                                                           NBWL1               441
nvlink_bandwidth_l2                                                           NBWL2               442
nvlink_bandwidth_l3                                                           NBWL3               443
nvlink_bandwidth_l4                                                           NBWL4               444
nvlink_bandwidth_l5                                                           NBWL5               445
nvlink_bandwidth_l12                                                          NBWL12              446
nvlink_bandwidth_l13                                                          NBWL13              447
nvlink_bandwidth_l14                                                          NBWL14              448
nvlink_bandwidth_total                                                        NBWLT               449
gpu_nvlink_errors                                                             GNVERR              450
nvlink_flit_crc_error_count_l6                                                NFEL6               451
nvlink_flit_crc_error_count_l7                                                NFEL7               452
nvlink_flit_crc_error_count_l8                                                NFEL8               453
nvlink_flit_crc_error_count_l9                                                NFEL9               454
nvlink_flit_crc_error_count_l10                                               NFEL10              455
nvlink_flit_crc_error_count_l11                                               NFEL11              456
nvlink_data_crc_error_count_l6                                                NDEL6               457
nvlink_data_crc_error_count_l7                                                NDEL7               458
nvlink_data_crc_error_count_l8                                                NDEL8               459
nvlink_data_crc_error_count_l9                                                NDEL9               460
nvlink_data_crc_error_count_l10                                               NDEL10              461
nvlink_data_crc_error_count_l11                                               NDEL11              462
nvlink_replay_error_count_l6                                                  NREL6               463
nvlink_replay_error_count_l7                                                  NREL7               464
nvlink_replay_error_count_l8                                                  NREL8               465
nvlink_replay_error_count_l9                                                  NREL9               466
nvlink_replay_error_count_l10                                                 NREL10              467
nvlink_replay_error_count_l11                                                 NREL11              468
nvlink_recovery_error_count_l6                                                NRCL6               469
nvlink_recovery_error_count_l7                                                NRCL7               470
nvlink_recovery_error_count_l8                                                NRCL8               471
nvlink_recovery_error_count_l9                                                NRCL9               472
nvlink_recovery_error_count_l10                                               NRCL10              473
nvlink_recovery_error_count_l11                                               NRCL11              474
nvlink_bandwidth_l6                                                           NBWL6               475
nvlink_bandwidth_l7                                                           NBWL7               476
nvlink_bandwidth_l8                                                           NBWL8               477
nvlink_bandwidth_l9                                                           NBWL9               478
nvlink_bandwidth_l10                                                          NBWL10              479
nvlink_bandwidth_l11                                                          NBWL11              480
nvlink_flit_crc_error_count_l15                                               NFEL15              481
nvlink_flit_crc_error_count_l16                                               NFEL16              482
nvlink_flit_crc_error_count_l17                                               NFEL17              483
nvlink_data_crc_error_count_l15                                               NDEL15              484
nvlink_data_crc_error_count_l16                                               NDEL16              485
nvlink_data_crc_error_count_l17                                               NDEL17              486
nvlink_replay_error_count_l15                                                 NREL15              487
nvlink_replay_error_count_l16                                                 NREL16              488
nvlink_replay_error_count_l17                                                 NREL17              489
nvlink_recovery_error_count_l15                                               NRCL15              491
nvlink_recovery_error_count_l16                                               NRCL16              492
nvlink_recovery_error_count_l17                                               NRCL17              493
nvlink_bandwidth_l15                                                          NBWL15              494
nvlink_bandwidth_l16                                                          NBWL16              495
nvlink_bandwidth_l17                                                          NBWL17              496
virtualization_mode                                                           VMODE               500
supported_type_info                                                           SPINF               501
creatable_vgpu_type_ids                                                       CGPID               502
active_vgpu_instance_ids                                                      VGIID               503
vgpu_instance_utilizations                                                    VIUTL               504
vgpu_instance_per_process_utilization                                         VIPPU               505
enc_stats                                                                     ENSTA               506
fbc_stats                                                                     FBCSTA              507
fbc_sessions_info                                                             FBCINF              508
vgpu_type_ids                                                                 VTID                509
vgpu_type_info                                                                VTPINF              510
vgpu_type_name                                                                VTPNM               511
vgpu_type_class                                                               VTPCLS              512
vgpu_type_license                                                             VTPLC               513
vgpu_instance_vm_id                                                           VVMID               520
vgpu_instance_vm_name                                                         VMNAM               521
vgpu_instance_type                                                            VITYP               522
vgpu_instance_uuid                                                            VUUID               523
vgpu_instance_driver_version                                                  VDVER               524
vgpu_instance_memory_usage                                                    VMUSG               525
vgpu_instance_license_status                                                  VLCST               526
vgpu_instance_frame_rate_limit                                                VFLIM               527
vgpu_instance_enc_stats                                                       VSTAT               528
vgpu_instance_enc_sessions_info                                               VSINF               529
vgpu_instance_fbc_stats                                                       VFSTAT              530
vgpu_instance_fbc_sessions_info                                               VFINF               531
vgpu_instance_license_state                                                   VLCIST              532
vgpu_instance_pci_id                                                          VPCIID              533
vgpu_instance_gpu_instance_id                                                 VGII                534
nvswitch_link_bandwidth_tx                                                    SWLNKTX             780
nvswitch_link_bandwidth_rx                                                    SWLNKRX             781
nvswitch_link_fatal_errors                                                    SWLNKFE             782
nvswitch_link_non_fatal_errors                                                SWLNKNF             783
nvswitch_link_replay_errors                                                   SWLNKRP             784
nvswitch_link_recovery_errors                                                 SWLNKRC             785
nvswitch_link_flit_errors                                                     SWLNKFL             786
nvswitch_link_crc_errors                                                      SWLNKCR             787
nvswitch_link_ecc_errors                                                      SWLNKEC             788
nvswitch_link_latency_low_vc0                                                 SWVCLL0             789
nvswitch_link_latency_low_vc1                                                 SWVCLL1             790
nvswitch_link_latency_low_vc2                                                 SWVCLL2             791
nvswitch_link_latency_low_vc                                                  SWVCLL3             792
nvswitch_link_latency_medium_vc0                                              SWVCLM0             793
nvswitch_link_latency_medium_vc1                                              SWVCLM1             794
nvswitch_link_latency_medium_vc2                                              SWVCLM2             795
nvswitch_link_latency_medium_vc3                                              SWVCLM3             796
nvswitch_link_latency_high_vc0                                                SWVCLH0             797
nvswitch_link_latency_high_vc1                                                SWVCLH1             798
nvswitch_link_latency_high_vc2                                                SWVCLH2             799
nvswitch_link_latency_high_vc3                                                SWVCLH3             800
nvswitch_link_latency_panic_vc0                                               SWVCLP0             801
nvswitch_link_latency_panic_vc1                                               SWVCLP1             802
nvswitch_link_latency_panic_vc2                                               SWVCLP2             803
nvswitch_link_latency_panic_vc3                                               SWVCLP3             804
nvswitch_link_latency_count_vc0                                               SWVCLC0             805
nvswitch_link_latency_count_vc1                                               SWVCLC1             806
nvswitch_link_latency_count_vc2                                               SWVCLC2             807
nvswitch_link_latency_count_vc3                                               SWVCLC3             808
nvswitch_link_crc_errors_lane0                                                SWLACR0             809
nvswitch_link_crc_errors_lane1                                                SWLACR1             810
nvswitch_link_crc_errors_lane2                                                SWLACR2             811
nvswitch_link_crc_errors_lane3                                                SWLACR3             812
nvswitch_link_ecc_errors_lane0                                                SWLAEC0             813
nvswitch_link_ecc_errors_lane1                                                SWLAEC1             814
nvswitch_link_ecc_errors_lane2                                                SWLAEC2             815
nvswitch_link_ecc_errors_lane3                                                SWLAEC3             816
nvswitch_fatal_error                                                          SEN00               856
nvswitch_non_fatal_error                                                      SEN01               857
nvswitch_current_temperature                                                  TMP01               858
nvswitch_slowdown_temperature                                                 TMP02               859
nvswitch_shutdown_temperature                                                 TMP03               860
nvswitch_bandwidth_tx                                                         SWTX                861
nvswitch_bandwidth_rx                                                         SWRX                862
nvswitch_physical_id                                                          SWPHID              863
nvswitch_reset_required                                                       SWFRMVER            864
nvlink_id                                                                     LNKID               865
nvswitch_pcie_dom                                                             SWPCIEDOM           866
nvswitch_pcie_bus                                                             SWPCIEBUS           867
nvswitch_pcie_dev                                                             SWPCIEDEV           868
nvswitch_pcie_fun                                                             SWPCIEFUN           869
nvswitch_nvlink_status                                                        SWNVLNKST           870
nvswitch_nvlink_dev_type                                                      SWNVLNKDT           871
link_pcie_remote_dom                                                          LNKDOM              872
link_pcie_remote_bus                                                          LNKBUS              873
link_pcie_remote_dev                                                          LNKDEV              874
link_pcie_remote_func                                                         LNKFNC              875
link_dev_link_id                                                              SWNVLNKID           876
link_dev_link_sid                                                             SWNVLNSID           877
link_dev_link_uuid                                                            SWNVLNUID           878
gr_engine_active                                                              GRACT               1001
sm_active                                                                     SMACT               1002
sm_occupancy                                                                  SMOCC               1003
tensor_active                                                                 TENSO               1004
dram_active                                                                   DRAMA               1005
fp64_active                                                                   FP64A               1006
fp32_active                                                                   FP32A               1007
fp16_active                                                                   FP16A               1008
pcie_tx_bytes                                                                 PCITX               1009
pcie_rx_bytes                                                                 PCIRX               1010
nvlink_tx_bytes                                                               NVLTX               1011
nvlink_rx_bytes                                                               NVLRX               1012
tensor_imma_active                                                            TIMMA               1013
tensor_hmma_active                                                            THMMA               1014
tensor_dfma_active                                                            TDFMA               1015
integer_active                                                                INTAC               1016
nvdec0_active                                                                 NVDEC0              1017
nvdec1_active                                                                 NVDEC1              1018
nvdec2_active                                                                 NVDEC2              1019
nvdec3_active                                                                 NVDEC3              1020
nvdec4_active                                                                 NVDEC4              1021
nvdec5_active                                                                 NVDEC5              1022
nvdec6_active                                                                 NVDEC6              1023
nvdec7_active                                                                 NVDEC7              1024
nvjpg0_active                                                                 NVJPG0              1025
nvjpg1_active                                                                 NVJPG1              1026
nvjpg2_active                                                                 NVJPG2              1027
nvjpg3_active                                                                 NVJPG3              1028
nvjpg4_active                                                                 NVJPG4              1029
nvjpg5_active                                                                 NVJPG5              1030
nvjpg6_active                                                                 NVJPG6              1031
nvjpg7_active                                                                 NVJPG7              1032
nvofa0_active                                                                 NVOFA0              1033
nvlink_l0_tx_bytes                                                            NVL0T               1040
nvlink_l0_rx_bytes                                                            NVL0R               1041
nvlink_l1_tx_bytes                                                            NVL1T               1042
nvlink_l1_rx_bytes                                                            NVL1R               1043
nvlink_l2_tx_bytes                                                            NVL2T               1044
nvlink_l2_rx_bytes                                                            NVL2R               1045
nvlink_l3_tx_bytes                                                            NVL3T               1046
nvlink_l3_rx_bytes                                                            NVL3R               1047
nvlink_l4_tx_bytes                                                            NVL4T               1048
nvlink_l4_rx_bytes                                                            NVL4R               1049
nvlink_l5_tx_bytes                                                            NVL5T               1050
nvlink_l5_rx_bytes                                                            NVL5R               1051
nvlink_l6_tx_bytes                                                            NVL6T               1052
nvlink_l6_rx_bytes                                                            NVL6R               1053
nvlink_l7_tx_bytes                                                            NVL7T               1054
nvlink_l7_rx_bytes                                                            NVL7R               1055
nvlink_l8_tx_bytes                                                            NVL8T               1056
nvlink_l8_rx_bytes                                                            NVL8R               1057
nvlink_l9_tx_bytes                                                            NVL9T               1058
nvlink_l9_rx_bytes                                                            NVL9R               1059
nvlink_l10_tx_bytes                                                           NVL10T              1060
nvlink_l10_rx_bytes                                                           NVL10R              1061
nvlink_l11_tx_bytes                                                           NVL11T              1062
nvlink_l11_rx_bytes                                                           NVL11R              1063
nvlink_l12_tx_bytes                                                           NVL12T              1064
nvlink_l12_rx_bytes                                                           NVL12R              1065
nvlink_l13_tx_bytes                                                           NVL13T              1066
nvlink_l13_rx_bytes                                                           NVL13R              1067
nvlink_l14_tx_bytes                                                           NVL14T              1068
nvlink_l14_rx_bytes                                                           NVL14R              1069
nvlink_l15_tx_bytes                                                           NVL15T              1070
nvlink_l15_rx_bytes                                                           NVL15R              1071
nvlink_l16_tx_bytes                                                           NVL16T              1072
nvlink_l16_rx_bytes                                                           NVL16R              1073
nvlink_l17_tx_bytes                                                           NVL17T              1074
nvlink_l17_rx_bytes                                                           NVL17R              1075
```

```
dcgmi dmon  -i 0  -e 1011,1012,1009,1010 -c 5
#Entity   NVLTX                       NVLRX                       PCITX                       PCIRX        
ID                                                                                                         
GPU 0     N/A                         N/A                         N/A                         N/A          
GPU 0     N/A                         N/A                         N/A                         N/A          
GPU 0     0                           0                           498948                      1555         
GPU 0     0                           0                           449138                      2074         
GPU 0     0                           0                           548740                      1555  
```


## dcgmi nvlink
用于获取系统中 GPU 和 NvSwitch 的 NvLink 链接状态或错误计数

```
> dcgmi nvlink --help

 nvlink -- Used to get NvLink link status or error counts for GPUs and
 NvSwitches in the system

 NVLINK Error description
 =========================
 CRC FLIT Error => Data link receive flow control digit CRC error.
 CRC Data Error => Data link receive data CRC error.
 Replay Error   => Data link transmit replay error.
 Recovery Error => Data link transmit recovery error.

Usage: dcgmi nvlink
   dcgmi nvlink --host <IP/FQDN> -g <gpuId> -e -j
   dcgmi nvlink --host <IP/FQDN> -s

Flags:
      --host       IP/FQDN    Connects to specified IP or fully-qualified domain
                               name. To connect to a host engine that was
                               started with -d (unix socket), prefix the unix
                               socket filename with 'unix://'. [default =
                               localhost]
  -e  --errors                Print NvLink errors for a given gpuId (-g).
  -s  --link-status           Print NvLink link status for all GPUs and
                               NvSwitches in the system.
  -h  --help                  Displays usage information and exits.
  -g  --gpuid      gpuId      The GPU ID to query. Required for -e
  -j  --json                  Print the output in a json format（json格式输出）
  --  --ignore_rest           Ignores the rest of the labeled arguments
                               following this flag.

NVIDIA Datacenter GPU Management Interface
```

json格式输出：

```
> dcgmi nvlink -g 0 -e -j
{
        "body" :
        {
                "Link 0" :
                {
                        "children" :
                        {
                                "CRC Data Error" :
                                {
                                        "value" : "0"
                                },
                                "CRC FLIT Error" :
                                {
                                        "value" : "0"
                                },
                                "Recovery Error" :
                                {
                                        "value" : "0"
                                },
                                "Replay Error" :
                                {
                                        "value" : "0"
                                }
                        }
                },
                "Link 1" :
                {
                        "children" :
                        {
                                "CRC Data Error" :
                                {
                                        "value" : "0"
                                },
                                "CRC FLIT Error" :
                                {
                                        "value" : "0"
                                },
                                "Recovery Error" :
                                {
                                        "value" : "0"
                                },
                                "Replay Error" :
                                {
                                        "value" : "0"
                                }
                        }
                },
                "Link 2" :
                {
                        "children" :
                        {
                                "CRC Data Error" :
                                {
                                        "value" : "0"
                                },
                                "CRC FLIT Error" :
                                {
                                        "value" : "0"
                                },
                                "Recovery Error" :
                                {
                                        "value" : "0"
                                },
                                "Replay Error" :
                                {
                                        "value" : "0"
                                }
                        }
                },
                "Link 3" :
                {
                        "children" :
                        {
                                "CRC Data Error" :
                                {
                                        "value" : "0"
                                },
                                "CRC FLIT Error" :
                                {
                                        "value" : "0"
                                },
                                "Recovery Error" :
                                {
                                        "value" : "0"
                                },
                                "Replay Error" :
                                {
                                        "value" : "0"
                                }
                        }
                },
                "Link 4" :
                {
                        "children" :
                        {
                                "CRC Data Error" :
                                {
                                        "value" : "0"
                                },
                                "CRC FLIT Error" :
                                {
                                        "value" : "0"
                                },
                                "Recovery Error" :
                                {
                                        "value" : "0"
                                },
                                "Replay Error" :
                                {
                                        "value" : "0"
                                }
                        }
                },
                "Link 5" :
                {
                        "children" :
                        {
                                "CRC Data Error" :
                                {
                                        "value" : "0"
                                },
                                "CRC FLIT Error" :
                                {
                                        "value" : "0"
                                },
                                "Recovery Error" :
                                {
                                        "value" : "0"
                                },
                                "Replay Error" :
                                {
                                        "value" : "0"
                                }
                        }
                },
                "Link 6" :
                {
                        "children" :
                        {
                                "CRC Data Error" :
                                {
                                        "value" : "0"
                                },
                                "CRC FLIT Error" :
                                {
                                        "value" : "0"
                                },
                                "Recovery Error" :
                                {
                                        "value" : "0"
                                },
                                "Replay Error" :
                                {
                                        "value" : "0"
                                }
                        }
                },
                "Link 7" :
                {
                        "children" :
                        {
                                "CRC Data Error" :
                                {
                                        "value" : "0"
                                },
                                "CRC FLIT Error" :
                                {
                                        "value" : "0"
                                },
                                "Recovery Error" :
                                {
                                        "value" : "0"
                                },
                                "Replay Error" :
                                {
                                        "value" : "0"
                                }
                        }
                }
        },
        "header" :
        [
                "NVLINK Error Counts",
                "GPU 0"
        ]
}

```



```
dcgmi nvlink -s
+----------------------+
|  NvLink Link Status  |
+----------------------+
GPUs:
    gpuId 0:
        U U U U U U U U _ _ _ _ _ _ _ _ _ _
    gpuId 1:
        U U U U U U U U _ _ _ _ _ _ _ _ _ _
    gpuId 2:
        U U U U U U U U _ _ _ _ _ _ _ _ _ _
    gpuId 3:
        U U U U U U U U _ _ _ _ _ _ _ _ _ _
    gpuId 4:
        U U U U U U U U _ _ _ _ _ _ _ _ _ _
    gpuId 5:
        U U U U U U U U _ _ _ _ _ _ _ _ _ _
    gpuId 6:
        U U U U U U U U _ _ _ _ _ _ _ _ _ _
    gpuId 7:
        U U U U U U U U _ _ _ _ _ _ _ _ _ _
NvSwitches:
    No NvSwitches found.

Key: Up=U, Down=D, Disabled=X, Not Supported=_
```

```
> dcgmi nvlink -g 1 -e
+-----------------------------+------------------------------------------------+
| NVLINK Error Counts                                                          |
| GPU 1                                                                        |
+=============================+================================================+
| Link 0                      |                                                |
| -> CRC FLIT Error           | 0                                              |
| -> CRC Data Error           | 0                                              |
| -> Replay Error             | 0                                              |
| -> Recovery Error           | 0                                              |
| Link 1                      |                                                |
| -> CRC FLIT Error           | 0                                              |
| -> CRC Data Error           | 0                                              |
| -> Replay Error             | 0                                              |
| -> Recovery Error           | 0                                              |
| Link 2                      |                                                |
| -> CRC FLIT Error           | 0                                              |
| -> CRC Data Error           | 0                                              |
| -> Replay Error             | 0                                              |
| -> Recovery Error           | 0                                              |
| Link 3                      |                                                |
| -> CRC FLIT Error           | 0                                              |
| -> CRC Data Error           | 0                                              |
| -> Replay Error             | 0                                              |
| -> Recovery Error           | 0                                              |
| Link 4                      |                                                |
| -> CRC FLIT Error           | 0                                              |
| -> CRC Data Error           | 0                                              |
| -> Replay Error             | 0                                              |
| -> Recovery Error           | 0                                              |
| Link 5                      |                                                |
| -> CRC FLIT Error           | 0                                              |
| -> CRC Data Error           | 0                                              |
| -> Replay Error             | 0                                              |
| -> Recovery Error           | 0                                              |
| Link 6                      |                                                |
| -> CRC FLIT Error           | 0                                              |
| -> CRC Data Error           | 0                                              |
| -> Replay Error             | 0                                              |
| -> Recovery Error           | 0                                              |
| Link 7                      |                                                |
| -> CRC FLIT Error           | 0                                              |
| -> CRC Data Error           | 0                                              |
| -> Replay Error             | 0                                              |
| -> Recovery Error           | 0                                              |
+-----------------------------+------------------------------------------------+
```


## 指标
支持以下新的设备级分析指标。 列出了定义和相应的 DCGM 字段 ID。 

默认情况下，DCGM 以 1Hz（每 1000毫秒(ms)）的采样率提供指标。 用户可以以任何可配置的频率（最小为 100 毫秒(ms)）从 DCGM 查询指标（例如：dcgmi dmon -d）。


以下是设备水平（level）的GPU指标
| Metric | Definition | DCGM Field Name (DCGM_FI_*) and ID |
| --- | --- | --- |
| Graphics Engine Activity | The fraction of time any portion of the graphics or compute engines were active. The graphics engine is active if a graphics/compute context is bound and the graphics/compute pipe is busy. The value represents an average over a time interval and is not an instantaneous value. | PROF_GR_ENGINE_ACTIVE (ID: 1001) |
| SM Activity | The fraction of time at least one warp was active on a multiprocessor, averaged over all multiprocessors. Note that “active” does not necessarily mean a warp is actively computing. For instance, warps waiting on memory requests are considered active. The value represents an average over a time interval and is not an instantaneous value. A value of 0.8 or greater is necessary, but not sufficient, for effective use of the GPU. A value less than 0.5 likely indicates ineffective GPU usage.Given a simplified GPU architectural view, if a GPU has N SMs then a kernel using N blocks that runs over the entire time interval will correspond to an activity of 1 (100%). A kernel using N/5 blocks that runs over the entire time interval will correspond to an activity of 0.2 (20%). A kernel using N blocks that runs over one fifth of the time interval, with the SMs otherwise idle, will also have an activity of 0.2 (20%). The value is insensitive to the number of threads per block (see `DCGM_FI_PROF_SM_OCCUPANCY`). | PROF_SM_ACTIVE (ID: 1002) |
| SM Occupancy | The fraction of resident warps on a multiprocessor, relative to the maximum number of concurrent warps supported on a multiprocessor. The value represents an average over a time interval and is not an instantaneous value. Higher occupancy does not necessarily indicate better GPU usage. For GPU memory bandwidth limited workloads (see `DCGM_FI_PROF_DRAM_ACTIVE`), higher occupancy is indicative of more effective GPU usage. However if the workload is compute limited (i.e. not GPU memory bandwidth or latency limited), then higher occupancy does not necessarily correlate with more effective GPU usage.Calculating occupancy is not simple and depends on factors such as the GPU properties, the number of threads per block, registers per thread, and shared memory per block. Use the [CUDA Occupancy Calculator](https://docs.nvidia.com/cuda/cuda-occupancy-calculator/index.html) to explore various occupancy scenarios. | PROF_SM_OCCUPANCY (ID: 1003) |
| Tensor Activity | The fraction of cycles the tensor (HMMA / IMMA) pipe was active. The value represents an average over a time interval and is not an instantaneous value. Higher values indicate higher utilization of the Tensor Cores. An activity of 1 (100%) is equivalent to issuing a tensor instruction every other cycle for the entire time interval. An activity of 0.2 (20%) could indicate 20% of the SMs are at 100% utilization over the entire time period, 100% of the SMs are at 20% utilization over the entire time period, 100% of the SMs are at 100% utilization for 20% of the time period, or any combination in between (see `DCGM_FI_PROF_SM_ACTIVE` to help disambiguate these possibilities). | PROF_PIPE_TENSOR_ACTIVE (ID: 1004) |
| FP64 Engine Activity | The fraction of cycles the FP64 (double precision) pipe was active. The value represents an average over a time interval and is not an instantaneous value. Higher values indicate higher utilization of the FP64 cores. An activity of 1 (100%) is equivalent to a FP64 instruction on [every SM every fourth cycle](https://docs.nvidia.com/cuda/volta-tuning-guide/index.html#sm-scheduling) on Volta over the entire time interval. An activity of 0.2 (20%) could indicate 20% of the SMs are at 100% utilization over the entire time period, 100% of the SMs are at 20% utilization over the entire time period, 100% of the SMs are at 100% utilization for 20% of the time period, or any combination in between (see DCGM_FI_PROF_SM_ACTIVE to help disambiguate these possibilities). | PROF_PIPE_FP64_ACTIVE (ID: 1006) |
| FP32 Engine Activity | The fraction of cycles the FMA (FP32 (single precision), and integer) pipe was active. The value represents an average over a time interval and is not an instantaneous value. Higher values indicate higher utilization of the FP32 cores. An activity of 1 (100%) is equivalent to a FP32 instruction every other cycle over the entire time interval. An activity of 0.2 (20%) could indicate 20% of the SMs are at 100% utilization over the entire time period, 100% of the SMs are at 20% utilization over the entire time period, 100% of the SMs are at 100% utilization for 20% of the time period, or any combination in between (see `DCGM_FI_PROF_SM_ACTIVE` to help disambiguate these possibilities). | PROF_PIPE_FP32_ACTIVE (ID: 1007) |
| FP16 Engine Activity | The fraction of cycles the FP16 (half precision) pipe was active. The value represents an average over a time interval and is not an instantaneous value. Higher values indicate higher utilization of the FP16 cores. An activity of 1 (100%) is equivalent to a FP16 instruction every other cycle over the entire time interval. An activity of 0.2 (20%) could indicate 20% of the SMs are at 100% utilization over the entire time period, 100% of the SMs are at 20% utilization over the entire time period, 100% of the SMs are at 100% utilization for 20% of the time period, or any combination in between (see `DCGM_FI_PROF_SM_ACTIVE` to help disambiguate these possibilities). | PROF_PIPE_FP16_ACTIVE (ID: 1008) |
| Memory BW Utilization | The fraction of cycles where data was sent to or received from device memory. The value represents an average over a time interval and is not an instantaneous value. Higher values indicate higher utilization of device memory. An activity of 1 (100%) is equivalent to a DRAM instruction every cycle over the entire time interval (in practice a peak of ~0.8 (80%) is the maximum achievable). An activity of 0.2 (20%) indicates that 20% of the cycles are reading from or writing to device memory over the time interval. | PROF_DRAM_ACTIVE (ID: 1005) |
| NVLink Bandwidth | 通过 NVLink 传输/接收的数据速率（不包括协议头(protocol headers)），以字节/秒为单位。 该值表示一段时间间隔内的平均值，而不是瞬时值。 该速率是时间间隔内的平均值。 例如，如果 1 秒内传输 1 GB 数据，则无论数据以恒定速率还是突发传输，速率均为 1 GB/s。 NVLink Gen2 的理论最大带宽为每个链路每个方向 25 GB/s。 | PROF_NVLINK_TX_BYTES (1011) and PROF_NVLINK_RX_BYTES (1012) |
| PCIe Bandwidth | 通过 PCIe 总线传输/接收的数据速率，包括协议标头和数据有效负载，以字节/秒为单位。 该值表示一段时间间隔内的平均值，而不是瞬时值。 该速率是时间间隔内的平均值。 例如，如果 1 秒内传输 1 GB 数据，则无论数据以恒定速率还是突发传输，速率均为 1 GB/s。 理论最大 PCIe Gen3 带宽为每通道 985 MB/s。 | PROF_PCIE_[T\|R]X_BYTES (ID: 1009 (TX); 1010 (RX)) |


```
> dcgmi dmon -i 0,1,2,3  -e 1011,1012
#Entity   NVLTX                       NVLRX
ID
GPU 3     19694075554                 19687914629
GPU 2     19777203418                 19819177524
GPU 1     19699841766                 22070216956
GPU 0     20779220484                 21900091841
GPU 3     12945588302                 12953884356
GPU 2     12558214740                 12560935679
GPU 1     13059621728                 10651057317
GPU 0     11576689215                 9600734242
GPU 3     11155319776                 11155326544
GPU 2     11155466819                 11155466298
GPU 1     11040517157                 12515409691
GPU 0     11592513041                 13925722805
GPU 3     1286216247                  1217881887
GPU 2     928524939                   860186978
GPU 1     1506174212                  50051
GPU 0     31802                       911367981
GPU 3     0                           0
GPU 2     0                           0
GPU 1     0                           0
GPU 0     0                           0
GPU 3     23309642310                 23377912493
GPU 2     23176458503                 23176459024
GPU 1     23447369511                 23507663607
GPU 0     23508249062                 23174848479
...


> dcgmi dmon  -e 1011,1012
#Entity   NVLTX                       NVLRX
ID
GPU 7     30570603980                 30638829242
GPU 6     30567094640                 30635348592
GPU 5     30628398352                 33089365519
GPU 4     33098848601                 36119516306
GPU 3     33750138990                 33825970205
GPU 2     31743752465                 31812022474
GPU 1     34030055050                 34098309807
GPU 0     32873620375                 29632298747
GPU 7     24371477520                 24370431480
GPU 6     24443717565                 24443653033
GPU 5     24450523113                 23160485855
GPU 4     23167734167                 23167708130
GPU 3     25744193567                 25744198774
GPU 2     25027562441                 25027562441
GPU 1     24099003605                 24099024433
GPU 0     24669591596                 24669655619
...



> dcgmi dmon  -e 1011,1012,1009,1010
GPU 7     9012312241                  9010146921                  3658375705                  1470950385
GPU 6     38735656050(36.07GB/s)      38739460219(36.08GB/s)      3715470394(3.46GB/s)        069653476(0.996GB/s)
GPU 5     37117100494                 37114692577                 3684018382                  1195478083
GPU 4     15832363949                 30483540427                 3617204053                  1084584434
GPU 3     11415357717                 11415357717                 3762838708                  3470626438
GPU 2     32126737331                 32124608391                 3860671178                  1817597475
GPU 1     37055654032                 37055676937                 3666866771                  1201785740
GPU 0     27827206810                 27762665999                 N/A                         1146900782
GPU 7     37300245001                 37302405771                 3843250109                  4599309358
GPU 6     14877616163                 14939829270                 3919059148                  4513192032
GPU 5     17320548737                 17382778744                 3889129122                  4641743864
GPU 4     30487341762                 16373502117                 3933037804                  6115081312
GPU 3     34918736873                 34918742079                 3910245112                  1761934955
GPU 2     16547291813                 19112960872                 2761505306                  3060783203
GPU 1     18380875930                 18390091637                 148870522                   2103742852
GPU 0     19407501485                 15881929591                 3711808007                  1055934784
```

统计6000次，将结果保存到文件。
```
dcgmi dmon  -e 1011,1012,1009,1010 -c 6000 >> bandwitch.txt
```







