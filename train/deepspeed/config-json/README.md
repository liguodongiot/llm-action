- https://www.deepspeed.ai/docs/config-json/



## bf16

```
"bf16": {
   "enabled": true
 }
```



## fp16
```
"fp16": {
    "enabled": true,
    "auto_cast": false,
    "loss_scale": 0,
    "initial_scale_power": 16,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "consecutive_hysteresis": false,
    "min_loss_scale": 1
}
```
