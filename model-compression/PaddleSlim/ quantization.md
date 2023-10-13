






## Shift

```
from paddleslim.quant.advanced import Shift, EMASampler

model = LLM()
model_config = {}
shift = Shift(model, model_config, sample_function=EMASampler())
for data in dataloader():
    model(data)
    shift.step += 1
shift.update_weight()
```




## Smooth


```
from paddleslim.quant.advanced import Smooth，MultiStepSampler

model = LLM()
model_config = {}
smooth = Smooth(model, model_config, sample_function=MultiStepSampler())
for data in dataloader():
    model(data)
    smooth.step += 1
smooth.update_weight()
```


## PieceWiseSearch

```
from paddleslim.quant.advanced import Smooth, MultiStepSampler, PieceWiseSearch, mse_loss

search_func =PieceWiseSearch(
                k_piece=3,
                bits_length=8,
                search_piece=False,
                search_alpha_min=0.2,
                search_alpha_max=0.8,
                search_scale_min=1.,
                search_scale_max=5.,
                weight_quant_method='abs_max_channel_wise',
                act_quant_method='abs_max',
                loss_function=mse_loss
            )
model = LLM()
model_config = {}
smooth = Smooth(model, model_config, sample_function=MultiStepSampler(), search_function=search_func)
for data in dataloader():
    model(data)
    smooth.step += 1
smooth.update_weight()

```


## GPTQ


```
from paddleslim.quant.advanced import GPTQ

model = LLM()
for cur_name, cur_layer in model.named_sublayers():
    if type(cur_layer) == paddle.nn.Linear:
        gptq_layer = GPTQ(cur_layer)
        # sample data
        for data in dataloader():
            model(data)
        # quant weight
        gptq_layer.fasterquant(act_order=True)
```


## LayerWiseQuantError


```
from paddleslim.quant.advanced import LayerWiseQuantError

model = LLM()
for cur_name, cur_layer in model.named_sublayers():
    if type(cur_layer) == paddle.nn.Linear:
        gptq_layer = LayerWiseQuantError(cur_layer)

for data in dataloader():
    model(data)

for cur_name, cur_layer in model.named_sublayers():
    if type(cur_layer) == LayerWiseQuantError:
        print(cur_name, cur_layer.losses.mean())
```











