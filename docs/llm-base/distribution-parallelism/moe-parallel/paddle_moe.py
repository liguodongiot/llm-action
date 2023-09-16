
# 导入需要的包
import paddle
from paddle.nn import Layer, LayerList, Linear, Dropout
from paddle.incubate.distributed.models.moe import MoELayer
from paddle.distributed.collective import Group
from paddle.distributed import fleet
import numpy as np


num_experts = 8
d_model = 512
d_hidden = 2048


# 封装专家层
class ExpertLayer(Layer):
    def __init__(self, d_model, d_hidden, name=None):
        super().__init__()
        self.htoh4 = Linear(d_model, d_hidden)
        self.h4toh = Linear(d_hidden, d_model)

    def forward(self, x):
        x = self.htoh4(x)
        x = self.h4toh(x)
        return x


# 初始化分布式环境，并构建 expert 通信组 moe_group
fleet.init(is_collective=True)
moe_group = paddle.distributed.new_group(list(range(fleet.worker_num())))


gate_config = {
    "type": "gshard",
    "top_k": 2,
}


experts_list = LayerList()
for expi in range(num_experts):
    exp_layer = ExpertLayer(d_model, d_hidden)
    experts_list.append(exp_layer)


# 调用 MoELayer API 封装并创建出 MoE 模型
class Model(Layer):
	def __init__(self, d_model, d_hidden, name=None):
	    super().__init__()
	    self.linear1 = Linear(d_model, d_model)
	    self.moe_layer = MoELayer(d_model = d_model,
	                            experts=experts_list,
	                            gate=gate_config,
	                            moe_group=moe_group,
	                            recompute_interval=0)

	    self.linear2 = Linear(d_model, d_model)
	    self.dropout = Dropout(p=0.1)

	def forward(self, x):
	    x = self.linear1(x)
	    x = self.moe_layer(x)
	    x = self.linear2(x)
	    x = self.dropout(x)
	    return x


model = Model(d_model, d_hidden)
optim = paddle.optimizer.SGD(parameters=model.parameters())

# 创建数据集，开始训练
for step in range(1, 100):
    x = paddle.rand([4, 256, d_model])

    y = model(x)
    loss = y.mean()
    loss.backward()
    optim.step()

    optim.clear_grad()

    print("=== step : {}, loss : {}".format(step, loss.numpy()))


