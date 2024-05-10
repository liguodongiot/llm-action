




- https://gitee.com/mindspore/mindspore



```
import numpy as np
import mindspore.context as context
from mindspore import Tensor
from mindspore.ops import functional as F

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

x = Tensor(np.ones([1,3,3,4]).astype(np.float32))
y = Tensor(np.ones([1,3,3,4]).astype(np.float32))
print(F.tensor_add(x, y))

```




https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.3.0rc1/MindSpore/unified/aarch64/mindspore-2.3.0rc1-cp39-cp39-linux_aarch64.whl




