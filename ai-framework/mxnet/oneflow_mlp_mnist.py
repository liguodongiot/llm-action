import oneflow as flow
import oneflow.nn as nn
from flowvision import transforms
from flowvision import datasets


BATCH_SIZE=64

DEVICE = "cpu" # "cuda" if flow.cuda.is_available() else "cpu"

print("Using {} device".format(DEVICE))





training_data = datasets.MNIST(
    root="/workspace/data/oneflow",
    train=True,
    transform=transforms.ToTensor(),
    download=True,
    source_url="https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/mnist/MNIST/",

)
test_data = datasets.MNIST(
    root="/workspace/data/oneflow",
    train=False,
    transform=transforms.ToTensor(),
    download=True,
    source_url="https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/mnist/MNIST/",
)



train_dataloader = flow.utils.data.DataLoader(
    training_data, BATCH_SIZE, shuffle=True
)
test_dataloader = flow.utils.data.DataLoader(
    test_data, BATCH_SIZE, shuffle=False
)



for x, y in train_dataloader:
    print("x.shape:", x.shape)
    print("y.shape:", y.shape)
    break



class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(DEVICE)
print(model)


loss_fn = nn.CrossEntropyLoss().to(DEVICE)
optimizer = flow.optim.SGD(model.parameters(), lr=1e-3)


def train(iter, model, loss_fn, optimizer):
    size = len(iter.dataset)
    for batch, (x, y) in enumerate(iter):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        # Compute prediction error
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        current = batch * BATCH_SIZE
        if batch % 100 == 0:
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")



def test(iter, model, loss_fn):
    size = len(iter.dataset)
    num_batches = len(iter)
    model.eval()
    test_loss, correct = 0, 0
    with flow.no_grad():
        for x, y in iter:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            pred = model(x)
            test_loss += loss_fn(pred, y)
            bool_value = (pred.argmax(1).to(dtype=flow.int64)==y)
            correct += float(bool_value.sum().numpy())
    test_loss /= num_batches
    print("test_loss", test_loss, "num_batches ", num_batches)
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}, Avg loss: {test_loss:>8f}")


# 定义5轮 epoch，每训练完一个 epoch 都使用 test 来评估一下网络的精度
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")


flow.save(model.state_dict(), "/workspace/output/model")
