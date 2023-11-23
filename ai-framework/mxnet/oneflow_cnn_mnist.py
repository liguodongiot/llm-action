import oneflow as flow
import oneflow.nn as nn
from flowvision import transforms
from flowvision import datasets
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import os

"""

python /workspace/code/oneflow/oneflow_cnn_mnist.py \


python /workspace/code/oneflow_cnn_mnist.py \
--train-dataset-path "/workspace/data/oneflow" \
--test-dataset-path "/workspace/data/oneflow" \
--output-path "/workspace/output/oneflow_model"

"""


transform = transforms.Compose([
    # 将PIL Image或numpy.ndarray转换为tensor，并除255归一化到[0,1]之间
    transforms.ToTensor(),
    # 标准化处理-->转换为标准正太分布，使模型更容易收敛
    transforms.Normalize((0.5,),(0.5,))
    ])


BATCH_SIZE=64

# DEVICE = "cpu" # "cuda" if flow.cuda.is_available() else "cpu"

DEVICE = "cuda" if flow.cuda.is_available() else "cpu"

print("Using {} device".format(DEVICE))


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=2)
        # 定义池化层，池化核的大小kernel_size为2，池化步长为2
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=5, stride=1, padding=2)
        # 定义池化层，池化核的大小kernel_size为2，池化步长为2
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 定义一层全连接层，输出维度是10
        self.fc = nn.Linear(in_features=980, out_features=10)
        self.relu=nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2(x)
        x = flow.reshape(x, [x.shape[0], -1])
        x = self.fc(x)
        return x


loss_list = []

def train(epoch, iter, model, loss_fn, optimizer):
    size = len(iter.dataset)
    sum_loss = []
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

        sum_loss.append(loss)

        if batch % 100 == 0:
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    avg_loss = sum(sum_loss) / len(sum_loss)
    loss_list.append(float(avg_loss))

    print("epoch: ", epoch,  f"loss: {avg_loss:>7f}")





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



def plot(loss_list, output_path):
    plt.figure(figsize=(10,5))

    freqs = [i for i in range(len(loss_list))]
    # 绘制训练损失变化曲线
    plt.plot(freqs, loss_list, color='#e4007f', label="image classification train/loss curve")

    # 绘制坐标轴和图例
    plt.ylabel("loss", fontsize='large')
    plt.xlabel("epoch", fontsize='large')
    plt.legend(loc='upper right', fontsize='x-large')

    plt.savefig(output_path+'/oneflow_cnn_image_classification_loss_curve.png')
    # plt.show()


def main():

    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument("--pretrain-model-path", dest="pretrain_model_path", required=False, type=str, default=None, help="预训练模型路径")
    parser.add_argument("--train-dataset-path", type=str, default="/Users/liguodong/data/mnist", help="训练集路径")
    parser.add_argument("--test-dataset-path", type=str, default="/Users/liguodong/data/mnist", help="测试集路径")
    parser.add_argument("--output-path", type=str, default="/Users/liguodong/output/oneflow_model",help="模型输出路径")

    args = parser.parse_args()
    train_dataset_path = args.train_dataset_path
    test_dataset_path = args.test_dataset_path
    output_path = args.output_path
    pretrain_model_path = args.pretrain_model_path

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        

    training_data = datasets.MNIST(
        root=train_dataset_path,
        train=True,
        transform=transforms.ToTensor(),
        download=True,
        source_url="https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/mnist/MNIST/",

    )
    test_data = datasets.MNIST(
        root=test_dataset_path,
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
        
    model = NeuralNetwork().to(DEVICE)
    print(model)

    loss_fn = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = flow.optim.Adam(model.parameters(), lr=1e-3)

    # 定义5轮 epoch，每训练完一个 epoch 都使用 test 来评估一下网络的精度
    epochs = 10
    for t in range(epochs):
        train(t, train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")

    flow.save(model.state_dict(), output_path)
    plot(loss_list, output_path)

        

if __name__ == "__main__":
    main()






