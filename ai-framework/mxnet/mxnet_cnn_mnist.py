from __future__ import print_function

import argparse
import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from mxnet import gluon, autograd as ag, nd
import os

"""
python /workspace/code/mxnet_cnn_mnist.py \
--train-dataset-path '/workspace/data/mxnet/' \
--test-dataset-path '/workspace/data/mxnet/' \
--output-path "/workspace/output/mxnet_model"

"""

import struct
import gzip
import matplotlib.pyplot as plt
from PIL import Image



def get_mnist(path='data'):
    def read_data(label_url, image_url):
        if not os.path.isdir(path):
            os.makedirs(path)
        with gzip.open(label_url) as flbl:
            struct.unpack(">II", flbl.read(8))
            label = np.frombuffer(flbl.read(), dtype=np.int8)
        with gzip.open(image_url, 'rb') as fimg:
            _, _, rows, cols = struct.unpack(">IIII", fimg.read(16))
            image = np.frombuffer(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
            image = image.reshape(image.shape[0], 1, 28, 28).astype(np.float32)/255
        return (label, image)

    # changed to mxnet.io for more stable hosting

    (train_lbl, train_img) = read_data(
        path+'train-labels-idx1-ubyte.gz', path+'train-images-idx3-ubyte.gz')
    (test_lbl, test_img) = read_data(
        path+'t10k-labels-idx1-ubyte.gz', path+'t10k-images-idx3-ubyte.gz')
    return {'train_data':train_img, 'train_label':train_lbl,
            'test_data':test_img, 'test_label':test_lbl}



transform = transforms.Compose([
    # 将PIL Image或numpy.ndarray转换为tensor，并除255归一化到[0,1]之间
    transforms.ToTensor(), 
    # 标准化处理-->转换为标准正太分布，使模型更容易收敛
    transforms.Normalize((0.5,),(0.5,))
    ])


# define network
class NeuralNetwork(gluon.Block):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
        self.conv1 = nn.Conv2D(20, kernel_size=(5,5), strides=(1,1), padding=(2,2), activation='relu')
        # 定义池化层，池化核的大小kernel_size为2，池化步长为2
        self.max_pool1 = nn.MaxPool2D(pool_size=(2,2), strides=(2,2))

        # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
        self.conv2 = nn.Conv2D(20, kernel_size=5, strides=(1,1), padding=(2,2), activation='relu')
        # 定义池化层，池化核的大小kernel_size为2，池化步长为2
        self.max_pool2 = nn.MaxPool2D(pool_size=(2,2), strides=(2,2))
        # 定义一层全连接层，输出维度是10
        self.fc = nn.Dense(10)


    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = x.reshape((x.shape[0], -1))
        x = self.fc(x)
        return x

net = NeuralNetwork()


# train
def test(ctx, val_data):
    metric = mx.metric.Accuracy()
    
    # 重置
    val_data.reset()

    for batch in val_data:
        # data = data.as_in_context(ctx)
        # label = label.as_in_context(ctx)
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        # Splits train labels into multiple slices along batch_axis
        # and copy each slice into a context.
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        outputs = []
        for x in data:
            output = net(x)
            outputs.append(output)
        metric.update(label, outputs)

    return metric.get()




train_acc_list = []
val_acc_list = []



def train(args, ctx, train_data, val_data):
    # Collect all parameters from net and its children, then initialize them.
    net.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    # Trainer is for updating parameters with gradient.
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': args.lr, 'momentum': args.momentum})
    metric = mx.metric.Accuracy()
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    
    for epoch in range(args.epochs):
        # reset data iterator and metric at begining of epoch.
        # 每次迭代后需要重置
        metric.reset()
        train_data.reset()
        # for i, batch in enumerate(train_data):
        #     data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        #         # Splits train labels into multiple slices along batch_axis
        #     # and copy each slice into a context.
        #     label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)

        for i, batch in enumerate(train_data):
            # Copy data to ctx if necessary
            # data = data.as_in_context(ctx)
            # label = label.as_in_context(ctx)
            data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
            # Splits train labels into multiple slices along batch_axis
            # and copy each slice into a context.
            label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)

            outputs = []
            # Start recording computation graph with record() section.
            # Recorded graphs can then be differentiated with backward.
            with autograd.record():
                for x, y in zip(data, label):
                    z = net(x)
                    L = loss(z, y)
                    L.backward()
                    outputs.append(z)

            # take a gradient step with batch_size equal to data.shape[0]
            trainer.step(batch.data[0].shape[0])
            # update metric at last.
            metric.update(label, outputs)

            if i % args.log_interval == 0 and i > 0:
                name, acc = metric.get()
                print('[Epoch %d Batch %d] Training: %s=%f'%(epoch, i, name, acc))

        name, acc = metric.get()
        print('[Epoch %d] Training: %s=%f'%(epoch, name, acc))


        name, val_acc = test(ctx, val_data)
        print('[Epoch %d] Validation: %s=%f'%(epoch, name, val_acc))
        train_acc_list.append(acc)
        val_acc_list.append(val_acc)



def plot(train_acc_list, val_acc_list, output_path):
    fig, ax = plt.subplots()

    train_freqs = [i for i in range(len(train_acc_list))]
    val_freqs = [i for i in range(len(val_acc_list))]

    # 绘制训练损失变化曲线
    ax.plot(train_freqs, train_acc_list, color='#e4007f', label=" train/accuracy curve")
    ax.plot(val_freqs, val_acc_list, color='#fff000', label="val/accuracy curve")

    # 绘制坐标轴和图例
    ax.set_ylabel("accuracy", fontsize='large')
    ax.set_xlabel("epoch", fontsize='large')
    ax.set_title("image classification")
    ax.legend(loc='upper right', fontsize='x-large')

    plt.savefig(output_path+'/mxnet_cnn_image_classification_accuracy_curve.png')
    # plt.show()



def main(ctx):
    parser = argparse.ArgumentParser(description='MXNet Gluon MNIST Example')
    parser.add_argument("--pretrain-model-path", dest="pretrain_model_path", required=False, type=str, default=None, help="预训练模型路径")
    parser.add_argument("--train-dataset-path", type=str, default="/Users/liguodong/data/mnist", help="训练集路径")
    parser.add_argument("--test-dataset-path", type=str, default="/Users/liguodong/data/mnist", help="测试集路径")
    parser.add_argument("--output-path", type=str, default="/Users/liguodong/output/oneflow_model",help="模型输出路径")
    parser.add_argument('--batch-size', type=int, default=100,
                        help='batch size for training and testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Train on GPU with CUDA')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')

    args = parser.parse_args()
    train_dataset_path = args.train_dataset_path
    test_dataset_path = args.test_dataset_path
    output_path = args.output_path
    pretrain_model_path = args.pretrain_model_path

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # train_dataset = gluon.data.vision.MNIST(train_dataset_path, train=True)
    # train_data = gluon.data.DataLoader(train_dataset.transform_first(transform), batch_size=args.batch_size, shuffle=True, last_batch='discard')
    # test_dataset = gluon.data.vision.MNIST(test_dataset_path, train=False)
    # val_data = gluon.data.DataLoader(test_dataset.transform_first(transform), batch_size=args.batch_size, shuffle=False)

    mx.random.seed(42)

    mnist = get_mnist(path=train_dataset_path)

    batch_size = 64
    train_data = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)
    val_data = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)

    train(args, ctx,train_data,val_data)
    net.save_parameters(output_path+'/mnist.params')
    plot(train_acc_list, val_acc_list, output_path)



if __name__ == '__main__':
    #ctx = mx.gpu(0)
    # ctx = mx.cpu()
    ctx = [mx.cpu()]
    # ctx = [mx.gpu(4)]

    # gpus = mx.test_utils.list_gpus()
    # ctx =  [mx.gpu()] if gpus else [mx.cpu(0), mx.cpu(1)]
    main(ctx)


