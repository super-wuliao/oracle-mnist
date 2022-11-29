import time
import torch
from torch import nn, optim
import torchvision
import sys
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#加载数据
def load_data_fashion_mnist(batch_size, resize=None, root='~/Datasets/FashionMNIST'):
    if sys.platform.startswith('win'):#如果sys.platform.startswith('win')
        num_workers = 0#num_workers为零
    else:#不然
        num_workers = 4#为4
    trans = []#定义trans列表，记录数据变换不同类型
    if resize:#如果resize的话
        trans.append(torchvision.transforms.Resize(size=resize))#trans增加数据变换类型：torchvision.transforms.Resize(size=resize)
    trans.append(torchvision.transforms.ToTensor())#trans增加数据变换类型：torchvision.transforms.ToTensor()变成张量
    transform = torchvision.transforms.Compose(trans)#把trans变成可迭代的容器transform：torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)#加载训练数据集mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)#加载测试数据集mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)#定义训练数据迭代器，用于将数据分成几份用于训练train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)#定义测试数据迭代器test_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_iter, test_iter#返回两个数据迭代器

batch_size = 128#batch_size设置为128
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)#定义train_iter, test_iter
class AlexNet(nn.Module):#定义网络模型AlexNet，继承自nn.Module
    def __init__(self):#定义def __init__(self):
        super().__init__()#继承父辈init
        self.conv = nn.Sequential(#定义self.conv为nn.Sequential,下面定义具体结构
            nn.Conv2d(1, 96, 11, 4), # Conv2d卷积层 in_channels 1, out_channels 96, kernel_size 11, stride 4, padding
            nn.ReLU(),#relu激活
            nn.MaxPool2d(3, 2), #MaxPool2d最大池化 kernel_size 3, stride 2
            nn.Conv2d(96, 256, 5, 1, 2),# 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数。卷积层，96,256,5,1,2,
            nn.ReLU(),#relu激活
            nn.MaxPool2d(3, 2),#MaxPool2d最大池化 kernel_size 3, stride 2
            nn.Conv2d(256, 384, 3, 1, 1),# 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。256, 384, 3, 1, 1
            nn.ReLU(),#relu激活
            nn.Conv2d(384, 384, 3, 1, 1),#384, 384, 3, 1, 1
            nn.ReLU(),#relu激活
            nn.Conv2d(384, 256, 3, 1, 1),#384, 256, 3, 1, 1
            nn.ReLU(),#relu激活
            nn.MaxPool2d(3, 2) #MaxPool2d最大池化 kernel_size 3, stride 2
        )
        self.fc = nn.Sequential(#定义结尾部分
            nn.Linear(256*5*5, 4096),#线性层，256*5*5,4096
            nn.ReLU(),#relu激活
            nn.Dropout(0.5),#dropout0.5
            nn.Linear(4096, 4096),#线性层，4096，4096
            nn.ReLU(),#relu激活
            nn.Dropout(0.5),#dropout0.5
            nn.Linear(4096, 10),#线性层，4096，10
        )

    def forward(self, img):#定义前向传播
        feature = self.conv(img)#feature=self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))#output=self.fc(feature.view(img.shape[0], -1))
        return output#返回结果

net = AlexNet()#定义网络net
def evaluate_accuracy(data_iter, net, device=None):#定义准确度计算，传入两个形参data_iter,net，前者在使用的时候传给他一个实参def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):#如果没指定device且net继承自Module
        device = list(net.parameters())[0].device#device使用net.parameters()列表的第一项的device
    acc_sum, n = 0.0, 0#初始化计算正确的总量acc_sum和总量n为零
    with torch.no_grad():#保证梯度不更新
        for X, y in data_iter:#加载data_iter中的X，y
            net.eval() # 评估模式, 这会关闭dropout
            acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()#计算accum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            net.train() # 改回训练模式
            n += y.shape[0]#更新样本总量n
    return acc_sum / n#返回准确度


def train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):#定义训练过程def train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)#给net指定device
    print("training on ", device)#打印当前所在device
    loss = torch.nn.CrossEntropyLoss()#选择loss函数loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):#根据epoch循环num_epochs次
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()#初始化训练的loss总和，训练过程预测正确的数量和，样本总量，这一轮中训练的批次，计时器
        for X, y in train_iter:#加载train_iter中的X，y
            X = X.to(device)#给X指定device
            y = y.to(device)#给y指定device
            y_hat = net(X)#计算经过训练后的X，即预测的y_hat
            l = loss(y_hat, y)#计算损失
            optimizer.zero_grad()#优化器梯度初始化optimizer.zero_grad()
            l.backward()#损失函数反向传播
            optimizer.step()#优化器更新
            train_l_sum += l.cpu().item()#更新训练loss总和，先从cpu中取出tensor张量，再转换为标量
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()#更新预测正确数量和(y_hat.argmax(dim=1)==y)......                  ??????????????minist中的y是什么格式？
            n += y.shape[0]#更新样本总量
            batch_count += 1#更新这一轮中训练批次
        test_acc = evaluate_accuracy(test_iter, net)#计算预测准确率
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'% (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))#打印epoch%d,loss%.4f,train acc%.3f,test acc%.3f,time%.1f
lr, num_epochs = 0.001,1#定义优化率和训练epochs为0.001和5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)#选择优化器
train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)#训练模型


