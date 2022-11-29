import torch.nn.functional as F
import torch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.optim as optim
import os
import time
import torchvision
import sys
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ***************************初始化一些函数********************************
# torch.cuda.set_device(gpu_id)#使用GPU


# *************************************数据集的设置****************************************************************************
#root = os.getcwd() + '/data1/'  # 数据集的地址


# 定义读取文件的格式
def default_loader(path):
    return Image.open(path).convert('L')


class MyDataset(Dataset):
    # 创建自己的类： MyDataset,这个类是继承的torch.utils.data.Dataset
    # **********************************  #使用__init__()初始化一些需要传入的参数及数据集的调用**********************
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        super(MyDataset, self).__init__()
        # 对继承自父类的属性进行初始化
        fh = open(txt, 'r')
        imgs=[]
        # 按照传入的路径和txt文本参数，以只读的方式打开这个文本
        for line in fh:  # 迭代该列表#按行循环txt文本中的内
            line = line.strip('\n')
            line = line.rstrip('\n')
            # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
            words = line.split()
            # 用split将该行分割成列表  split的默认参数是空格，所以不传递任何参数时分割空格
            imgs.append((words[0], int(words[1])))
            # 把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
        # 很显然，根据我刚才截图所示txt的内容，words[0]是图片信息，words[1]是lable
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        # *************************** #使用__getitem__()对数据进行预处理并返回想要的信息**********************


    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        fn, label = self.imgs[index]
    # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
    #     img = self.loader(fn)
        img=Image.open(fn)    # 按照路径读取图片
        if self.transform is not None:
            img = self.transform(img)
        # 数据标签转换为Tensor
        return img, label
    # return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容
    # **********************************  #使用__len__()初始化一些需要传入的参数及数据集的调用**********************


    def __len__(self):
    # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)

trans = []
trans.append(torchvision.transforms.Resize(size=224))
trans.append(torchvision.transforms.ToTensor())
transform = torchvision.transforms.Compose(trans)

train_data = MyDataset(txt=r'C:\Users\Nicole\PycharmProjects\pythonProject1\ORACLE_MNIST_SRC\data\train.txt', transform=transform)
test_data = MyDataset(txt=r'C:\Users\Nicole\PycharmProjects\pythonProject1\ORACLE_MNIST_SRC\data\test.txt', transform=transform)

train_iter = DataLoader(dataset=train_data, batch_size=128, shuffle=True,num_workers=0)
test_iter = DataLoader(dataset=test_data, batch_size=128, shuffle=False,num_workers=0)
batch_size=128
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
lr, num_epochs = 0.001,8#定义优化率和训练epochs为0.001和5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)#选择优化器
if __name__ == '__main__':
    train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)#训练模型

