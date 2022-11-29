import torch.nn.functional as F
import torch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import numpy as np
import torch.optim as optim
import os
import time
import torchvision
import sys

class MyDataset(Dataset):
    # 创建自己的类： MyDataset,这个类是继承的torch.utils.data.Dataset
    # **********************************  #使用__init__()初始化一些需要传入的参数及数据集的调用**********************
    def __init__(self, txt, transform=None, target_transform=None, loader=None):
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
        print(fn)
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

if __name__ == '__main__':
    train_data = MyDataset(txt=r'C:\Users\Nicole\PycharmProjects\pythonProject1\ORACLE_MNIST_SRC\data\train.txt',
                           transform=None)
    test_data = MyDataset(txt=r'C:\Users\Nicole\PycharmProjects\pythonProject1\ORACLE_MNIST_SRC\data\test.txt',
                          transform=None)

    train_iter = DataLoader(dataset=train_data, batch_size=128, shuffle=True, num_workers=0)
    test_iter = DataLoader(dataset=test_data, batch_size=128, shuffle=False, num_workers=0)
    #
    # for item in test_iter:
    #     print(item)
    img = Image.open(r'C:\Users\Nicole\PycharmProjects\pythonProject1\ORACLE_MNIST_SRC\data\oracle-mnist-test-easy\test_1001_5.jpg')
    # img=cv2.imread(r'C:\Users\Nicole\PycharmProjects\pythonProject1\oracle-mnist-test-easy\test_1001_5.jpg')