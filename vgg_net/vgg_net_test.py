# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
from utils import train

'''
定义一个vgg的block，传入三个参数，第一个是模型层数，第二个是通道数，第三个是
输出的通道数，第一层卷积接受的输入通道就是图片输入的通道数，然后输出最后的输出
通道数，后面的卷积接受的通道数就是最后的输出通道数
'''
def vgg_block(num_convs, in_channels, out_channels):
    #定义第一层
    net = [nn.Conv2d(in_channels, out_channels, kernel_size=3,padding=1),nn.ReLU(True)]
    
    for i in range(num_convs-1):#定义后面很多层
        net.append(nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1))
        net.append(nn.ReLU(True))
        
    net.append(nn.MaxPool2d(2,2))#定义池化层
    return nn.Sequential(*net)

block_demo = vgg_block(3,64,128)
print(block_demo)

#首先定义输入为(1,64,300,300)
input_demo = Variable(torch.zeros(1,64,300,300))
output_demo = block_demo(input_demo)
print(output_demo.shape)

#定义vgg block进行堆叠
def vgg_stack(num_convs,channels):
    net = []
    for n,c in zip(num_convs,channels):
        in_c = c[0]
        out_c = c[1]
        net.append(vgg_block(n,in_c,out_c))
    
    return nn.Sequential(*net)

#作为实例，我们定义一个稍微简单一点的vgg结构，其中有8个卷积层
vgg_net = vgg_stack((1,1,2,2,2),((3,64),(64,128),(128,256),(256,512),(512,512)))
print(vgg_net)

test_x = Variable(torch.zeros(1,3,256,256))
test_y = vgg_net(test_x)

print(test_y.shape)#可以看到图片减小了2**5倍，最后再加上几层全连接层，就能够得到我们想要的分类输出

class vgg(nn.Module):
    def __init__(self):
        super(vgg,self).__init__()
        self.feature = vgg_net
        self.fc = nn.Sequential(
            nn.Linear(512,100),
            nn.ReLU(True),
            nn.Linear(100,10)
            )
        
    def forward(self,x):
        x = self.feature(x)
        x = x.view(x.shape[0],-1)
        x = self.fc(x)
        return x
    
class vgg_IP102(nn.Module):
    def __init__(self):
        super(vgg,self).__init__()
        self.feature = vgg_net
        self.fc = nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(True),
            nn.Linear(256,102)
            )
        
    def forward(self,x):
        x = self.feature(x)
        x = x.view(x.shape[0],-1)
        x = self.fc(x)
        return x
    
    
#from utils import train

def data_tf(x):
    x = np.array(x,dtype='float32')/255
    x = (x-0.5)/0.5 #
    x = x.transpose((2,0,1))
    x = torch.from_numpy(x)
    return x

train_set = CIFAR10('../data',train=True,transform=data_tf)
train_data = torch.utils.data.DataLoader(train_set,batch_size=64,shuffle=True)
test_set = CIFAR10('../data',train=False,transform=data_tf)
test_data = torch.utils.data.DataLoader(test_set,batch_size=128,shuffle=False)

net = vgg()
optimizer = torch.optim.SGD(net.parameters(),lr=1e-1)
criterion = nn.CrossEntropyLoss()
print(len(train_data),len(test_data))
print('开始训练并测试。。。')
train(net,train_data,test_data,20,optimizer,criterion)






























