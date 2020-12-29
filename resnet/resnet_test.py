# -*- coding: utf-8 -*-


import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.datasets import CIFAR10

def conv33(in_channel,out_channel,stride=1):
    return nn.Conv2d(in_channel,out_channel,3,stride=stride,padding=1,bias=False)

class residual_block(nn.Module):
    def __init__(self,in_channel,out_channel,same_shape=True):
        super(residual_block,self).__init__()
        self.same_shape = same_shape
        stride=1 if self.same_shape else 2
        
        self.conv1 = conv33(in_channel,out_channel,stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        
        self.conv2 = conv33(out_channel,out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        
        if not self.same_shape:
            self.conv3 = nn.Conv2d(in_channel,out_channel,1,stride=stride)
            
    def forward(self,x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out),True)
        out = self.conv2(out)
        out = F.relu(self.bn2(out),True)
        
        if not self.same_shape:
            x = self.conv3(x)
        return F.relu(x+out,True)
   
'''

'''
#下面我们尝试实现一个ResNet，它就是residual block模块的堆叠
class resnet(nn.Module):
    def __init__(self,in_channel,num_classes,verbose=False):
        super(resnet,self).__init__()
        self.verbose = verbose
        
        self.block1 = nn.Conv2d(in_channel,64,7,2)
        self.block2 = nn.Sequential(
            nn.MaxPool2d(3,2),
            residual_block(64,64),
            residual_block(64,64)            
            )

        self.block3 = nn.Sequential(
            residual_block(64,128,False),
            residual_block(128,128)            
            )
        
        self.block4 = nn.Sequential(
            residual_block(128,256,False),
            residual_block(256,256),
            )
        
        self.block5 = nn.Sequential(
            residual_block(256,512,False),
            residual_block(512,512),
            nn.AvgPool2d(3)
            )
        
        self.classifer = nn.Linear(512,num_classes)
        
    def forward(self,x):
        x = self.block1(x)
        if self.verbose:
            print('block 1 output:{}'.format(x.shape))
        
        x = self.block2(x)
        if self.verbose:
            print('block 2 output:{}'.format(x.shape))
            
        x = self.block3(x)
        if self.verbose:
            print('block 3 output:{}'.format(x.shape))
            
        x = self.block4(x)
        if self.verbose:
            print('block 4 output:{}'.format(x.shape))
            
        x = self.block5(x)
        if self.verbose:
            print('block 5 output:{}'.format(x.shape))
            
        x = x.view(x.shape[0],-1)
        x = self.classifer(x)
        
        return x
    

import sys
import os
#得到当前根目录
o_path = os.getcwd()
sys.path.append(o_path)
from vgg_net import utils
def data_tf(x):
    x = x.resize((96,96),2) #将图片放大到96*96
    x = np.array(x,dtype='float32')/255
    x = (x-0.5)/0.5 #标准化
    x = x.transpose((2,0,1)) #将channel放到第一维，只是pytorch要求的输入方式
    x = torch.from_numpy(x)
    return x

if __name__ == '__main__':
    #输入输出形状相同
    
    test_net = residual_block(3,32,False)
    test_x = Variable(torch.zeros(1,3,96,96))
    print('input:{}'.format(test_x.shape))
    test_y = test_net(test_x)
    print('output:{}'.format(test_y.shape))
    
    train_set = CIFAR10('../data',train=True,transform=data_tf)
    train_data = torch.utils.data.DataLoader(train_set,batch_size=64,shuffle=True)
    
    test_set = CIFAR10('../data',train=False,transform=data_tf)
    test_data = torch.utils.data.DataLoader(test_set,batch_size=128,shuffle=True)
    
    net= resnet(3,10)
    optimizer = torch.optim.SGD(net.parameters(),lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    utils.train(net,train_data,test_data,20,optimizer,criterion)






























    