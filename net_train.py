# -*- coding: utf-8 -*-

import torch as t
from pytorch_config import Config

print('读取数据。。。')


print('\n\n开始配置。。。')

from LeNet_pytorch_test import NetForIP102
#定义损失函数和优化器（loss和optimizer）
from torch import optim
import torch.nn as nn
#import torch as t
import torchvision as tv
from IP102.dataset_ip102 import Dataset_IP102,transform

'''
opt = Config()
transforms = tv.transforms.Compose([
    tv.transforms.Resize(opt.image_size),
    tv.transforms.CenterCrop(opt.image_size),
    tv.transforms.ToTensor(),
    #tv.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

#dataset = tv.datasets.ImageFolder(opt.image_path,transform = transforms)
dataset = tv.datasets.ImageFolder('f:/5.backup/ip102_20201116/ip102_v1.1/',transform = transforms)
'''
train_dataset = Dataset_IP102('f:/5.backup/ip102_20201116/ip102_v1.1/',train=True,transforms=transform)
dataloader = t.utils.data.DataLoader(train_dataset,
                                     batch_size=1,#14
                                     shuffle=False,
                                     drop_last=True)

net = NetForIP102()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

print('开始训练。。。')

for epoch in range(1):
    
    running_loss = 0.0
    for i, data in enumerate(dataloader,0):
    #for i, data in enumerate(zip(x_train,y_train),0):
        #输入数据
        
        inputs,labels = data
        
        '''
        print(inputs,labels)
        inputs = t.tensor(inputs) #转换ndarray数据类型为tensor
        labels = t.tensor(labels) #转换ndarray数据类型为tensor
        print(inputs,labels)
        '''
        
        #梯度清零
        optimizer.zero_grad()
        
        #forward + backward
        outputs = net(inputs)        
        loss = criterion(outputs,labels)
        loss.backward()
        
        #更新参数
        optimizer.step()
        
        #打印log信息
        #loss 是一个scalar,需要使用loss.item()来获取数值，不能使用loss[0]
        running_loss += loss.item()
        
        if i%200 == 199:#每2000个batch打印一下训练状态
            print(inputs,'\n\n')
            print(labels)
            print('[%d,%5d] loss:%.3f'%(epoch+1,i+1,running_loss/200))
            running_loss = 0.0
t.save(net,'data/lenet.pkl')
            
print('Finished Training')
