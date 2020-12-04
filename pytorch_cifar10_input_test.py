# -*- coding: utf-8 -*-
##################################################
'''
小试牛刀：CIFAR-10分类
下面我们来尝试实现对CIFAR-10数据集的分类，步骤如下：
1.使用torchvision加载并预处理CIFAR-10数据集
2.定义网络
3.定义损失函数和优化器
4.训练网络并更新网络参数
5.测试网络
'''
##################################################
import torch as t
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage

show = ToPILImage()

#定义对数据的预处理
transform = transforms.Compose([
    transforms.ToTensor(),#转为Tensor
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),#归一化
    ]) 

#训练集
trainset = tv.datasets.CIFAR10(
    root = './data/',
    train = True,
    download = True,
    transform=transform)

trainloader = t.utils.data.DataLoader(
    trainset,batch_size=4,shuffle=True)

#测试集
testset = tv.datasets.CIFAR10(
    './data/',train=False,download=True,transform=transform)

testloader = t.utils.data.DataLoader(
    testset,batch_size=4,shuffle=False,num_workers=2)

classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

(data,label) = trainset[100]
print(classes[label])

#（data+1)/2 是为了还原被归一化的数据
show((data+1)/2).resize((100,100))

dataiter = iter(trainloader)
images,labels = dataiter.next()#返回四张图片及标签
print(''.join('%11s'%classes[labels[j]] for j in range(4)))
#show(tv.utils.make_grid((images+1)/2).resize((400,100)))
show(tv.utils.make_grid((images+1)/2))

