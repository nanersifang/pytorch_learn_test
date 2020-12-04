# -*- coding: utf-8 -*-

from LeNet_pytorch_test import Net
import torchvision as tv

net = Net()
print(net)

#定义损失函数和优化器（loss和optimizer）
from torch import optim
import torch.nn as nn
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

#训练网络
#所有网络的训练流程都是类似的，不断地执行如下流程：
#输入数据
#前向传播+反向传播
#更新参数
import torch as t
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')
print(str(device))

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

#(data,label) = trainset[100]


#t.set_num_threads(8)
for epoch in range(200):
    
    running_loss = 0.0
    for i, data in enumerate(trainloader,0):
        #输入数据
        inputs,labels = data
        
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
        
        if i%2000 == 1999:#每2000个batch打印一下训练状态
            print('[%d,%5d] loss:%.3f'%(epoch+1,i+1,running_loss/2000))
            running_loss = 0.0
            
print('Finished Training')
'''
#此处仅训练了2个epoch(遍历完一遍数据集称为一个epoch),来看看网络有没有效果。将测试图片输入到网络中，
#计算它的label，然后与实际的label进行比较。
print(1)
dataiter = iter(testloader)
print(2)
images, labels = dataiter.next() #一个batch返回4张图片
print(3)
print('实际的label:',''.join('%08s'%classes[labels[j]] for j in range(4)))
print(4)
show(tv.utils.make_grid(images/2-0.5)).resize((400,100))
print(5)

#计算图片在每个类别上的分数
outputs = net(images)
#得分最高的那个类
_,predicted = t.max(outputs.data,1)

print('预测结果：',''.join('%5s'%classes[predicted[j]] for j in range(4)))
'''

'''
correct = 0 #预测正确的图片数
total = 0 #总共的图片数

#由于测试的时候不需要求导，可以暂时关闭autograd,提高速度，节约内存
num=0
print('开始测试了')
with t.no_grad():
    print('到这里了')
    for data in testloader:
        print('开始测试：',num,end=' ')
        num+=1
        images, labels = data
        outputs = net(images)
        _,predicted = t.max(outputs,1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        print('结束测试：',num)
        
print('10000张测试集中的准确率：%d %%' % (100 * int(correct)/total))
'''


'''
net.to(device)
images = images.to(device)
labels = labels.to(device)
output = net(images)
loss = criterion(output,labels)

print(loss)
'''   


def run():
    t.multiprocessing.freeze_support()
    print('loop')

if __name__ == '__main__':
    run()
    correct = 0 #预测正确的图片数
    total = 0 #总共的图片数
    
    #由于测试的时候不需要求导，可以暂时关闭autograd,提高速度，节约内存
    num=0
    print('开始测试了')
    with t.no_grad():
        print('到这里了')
        for data in testloader:
            print('开始测试：',num,end=' ')
            num+=1
            images, labels = data
            outputs = net(images)
            _,predicted = t.max(outputs,1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            print('结束测试：',num)
            
    print('10000张测试集中的准确率：%d %%' % (100 * int(correct)/total))

        
        
        
        