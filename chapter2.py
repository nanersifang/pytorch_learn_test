#定义网络
#####################################
'''
定义网络时，需要继承nn.Module，并实现它的forward方法，把网络中具有可学习参数的层放在
构造函数__init__()中。如果某一层（如ReLU)不具有可学习的参数，则既可以放在构造函数中，
也可以不放，但建议不放在其中，而在forward中使用nn.functional代替。
'''
#####################################
import torch as t
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        #nn.Module子类的函数必须在构造函数中执行父类的构造函数
        #下式等价于nn.Module.__init__(self)
        super(Net,self).__init__()
        
        #卷积层’1‘表示输入图片为单通道，’6‘表示输出通道数，’5‘表示卷积核为5*5
        self.conv1 = nn.Conv2d(1,6,5)
        #卷积层
        self.conv2 = nn.Conv2d(6,16,5)
        #仿射层/全连接层，y=Wx+b
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
    def forward(self,x):
        #卷积->激活->池化
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        #reshape,'-1'表示自适应
        x = x.view(x.size()[0],-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
net = Net()
print(net)
params = list(net.parameters())
print(len(params))
for name,parameters in net.named_parameters():
    print(name,':',parameters.size())
    
tensor_input = t.randn(1,1,32,32)
out = net(tensor_input)
print(out.size())
print("net.zero_grad():",net.zero_grad())
print('out.backward(t.ones(1,10)):',out.backward(t.ones(1,10)))

output = net(tensor_input)
target = t.arange(0,10).view(1,10).float()
criterion = nn.MSELoss()
loss = criterion(output,target)
print('loss:',loss)

#运行.backward，观察调用之前和调用之后的grad
net.zero_grad() #把net中所有可学习参数的梯度清零
print('反向传播之前 conv1.bias的梯度')
print(net.conv1.bias.grad)
loss.backward()
print('反向传播之后的conv1.bias的梯度')
print(net.conv1.bias.grad)











