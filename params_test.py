# -*- coding: utf-8 -*-

import torch.optim as optim
from chapter2 import net,tensor_input,criterion,target

#新建一个优化器，指定要调整的参数和学习率
optimizer = optim.SGD(net.parameters(),lr=0.01)

#在训练过程中
#先梯度清零（与net.zero_grad()效果一样）
optimizer.zero_grad()

#计算损失

output = net(tensor_input)
loss = criterion(output,target)

#反向传播
loss.backward()

#更新参数
optimizer.step()
