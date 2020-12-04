# -*- coding: utf-8 -*-

import torch as t
from resnet_test import resnet
import os,sys

#得到当前根目录
o_path = os.getcwd()
sys.path.append(o_path)

from vgg_net import utils

from IP102.dataset_ip102 import Dataset_IP102,transform


train_dataset = Dataset_IP102('f:/5.backup/ip102_20201116/ip102_v1.1/',train=True,transforms=transform)
train_data = t.utils.data.DataLoader(train_dataset,
                                     batch_size=64,#14
                                     shuffle=False,
                                     drop_last=True)

test_dataset = Dataset_IP102('f:/5.backup/ip102_20201116/ip102_v1.1/',train=False,transforms=transform)
test_data = t.utils.data.DataLoader(train_dataset,
                                     batch_size=64,#14
                                     shuffle=False,
                                     drop_last=True)

net = resnet(3,102)
optimizer = t.optim.SGD(net.parameters(),lr=0.01)
criterion = t.nn.CrossEntropyLoss()

utils.train(net,train_data,test_data,3,optimizer,criterion)
