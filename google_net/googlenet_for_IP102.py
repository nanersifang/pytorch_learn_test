# -*- coding: utf-8 -*-

import torch as t
from google_net import googlenet,data_tf
import os,sys

#得到当前根目录
o_path = os.getcwd()
sys.path.append(o_path)

from vgg_net import utils

from IP102.dataset_ip102 import Dataset_IP102

file_dir = 'F:/5.datasets/ip102_v1.1/'
train_dataset = Dataset_IP102(file_dir,train=True,transforms=data_tf)
train_data = t.utils.data.DataLoader(train_dataset,
                                     batch_size=64,#14
                                     shuffle=False,
                                     drop_last=True)

test_dataset = Dataset_IP102(file_dir,train=False,transforms=data_tf)
test_data = t.utils.data.DataLoader(test_dataset,
                                     batch_size=64,#14
                                     shuffle=False,
                                     drop_last=True)

net = googlenet(3,102)
optimizer = t.optim.SGD(net.parameters(),lr=0.01)
criterion = t.nn.CrossEntropyLoss()

utils.train(net,train_data,test_data,20,optimizer,criterion)
