# -*- coding: utf-8 -*-
#对扩展充后的数据进行训练

import torch as t
from dense_net import densenet,data_tf
import os,sys

#得到当前根目录
o_path = os.getcwd()
sys.path.append(o_path)

from vgg_net import utils

#from IP102.dataset_ip102 import Dataset_IP102
from IP102.dataset_ip102_category import plant
from IP102.dataset_ip102_category_v2_for_more_data import Dataset_IP102_Augmentation

#file_dir = 'F:/5.datasets/ip102_v1.1'

#删除脏数据后的数据集v2
#file_dir = 'F:/5.datasets/IP102_V2.1'
#删除脏数据并进行数据增强后的数据集v2
file_dir = 'F:/5.datasets/ip102_v2.2'
for k,v in plant.items():
    train_dataset = Dataset_IP102_Augmentation(file_dir,train=True,transforms=data_tf,category=v)
    print('训练集类别为',k,'的图片数量有：',train_dataset.__len__())
    train_data = t.utils.data.DataLoader(train_dataset,
                                         batch_size=16,#14
                                         shuffle=True,
                                         drop_last=True)
    
    test_dataset = Dataset_IP102_Augmentation(file_dir,train=False,transforms=data_tf,category=v)
    test_data = t.utils.data.DataLoader(test_dataset,
                                         batch_size=16,#14
                                         shuffle=True,
                                         drop_last=True)
    
    net = densenet(3,102)
    optimizer = t.optim.SGD(net.parameters(),lr=0.01)
    criterion = t.nn.CrossEntropyLoss()
    
    net = utils.train(net,train_data,test_data,25,optimizer,criterion,'densenet'+ '_on_'+ k)
    
    print('\n')