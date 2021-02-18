# -*- coding: utf-8 -*-
#用来显示模型的model
import numpy as np
import torch
from tensorboardX import SummaryWriter
import torchvision
from dataset_ip102_category_v3_for_test_data import Dataset_IP102,plant
'''
writer = SummaryWriter('runs/exp2')
model = torchvision.models.resnet18(False)
writer.add_graph(model, torch.rand([1,3,224,224]))   # 自己定义的网络有时显示错误
writer.close()

'''

writer1 = SummaryWriter('runs/exp7')
model_dir = '../data/densenet_on_rice_2021-02-09.pkl'
model = torch.load(model_dir)
#imgs = torch.randn(96,3,96,96).requires_grad_(True)
imgs = torch.randn(32,3,96,96)
device = torch.device('cuda')  
imgs = imgs.to(device)
writer1.add_graph(model,imgs)
writer1.close()

'''
#删除脏数据后的数据集v2
file_dir = 'F:/5.datasets/IP102_V2.1'

def data_tf(x):
    x = x.resize((96,96),2) #将图片放大到96*96
    x = np.array(x, dtype='float32') / 255
    x = (x-0.5) / 0.5 #标准化
    x = x.transpose((2,0,1)) #将 channel 放到第一维，只是pytorch要求的输入方式
    x = torch.from_numpy(x)
    
    return x

for k,v in plant.items():
    test_dataset = Dataset_IP102(file_dir,train=False,transforms=data_tf,category=v)
    test_data = torch.utils.data.DataLoader(test_dataset,
                                         batch_size=16,#14
                                         shuffle=True,
                                         drop_last=True)

    model_dir = '../data/densenet_on_rice_2021-02-09.pkl'
    model = torch.load(model_dir)
    # Creates writer1 object.
    # The log will be saved in 'runs/exp'
    writer1 = SummaryWriter('runs/exp5')
    #x = torch.randn(96,3,96,96).requires_grad_(True)
    for (imgs, labels) in test_data:
        device = torch.device('cuda')  
        imgs = imgs.to(device)
        writer1.add_graph(model,imgs)
        break
    
    writer1.close()
    break
'''

