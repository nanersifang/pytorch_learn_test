# -*- coding: utf-8 -*-
import numpy as np
import torch
#from vgg_net.utils import test
from dataset_ip102_category_v3_for_test_data import Dataset_IP102,plant

import os,sys
#得到当前根目录
o_path = os.getcwd()
sys.path.append(o_path)
#from densenet.dense_net import densenet

#删除脏数据后的数据集v2
file_dir = 'F:/5.datasets/IP102_V2.1'

def data_tf(x):
    x = x.resize((96,96),2) #将图片放大到96*96
    x = np.array(x, dtype='float32') / 255
    x = (x-0.5) / 0.5 #标准化
    x = x.transpose((2,0,1)) #将 channel 放到第一维，只是pytorch要求的输入方式
    x = torch.from_numpy(x)
    
    return x

#在测试集上进行测试，看看模型的准确性如何
def test(model_dir,test_data,net_name='SomeNet'): 
    model = torch.load(model_dir)
    device = torch.device('cuda')  
    '''
    #print(model.eval())
    correct = 0
    total = len(test_data)
    for x, y in test_data:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().item()
    #print(net_name,':',correct/total)'''
    correct = 0
    total = 0
    for (imgs, labels) in test_data:
        imgs = imgs.to(device)
        labels = labels.to(device)
        out = model(imgs)
        _, pre = torch.max(out.data, 1)
        total += labels.size(0)
        correct += (pre == labels).sum().item()
        
    return correct / total



#model = torch.load('../data/densenet_on_beet_2020-12-28.pkl')
model_dir = '../data/densenet_on_rice_2021-02-09.pkl'
for k,v in plant.items():
    test_dataset = Dataset_IP102(file_dir,train=False,transforms=data_tf,category=v)
    test_data = torch.utils.data.DataLoader(test_dataset,
                                         batch_size=16,#14
                                         shuffle=True,
                                         drop_last=True)
    acc = test(model_dir,test_data,'densenet_on_beet')
    print('densenet_on_rice\'s acc on ' + k + ' is ',acc)