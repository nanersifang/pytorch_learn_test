# -*- coding: utf-8 -*-

import torch as t
from torch.utils import data
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T

transform = T.Compose([
    T.Resize((32,32)),#缩放图片（Image),保持长宽比不变，最短边为32像素
    T.CenterCrop(32), #从图片中间切出32*32的图片
    T.ToTensor(),#将图片（image)转成Tensor，归一化至[0,1]
    #T.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5]) #标准化至[-1,1],规定均值和标准差
    ])

class Dataset_IP102(data.Dataset):
    def __init__(self,root,train = True,transforms=None):
        #定义取训练集的数据集
        
        self.root = root
        self.dict_labels = dict()
        self.transforms = transforms
        if train:
            train_dir = root+'/' + 'train.txt'
            
            with open(train_dir) as f:
                lines = f.readlines()
                for line in lines:
                    str_txt = line.split(' ')
                    if not os.path.exists(root+'/images/'+str_txt[0]):
                        continue
                    self.dict_labels[str(str_txt[0])[:-4]] = int(str_txt[1].replace('\n',''))
                              
            #imgs = os.listdir(root)
            #所有的图片的绝对路径
            #这里不实际加载图片，只是指定路径，当调用__getitem__时才会真正读图片
            #self.imgs = [os.path.join(root,img) for img in imgs]
        else:
            test_dir = root+'/' + 'test.txt'
            self.dict_labels = dict()
            with open(test_dir) as f:
                lines = f.readlines()
                for line in lines:
                    str_txt = line.split(' ')
                    if not os.path.exists(root+'/images/'+str_txt[0]):
                        continue
                    self.dict_labels[str(str_txt[0])[:-4]] = int(str_txt[1].replace('\n',''))
            
        self.lst = list(self.dict_labels.items())
    
    def __getitem__(self,index):
        #print(self.dict_labels)
        
        #print(lst)
        img_path = self.lst[index]
        #dog->1,cat->0
        label = self.lst[index][1]
        data = Image.open(self.root+'/images/'+img_path[0] + '.jpg')
        data = data.convert('RGB')
        #array = np.asarray(pil_img)
        #data = t.from_numpy(array)
        if self.transforms:
            data = self.transforms(data)
        return data,label
    
    def __len__(self):
        return len(self.lst)
    
    
if __name__=='__main__':
    dataset = Dataset_IP102('f:/5.backup/ip102_20201116/ip102_v1.1/',train=True,transforms=transform)
    #img,label = dataset[0]
    for img,label in dataset:
        print(img.size(),label,dataset.__len__())