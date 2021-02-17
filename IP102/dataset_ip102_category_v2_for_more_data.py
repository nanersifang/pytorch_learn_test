# -*- coding: utf-8 -*-
#这个程序是为临时获取更多的数据参与训练所用，是一个过渡版本，以后还需要升级，2021。2。17

#import torch as t
from torch.utils import data
#import numpy as np
import os
from PIL import Image
from torchvision import transforms as T

transform = T.Compose([
    T.Resize((96,96),2),#缩放图片（Image),保持长宽比不变，最短边为32像素
    T.CenterCrop(32), #从图片中间切出32*32的图片
    T.ToTensor(),#将图片（image)转成Tensor，归一化至[0,1]
    #T.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5]) #标准化至[-1,1],规定均值和标准差
    ])

rice = [str(num) for num in range(0,14)] #大米
corn = [str(num) for num in range(14,27)] #玉米
wheat = [str(num) for num in range(27,36)] #小麦
beet = [str(num) for num in range(36,44)] #甜菜
alfalfa = [str(num) for num in range(44,57)] #苜蓿䓍
vitis = [str(num) for num in range(57,73)] #葡萄
citrus = [str(num) for num in range(73,92)] #柑橘
mango = [str(num) for num in range(92,102)] #芒果

plant = {'rice':rice,
         'corn':corn,
         'wheat':wheat,
         'beet':beet,
         'alfalfa':alfalfa,
         'vitis':vitis,
         'citrus':citrus,
         'mango':mango,
         }

class Dataset_IP102(data.Dataset):
    def __init__(self,root,train = True,transforms=None,category=[x for x in range(0,102)]):
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
                    class_num = str_txt[1].replace('\n','')
                    if not os.path.exists(root+'/images/'+str_txt[0]) or class_num not in category:
                            continue
                        
                    self.dict_labels[str(str_txt[0])[:-4]] = int(class_num)
                    
                    #把数据增强后的图片也加入到训练集中
                    for i in range(1,26):
                        if not os.path.exists(root+'/images/'+str_txt[0][:-4] + 'augmentation' + str(i) +'.jpg') or class_num not in category:
                            continue
                        
                        self.dict_labels[str(str_txt[0])[:-4]] = int(class_num)
                              
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
                    class_num = str_txt[1].replace('\n','')
                    if not os.path.exists(root+'/images/'+str_txt[0]) or class_num not in category:
                        continue
                    self.dict_labels[str(str_txt[0])[:-4]] = int(class_num)
            
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
    
def show_category_len():
    file_dir = 'F:/5.datasets/IP102_V2.1'
    #print('train')
    for k,v in plant.items():
        train_dataset = Dataset_IP102(file_dir,train=True,transforms=transform,category=v)
        print('train:' + k  + ' ' + str(train_dataset.__len__()))
        
        test_dataset = Dataset_IP102(file_dir,train=False,transforms=transform,category=v)
        print('test:' + k + ' ' + str(test_dataset.__len__()))
        
        

if __name__=='__main__':
    #dataset = Dataset_IP102('F:/5.datasets/ip102_v1.1/',train=False,transforms=transform,category=corn)
    #img,label = dataset[0]
    '''
    for img,label in dataset:
        print(img.size(),label,dataset.__len__())
    '''
        
    show_category_len()