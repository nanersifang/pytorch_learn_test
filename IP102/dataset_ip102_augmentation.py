# -*- coding: utf-8 -*-
#对数据集进行数据增强，数据增强后返回结果。2021年2月16日


#import torch as t
from torch.utils import data
#import numpy as np
import os
from PIL import Image
from torchvision import transforms as T

#为了数据增强新加的库
import torchvision.transforms.functional as TF
from imgaug import augmenters as iaa
import imageio
import matplotlib.pyplot as plt

# Define an augmentation pipeline
aug_pipeline = iaa.Sequential([
    iaa.Sometimes(0.5, iaa.GaussianBlur((0, 3.0))), # apply Gaussian blur with a sigma between 0 and 3 to 50% of the images
    # apply one of the augmentations: Dropout or CoarseDropout
    iaa.OneOf([
        iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
        iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
    ]),
    # apply from 0 to 3 of the augmentations from the list
    iaa.SomeOf((0, 3),[
        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
        iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
        iaa.Fliplr(1.0), # horizontally flip
        iaa.Sometimes(0.5, iaa.CropAndPad(percent=(-0.25, 0.25))), # crop and pad 50% of the images
        iaa.Sometimes(0.5, iaa.Affine(rotate=5)) # rotate 50% of the images
    ])
],
random_order=True # apply the augmentations in random order
)

# Define the augmentations
AUG_TRAIN = aug_pipeline # use our pipeline as train augmentations

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

class Dataset_IP102_Augmentation(data.Dataset):
    def __init__(self,root,train = True,transforms=None,category=[x for x in range(0,102)],augmentations = AUG_TRAIN):
        #定义取训练集的数据集
        self.augmentations = augmentations
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
        
        data = imageio.imread(self.root+'/images/'+img_path[0] + '.jpg')
        try:
            data = self.augmentations.augment_image(data)
            imageio.imwrite(r'F:/5.datasets/test/aug_images/'+img_path[0] + '-aug_pipline.jpg',data)
            data = Image.open(r'F:/5.datasets/test/aug_images/'+img_path[0] + '-aug_pipline.jpg')
        except Exception as e:
            print(e)
        finally:
            data = Image.open(self.root+'/images/'+img_path[0] + '.jpg')
        #data = data.convert('RGB')
        #array = np.asarray(pil_img)
        #data = t.from_numpy(array)
        #plt.imshow(data)
        
        #plt.imshow(data)
        #data = Image.open(r'F:/5.datasets/test/aug_images/'+img_path[0] + '-aug_pipline.jpg')
        data = data.convert('RGB')
        
        if self.transforms:
            #data = self.transforms(TF.to_tensor(data))
            data = self.transforms(data)
            
        
        #data = TF.to_tensor(self.augmentations.augment_image(data))
        return data,label
    
    def __len__(self):
        return len(self.lst)
    
def show_category_len():
    file_dir = 'F:/5.datasets/IP102_V2.1'
    #print('train')
    for k,v in plant.items():
        train_dataset = Dataset_IP102_Augmentation(file_dir,train=True,transforms=transform,category=v)
        print('train:' + k  + ' ' + str(train_dataset.__len__()))
        #print(next(train_dataset))
        
        test_dataset = Dataset_IP102_Augmentation(file_dir,train=False,transforms=transform,category=v)
        print('test:' + k + ' ' + str(test_dataset.__len__()))
        #print(next(test_dataset))
        
        train_dataset[0]
        test_dataset[0]
        
        
        
        

if __name__=='__main__':
    #dataset = Dataset_IP102('F:/5.datasets/ip102_v1.1/',train=False,transforms=transform,category=corn)
    #img,label = dataset[0]
    '''
    for img,label in dataset:
        print(img.size(),label,dataset.__len__())
    '''
        
    show_category_len()