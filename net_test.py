# -*- coding: utf-8 -*-

import torch as t
from IP102.dataset_ip102 import Dataset_IP102,transform
#from LeNet_pytorch_test import NetForIP102

test_dataset = Dataset_IP102('f:/5.backup/ip102_20201116/ip102_v1.1/',train=False,transforms=transform)
test_dataloader = t.utils.data.DataLoader(test_dataset,
                                     batch_size=1,#14
                                     shuffle=False,
                                     drop_last=True)

def run():
    t.multiprocessing.freeze_support()
    print('loop')

if __name__ == '__main__':
    net = t.load('data/lenet.pkl')
    run()
    correct = 0 #预测正确的图片数
    total = 0 #总共的图片数
    
    #由于测试的时候不需要求导，可以暂时关闭autograd,提高速度，节约内存
    num=0
    print('开始测试了')
    with t.no_grad():
        print('到这里了')
        for data in test_dataloader:
            if num%1000==0:
                print(num,'/',test_dataloader.__len__(),' 开始测试：',num)
            num+=1
            images, labels = data
            outputs = net(images)
            _,predicted = t.max(outputs,1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            #print('结束测试：',num)
            
    print(test_dataloader.__len__(),'张测试集中的准确率：%d %%' % (100 * int(correct)/total))
