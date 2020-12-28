# -*- coding: utf-8 -*-
from datetime import datetime
import torch
from torch.autograd import Variable
import docx

def get_acc(output,label):
    total = output.shape[0]
    _,pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct/total

def train(net,train_data,valid_data,num_epochs,optimizer,criterion,net_name='SomeNet'):
    dc = docx.Document()
    if torch.cuda.is_available():
        net = net.cuda()
    prev_time = datetime.now()
    
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        net = net.train()
        for im,label in train_data:
            if torch.cuda.is_available():
                im = Variable(im.cuda())
                label = Variable(label.cuda())
            else:
                im = Variable(im)
                label = Variable(label)
            
            #forward
            output = net(im)
            loss = criterion(output,label)
            #backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()#书中原文是loss.data[0]
            train_acc += get_acc(output,label)
        cur_time = datetime.now()
        h,remainder = divmod((cur_time-prev_time).seconds,3600)
        m,s = divmod(remainder,60)
        time_str = "Time %02d:%02d:%02d"%(h,m,s)
        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            net = net.eval()
            for im,label in valid_data:
                if torch.cuda.is_available():
                    im = Variable(im.cuda(),volatile=True)
                    label = Variable(label.cuda(),volatile=True)
                else:
                    with torch.no_grad():
                        im = Variable(im)
                        label = Variable(label)
                output = net(im)
                loss = criterion(output,label)
                valid_loss += loss.item()
                valid_acc += get_acc(output,label)
                #print(valid_acc,output,label)
            epoch_str = (
                "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss:%f,Valid Acc: %f, "%(epoch,train_loss/len(train_data),
                train_acc  / len(train_data),
                valid_loss / len(valid_data),
                valid_acc / len(valid_data) 
                ))
        else:
            epoch_str = ("Epoch %d. Train Loss:%f,Train Acc:%f,"%
                         (epoch,train_loss/len(train_data),
                          train_acc / len(train_data)))
        
        prev_time = cur_time
        print(net_name + epoch_str + time_str)
        dc.add_paragraph(net_name + epoch_str + time_str)
        
    #保存学习到的网络
    torch.save(net,'../data/'+net_name+'_'+str(datetime.now().date())+'.pkl')
    #保存日志
    dc.save('../data/'+net_name+'_'+str(datetime.now().date())+'.docx')
            
        
            