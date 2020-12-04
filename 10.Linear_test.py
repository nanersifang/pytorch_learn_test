# -*- coding: utf-8 -*-

import numpy as np
import torch

x_train = np.stack([x_sample ** i for i in range(1,4)],axis=1)
x_train = torch.from_numpy(x_train).float()#

y_train = torch.from_numpy(y_sample).float().unsqueeze(1)




'''
import json
with open('data/gdp_json.json') as f:
    lst = json.load(f)
    #print(lst)
    lst.sort(key=(lambda x :(x['Year'],x["Value"])),reverse=True)
    
    #print(lst)
    
    with open('data/ordered_gdp.json','w') as f2:
        #f2.write(str(lst))
        json.dump(lst,f2)

    with open('data/ordered_gdp.json')as f3:
        lst2 = json.load(f3)
        print(lst2)
'''
'''     
lst = [1,[0,0,0,0],2,[3,[0,0,0,0],[1,[0,0,0,0]]],3]
def digui(lst_tmp):
    #global lst
    for l in lst_tmp:
        print(l)
        if l==[0,0,0,0]:
            lst_tmp.remove(l)
            digui(lst_tmp)
        elif type(l)==list:
            digui(l)
        
def digui2(lst_tmp):
    
    length = len(lst_tmp)
    i=0   
    while i<length:
        #print(lst_tmp[i])
        if lst_tmp[i]==[0,0,0,0]:            
            del lst_tmp[i]
            i-=1
            length-=1
            
        elif type(lst_tmp[i])==list:
            digui(lst_tmp[i])
            
        i+=1
   
if __name__ == '__main__':
    #global lst
    print(lst)
    lst_new = lst.copy()
    digui(lst_new)
    print(lst_new)
'''   