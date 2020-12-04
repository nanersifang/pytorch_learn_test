# -*- coding: utf-8 -*-

#import xlwings as xw
import pandas as pd

df = pd.read_excel('data/abc - 副本.xlsx')

df[['abc']].astype('str')
print(df)
#把逗号替换为空的函数f
f = lambda x:x.replace(',','')
print(df)
print(type(df['abc']))

#对某一列的应用这个函数f
df['abc'] = df[['abc']].apply(f)
print(df)
#print(df)
df[['abc']].astype('float')

df.to_excel('data/abc - 副本.xlsx',index=False)

