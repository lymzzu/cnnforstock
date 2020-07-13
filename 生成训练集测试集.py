# -*- coding: utf-8 -*-
"""
Created on Sun May 24 22:22:57 2020

@author: hunan
"""

import pandas as pd
import tool
import os
#from multiprocessing import Pool
def get_data(i,path="./Data/original_data/000001.csv",window_size=30,step=5):
    df = pd.read_csv(path)
    #x_train=[]
    #y_train = []
    #for index in range(window_size,len(df)-step):
    for index in range(window_size,window_size+10):
        # 取数据[-2]表示使用的特征是由今天之前的数据计算得到的
        data = df.iloc[index-window_size:index]
  #      print(data)
        tool.fig(data,'./Data/train_X/'+i[:-4]+'_{}'.format(index))
       # x_train.append(img)
        # 对今天的交易进行打标签，涨则标记1，跌则标记0
        if df['close'][index-1]* 1.03 < df['close'][index:index+step+1].max():
            label = 1
        else:
            label = 0
        y_train.append([i[:-4]+'_{}'.format(index),label])
        #print(i[:-4]+'_{}'.format(index))
    # Todo
     #   画出Kxian,保存为图片的矢量矩阵。
    pd.DataFrame(y_train).to_csv('./Data/train_label_y.csv',index =False)
    #len(x_train)
if __name__ == '__main__':
    namelist = os.listdir("./Data/original_data")
    #p_pool = Pool(5)
    y_train=[] #全局变量
    for i in namelist[0:50]:
        #p_pool.apply_async(func=get_data, args=(i,"./Data/original_data/"+i))
        get_data(i,path = "./Data/original_data/"+i) 
        print(i)
        #break
    #p_pool.close()
    #p_pool.join()
'''
from multiprocessing import Pool

import os, time, random

def long_time_task(name):
    print('Run task %s (%s)...' % (name, os.getpid()))
    start = time.time()
    time.sleep(random.random() * 3)
    end = time.time()
    print('Task %s runs %0.2f seconds.' % (name, (end - start)))

if __name__=='__main__':
    print('Parent process %s.' % os.getpid())
    p = Pool(4)
    for i in range(5):
        p.apply_async(long_time_task, args=(i,))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')
'''