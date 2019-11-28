# -*- coding: utf-8 -*-
"""
Created on 2019-11-27 08:30:25
@author: lxc
"""
#this code is to extract the yolov3 train log

import inspect
import os
import random
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser

parser = ArgumentParser(description='''
                        put the yolo log path to parse
                        this python script will output [log_path]_loss.txt and [log_path]_iou.txt
                        and plot a loss curve , a iou curve of the log file
                        ''')
parser.add_argument('--log_root',type=str,default="logdir",
                    help="log diratory path")
parser.add_argument('--log',type=str,default="first_train_yolov3.log",
                    help="log file name")
args = parser.parse_args()
log_pth = args.log_root+'/log/'+args.log
loss_pth = args.log_root+'/csv/'+args.log.split('.')[0]+'_loss.csv'
iou_pth = args.log_root+'/csv/'+args.log.split('.')[0]+'_iou.csv'
png_root =args.log_root+'/png/' 
def extract_log(log_file,new_log_file,key_word):
    f=open(log_file,'r')
    train_log=open(new_log_file,'w')
    for line in f:
        if 'Syncing' in line:        #多gpu同步信息，我就一个GPU,这里是可以不要的。
            continue
        if 'nan' in line:             #包含nan的不要
            continue
        if key_word in line:        #包含关键字
            train_log.write(line)
    f.close()
    train_log.close()


def plot_loss():
    lines =16000       #rows to be draw
    result = pd.read_csv(loss_pth, skiprows=[x for x in range(lines) if ((x%10!=9) |(x<1000))] ,error_bad_lines=False, names=['loss', 'avg', 'rate', 'seconds', 'images'])
    result.head()

    #print(result)

    result['loss']=result['loss'].str.split(' ').str.get(1)
    result['avg']=result['avg'].str.split(' ').str.get(1)
    result['rate']=result['rate'].str.split(' ').str.get(1)
    result['seconds']=result['seconds'].str.split(' ').str.get(1)
    result['images']=result['images'].str.split(' ').str.get(1)
    result.head()
    result.tail()

    '''
    print(result['loss'])
    print(result['avg'])
    print(result['rate'])
    print(result['seconds'])
    print(result['images'])
    '''
    result['loss']=pd.to_numeric(result['loss'])
    result['avg']=pd.to_numeric(result['avg'])
    result['rate']=pd.to_numeric(result['rate'])
    result['seconds']=pd.to_numeric(result['seconds'])
    result['images']=pd.to_numeric(result['images'])
    result.dtypes

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(result['avg'].values,label='avg_loss')

    min_x = np.argmin(result['avg'])
    ax.plot(min_x,result['avg'][min_x],'ro',label='min avg_loss')
    show_min='(min:{:.4f})'.format(result['avg'][min_x])
    ax.annotate(show_min,xytext=(min_x,result['avg'][min_x]),xy=(min_x,result['avg'][min_x]))
    
    #ax.plot(result['loss'].values,label='loss')
    ax.legend(loc='best')
    ax.set_title('The loss curves')
    ax.set_xlabel('batches*10')
    fig.savefig(png_root+'avg-loss.png',dpi=600)
def plot_iou():
    lines = 16000    #根据train_log_iou.txt的行数修改
    result = pd.read_csv(iou_pth, skiprows=[x for x in range(lines) if (x%10==0 or x%10==9) ] ,error_bad_lines=False, names=['Region Avg IOU', 'Class', 'Obj', 'No Obj', 'Avg Recall','count'])
    result.head()
    
    result['Region Avg IOU']=result['Region Avg IOU'].str.split(': ').str.get(1)
    result['Class']=result['Class'].str.split(': ').str.get(1)
    result['Obj']=result['Obj'].str.split(': ').str.get(1)
    result['No Obj']=result['No Obj'].str.split(': ').str.get(1)
    result['Avg Recall']=result['Avg Recall'].str.split(': ').str.get(1)
    result['count']=result['count'].str.split(': ').str.get(1)
    result.head()
    result.tail()
    
    # print(result.head())
    # print(result.tail())
    # print(result.dtypes)
    print(result['Region Avg IOU'])
    
    result['Region Avg IOU']=pd.to_numeric(result['Region Avg IOU'])
    result['Class']=pd.to_numeric(result['Class'])
    result['Obj']=pd.to_numeric(result['Obj'])
    result['No Obj']=pd.to_numeric(result['No Obj'])
    result['Avg Recall']=pd.to_numeric(result['Avg Recall'])
    result['count']=pd.to_numeric(result['count'])
    result.dtypes
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(result['Region Avg IOU'].values,label='Region Avg IOU')
    
    max_x = np.argmax(result['Region Avg IOU'].values)
    ax.plot(max_x,result['Region Avg IOU'][max_x],'ro',label='max IOU')
    show_max='(max:{})'.format(result['Region Avg IOU'][max_x])
    ax.annotate(show_max,xytext=(max_x,result['Region Avg IOU'][max_x]),xy=(max_x,result['Region Avg IOU'][max_x]))
    
    # ax.plot(result['Class'].values,label='Class')
    # ax.plot(result['Obj'].values,label='Obj')
    # ax.plot(result['No Obj'].values,label='No Obj')
    # ax.plot(result['Avg Recall'].values,label='Avg Recall')
    # ax.plot(result['count'].values,label='count')
    ax.legend(loc='best')
    ax.set_title('The Region Avg IOU curves')
    ax.set_xlabel('batches')
    fig.savefig(png_root+'region-avg-iou.png',dpi=600)

if __name__ == "__main__":
    extract_log(log_pth,loss_pth,'images')
    extract_log(log_pth,iou_pth,'IOU')
    plot_loss()
    plot_iou()
