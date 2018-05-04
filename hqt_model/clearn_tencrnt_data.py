# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 12:20:53 2018

@author: jiguang

"""

import pandas as pd
import numpy as np
from csv import DictWriter
file_data='C:\\Users\\jiguang\\Desktop\\tencent\\preliminary_contest_data\\userFeature.data.txt'
save_data='C:\\Users\\jiguang\\Desktop\\tencent\\preliminary_contest_data\\userFeature.csv'
def clarn_data(file_data,save_data):
    df_data_iterator=pd.read_csv(file_data,sep ='\t',iterator=True)
    with open(save_data,'w') as csv_file:
        headers=['uid', 'age', 'gender', 'marriageStatus', 'education', 'consumptionAbility', 'LBS', 'interest1', 'interest2',
                   'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3',  'topic1', 'topic2', 'topic3', 'appIdInstall',
                   'appIdAction', 'ct', 'os', 'carrier', 'house']
        writer=DictWriter(csv_file,fieldnames=headers,lineterminator='\n')
        writer.writeheader() 
        count=0
        while 1:
            try:
                df_data=df_data_iterator.get_chunk(10000)
                for i in xrange(df_data.shape[0]):
                    t=df_data.iloc[i,0]
                    tt=t.strip().split('|')
                    data_dic={}
                    for line in tt:
                        line_list=line.split(' ')
                        data_dic[line_list[0]]=' '.join(line_list[1:])
                    writer.writerow(data_dic)
                count+=1
                if count%50==0:
                    print count*10000
            except :
                break