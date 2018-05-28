#! usr/bin/env python2.7
#coding: utf-8
import pandas as pd
import numpy as np
#import lightgbm as lgb
#from sklearn.model_selection import train_test_split
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.preprocessing import OneHotEncoder,LabelEncoder
#from scipy import sparse
#from sklearn.linear_model import LogisticRegression,SGDClassifier
#from scikitplot.metrics import plot_roc_curve
#from sklearn.metrics import roc_auc_score,roc_curve
#from sklearn.externals import joblib
#import os
#from datetime import datetime
#import re
#from operator import sub,add
#import matplotlib.pyplot as plt
#import numpy
#import random
import scipy.special as special
import math
from math import log
import os
from multiprocessing import Pool
class HyperParam(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def sample_from_beta(self, alpha, beta, num, imp_upperbound):
        sample = numpy.random.beta(alpha, beta, num)
        I = []
        C = []
        for click_ratio in sample:
            imp = random.random() * imp_upperbound
            #imp = imp_upperbound
            click = imp * click_ratio
            I.append(imp)
            C.append(click)
        return I, C

    def update_from_data_by_FPI(self, tries, success, iter_num, epsilon):
        '''estimate alpha, beta using fixed point iteration'''
        #input np.array or Series
        for i in range(iter_num):
            new_alpha, new_beta = self.__fixed_point_iteration(tries, success, self.alpha, self.beta)
            if abs(new_alpha-self.alpha)<epsilon and abs(new_beta-self.beta)<epsilon:
                break
            self.alpha = new_alpha
            self.beta = new_beta

    def __fixed_point_iteration(self, tries, success, alpha, beta):
        '''fixed point iteration'''
        sumfenzialpha = 0.0
        sumfenzibeta = 0.0
        sumfenmu = 0.0
        sumfenzialpha += (special.digamma(success+alpha) - special.digamma(alpha)).sum()
        sumfenzibeta += (special.digamma(tries-success+beta) - special.digamma(beta)).sum()
        sumfenmu = (special.digamma(tries+alpha+beta) - special.digamma(alpha+beta)).sum()

        return alpha*(sumfenzialpha/sumfenmu), beta*(sumfenzibeta/sumfenmu)
def execute_hql(hql):
    status = os.system(
        "hive -e \" %s\";" % hql)
    return status

def save_smoot_wenben(feature):
    #intersec_feature_path='/home/heqt/tencent/corr_feature/single_corr/for_smoot/'+feature+'_smoot.csv'
    count_feature_path='/home/heqt/tencent/corr_feature/single_corr/for_smoot/'+feature+'_train_count.csv'
    #df_cor_fea=pd.read_csv(intersec_feature_path,header=None,names=['uid','aid',feature+'single'])
    #because smoe label=-1 will cause null,need df_cor_count to fill null
    df_cor_count=pd.read_csv(count_feature_path,header=None,names=[feature+'single',feature+'dianji',feature+'baoguang'])
    hyper = HyperParam(1, 1)
    hyper.update_from_data_by_FPI(df_cor_count[feature+'baoguang'], df_cor_count[feature+'dianji'], 1000, 0.00000001)
    print "alpha and beta",hyper.alpha, hyper.beta
    df_cor_count[feature+'dianji']=df_cor_count[feature+'dianji'].map(lambda x: x+hyper.alpha)
    df_cor_count[feature+'baoguang']=df_cor_count[feature+'baoguang'].map(lambda x: x+hyper.alpha+hyper.beta) 
    df_cor_count[feature+'ctr']=df_cor_count[[feature+'dianji',feature+'baoguang']].apply(lambda x:1-x[0]/x[1],axis=1)

    #df_cor_fea=pd.merge(df_cor_fea,df_cor_count,how='left',on=['{0}single'.format(feature)])
    #print df_cor_fea.shape
    #print df_cor_fea.head(1)
    #df_cor_fea_2=df_cor_fea.groupby(by=['uid','aid'])[feature+'ctr'].apply(lambda x:np.exp(sum(np.log(x))))
   
    #df_cor_fea=df_cor_fea_2.reset_index()
    #save df_cor_fea,
    df_cor_count.to_csv("/home/heqt/tencent/corr_feature/single_corr/for_smoot/smooted_ctr/"+feature+"_smooted.csv",index=None,header=None)
    load_path="/home/heqt/tencent/corr_feature/single_corr/for_smoot/smooted_ctr/"+feature+"_smooted.csv"
    sql = '''
    load data local inpath '{0}' overwrite into table tmp.sigle_smoot_finish
    partition(factor_name = '{1}')
    '''.format(load_path,feature)
    res = execute_hql(sql)
def save_smoot_10jiaocha(feature):
    #intersec_feature_path='/home/heqt/tencent/corr_feature/single_corr/for_smoot/'+feature+'_smoot.csv'
    count_feature_path='/home/heqt/tencent/corr_feature/repair_aid/'+feature+'_corr_aid_smoot.csv'
    #df_cor_fea=pd.read_csv(intersec_feature_path,header=None,names=['uid','aid',feature+'single'])
    #because smoe label=-1 will cause null,need df_cor_count to fill null
    df_cor_count=pd.read_csv(count_feature_path,header=None,names=[feature,'aid',feature+'dianji',feature+'baoguang'])
    hyper = HyperParam(1, 1)
    hyper.update_from_data_by_FPI(df_cor_count[feature+'baoguang'], df_cor_count[feature+'dianji'], 1000, 0.00000001)
    print "alpha and beta",hyper.alpha, hyper.beta
    df_cor_count[feature+'dianji']=df_cor_count[feature+'dianji'].map(lambda x: x+hyper.alpha)
    df_cor_count[feature+'baoguang']=df_cor_count[feature+'baoguang'].map(lambda x: x+hyper.alpha+hyper.beta) 
    df_cor_count[feature+'ctr']=df_cor_count[[feature+'dianji',feature+'baoguang']].apply(lambda x:x[0]/x[1],axis=1)

    #df_cor_fea=pd.merge(df_cor_fea,df_cor_count,how='left',on=['{0}single'.format(feature)])
    #print df_cor_fea.shape
    #print df_cor_fea.head(1)
    #df_cor_fea_2=df_cor_fea.groupby(by=['uid','aid'])[feature+'ctr'].apply(lambda x:np.exp(sum(np.log(x))))
   
    #df_cor_fea=df_cor_fea_2.reset_index()
    #save df_cor_fea,
    df_cor_count.to_csv("/home/heqt/tencent/corr_feature/repair_aid/"+feature+"_ctr_smoot.csv",index=None,header=None)
    

def main():
    #paramas=["appidaction","appidinstall","interest1","interest2","interest3","interest4","interest5","kw1",
    #"kw2","kw3","topic1","topic2","topic3"]
    #下面是10个交叉，以前根据经验排除了一个
    paramas=["lbs","age","carrier","consumptionability","education","gender","os","ct","marriagestatus","house"]
    pool=Pool(10)
    pool.map(save_smoot_10jiaocha, paramas)



if __name__ == '__main__':
    main()