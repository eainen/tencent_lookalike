#! /usr/bin/env python2.7
#coding:utf-8

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
from sklearn.linear_model import LogisticRegression,SGDClassifier
#from scikitplot.metrics import plot_roc_curve
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.externals import joblib
import os
from datetime import datetime
import re
from operator import sub,add
#import matplotlib.pyplot as plt


one_hot_feature=['lbs','age','carrier','consumptionability','education','gender','house','advertiserid','campaignid', 'creativeid',
       'adcategoryid', 'creativesize','productid', 'producttype']
vector_feature=['appidaction','appidinstall','interest1','interest2','interest3','interest4','interest5',
'kw1','kw2','kw3','topic1','topic2','topic3','os','ct','marriagestatus']

data_x=pd.read_csv('/home/heqt/tencent/20180506/data_all_cut.csv')
print 'data_x_cut finish'
df_feature_map=pd.DataFrame()
LE=LabelEncoder()
for feature in one_hot_feature:
    try:
        data_x[feature]=LE.fit_transform(data_x[feature].map(np.int))

    except:
        data_x[feature]=LE.fit_transform(data_x[feature])
    #feature important mapping
    finally:
        df_tmp=pd.DataFrame(LE.classes_,columns=['val'])
        df_tmp['feature']='%s' %feature
        df_feature_map=pd.concat([df_feature_map,df_tmp])

print 'LabelEncoder finish'

#data_fit=data_x[data_x.label!=-1]
x_train=data_x[data_x.label!=-1]
data_y=x_train.pop('label')
data_test=data_x[data_x.label==-1]

#x_train,x_valid,y_train,y_valid= train_test_split(data_fit,data_y,test_size=0.3, random_state=2018)



data_x_train=pd.DataFrame()
#data_x_valid=pd.DataFrame()
data_x_test=pd.DataFrame()
OHE=OneHotEncoder()
for feature in one_hot_feature:
    OHE.fit(data_x[feature].values.reshape(-1,1))
    train_a=OHE.transform(x_train[feature].values.reshape(-1,1))
    #valid_a=OHE.transform(x_valid[feature].values.reshape(-1,1))
    test_a=OHE.transform(data_test[feature].values.reshape(-1,1))
    data_x_train=sparse.hstack((data_x_train,train_a))
    #data_x_valid=sparse.hstack((data_x_valid,valid_a))
    data_x_test=sparse.hstack((data_x_test,test_a))
print 'one_hot finish'

CVec=CountVectorizer(analyzer='word',token_pattern=r'(?u)\b\w+\b',tokenizer =lambda x: x.split(' '))
#CVec=CountVectorizer()
for feature in vector_feature:
    CVec.fit(data_x[feature])
    train_a=CVec.transform(x_train[feature])
    #valid_a=CVec.transform(x_valid[feature])
    test_a=CVec.transform(data_test[feature])
    data_x_train=sparse.hstack((data_x_train,train_a))
    #data_x_valid=sparse.hstack((data_x_valid,valid_a))
    data_x_test=sparse.hstack((data_x_test,test_a))
    df_tmp=pd.DataFrame(CVec.get_feature_names(),columns=['val'])
    #feature important mapping
    df_tmp['feature']='%s' %feature
    df_feature_map=pd.concat([df_feature_map,df_tmp])
print ' countvec finish'
save_path="/home/heqt/tencent/20180506/all/"
df_feature_map.to_csv(save_path+"feature_important_mapping_cut.csv")

sparse.save_npz(save_path+"data_x_train_cut.npz",data_x_train)
x_train.to_csv(save_path+"x_train_cut.csv",index=None)
y_train.to_csv(save_path+"data_y_train_cut.csv",index=None)

#sparse.save_npz(save_path+"data_x_valid_cut.npz",data_x_valid)
#x_valid.to_csv(save_path+"x_valid_cut.csv",index=None)
#y_valid.to_csv(save_path+"data_y_valid_cut.csv",index=None)

sparse.save_npz(save_path+"data_x_test_cut.npz",data_x_test)
data_test.to_csv(save_path+"data_test_cut.csv",index=None)
#result=data_test[['aid','uid']]
#result.to_csv(save_path+"result_cut.csv",index=None)


#与广告的2交叉组合,选取了3个交叉变量

corr_feature=["age","gender","house"]

x_train['uid']=x_train['uid'].map(np.int64)
x_train['aid']=x_train['aid'].map(np.int64)
#x_valid['uid']=x_valid['uid'].map(np.int64)
#x_valid['aid']=x_valid['aid'].map(np.int64)
data_test['aid']=data_test['aid'].map(np.int64)
data_test['uid']=data_test['uid'].map(np.int64)

for feature in corr_feature:
    corr_feature_path='/home/heqt/tencent/corr_feature/'+str(feature)+'_aid_zuhe.csv'
    df_cor_fea=pd.read_csv(corr_feature_path,header=None,names=['uid','aid',feature+'_aid'],dtype={0: np.int64,1: np.int64})
    print "df_cor_fea -----> shape",df_cor_fea.shape
    print df_cor_fea.head(1)
    x_train[feature+'_aid']=pd.merge(x_train,df_cor_fea,how='left',on=['uid','aid'])[feature+'_aid']
    #x_valid[feature+'_aid']=pd.merge(x_valid,df_cor_fea,how='left',on=['uid','aid'])[feature+'_aid']
    data_test[feature+'_aid']=pd.merge(data_test,df_cor_fea,how='left',on=['uid','aid'])[feature+'_aid']
x_train['qubie']=1
#x_valid['qubie']=0
data_test['qubie']=2
#data_x=pd.concat([x_train,x_valid])
data_x=pd.concat([x_train,data_test])
LE=LabelEncoder()
for feature in corr_feature:
    try:
        data_x[feature+'_aid']=LE.fit_transform(data_x[feature+'_aid'].map(np.int))
    except:
        data_x[feature+'_aid']=LE.fit_transform(data_x[feature+'_aid'])
print "LabelEncoder finish"
x_train=data_x[data_x.qubie==1]
#x_valid=data_x[data_x.qubie==0]
data_test=data_x[data_x.qubie==2]
data_x.pop('qubie'),x_train.pop('qubie'),data_test.pop('qubie')
OHE=OneHotEncoder()
for feature in corr_feature:
    OHE.fit(data_x[feature+'_aid'].values.reshape(-1,1))
    train_a=OHE.transform(x_train[feature+'_aid'].values.reshape(-1,1))
    #valid_a=OHE.transform(x_valid[feature+'_aid'].values.reshape(-1,1))
    test_a=OHE.transform(data_test[feature+'_aid'].values.reshape(-1,1))
    print 'train_a ----> one hot--->shape',train_a.shape
    data_x_train=sparse.hstack((data_x_train,train_a))
    #data_x_valid=sparse.hstack((data_x_valid,valid_a))
    data_x_test=sparse.hstack((data_x_test,test_a))
    print "data_x_train ---->shape",data_x_train.shape
    print feature
    print "----------------"

#组合统计量
#corr_feature=["age","gender","house"]
corr_feature=["age","gender","campaignid"]
x_train['uid']=x_train['uid'].map(np.int64)
x_train['aid']=x_train['aid'].map(np.int64)
#x_valid['uid']=x_valid['uid'].map(np.int64)
data_test['aid']=data_test['aid'].map(np.int64)
data_test['uid']=data_test['uid'].map(np.int64)

for feature in corr_feature:
    corr_feature_path='/home/heqt/tencent/corr_feature/'+str(feature)+'_aid_count.csv'
    df_cor_fea=pd.read_csv(corr_feature_path,header=None,names=['uid','aid','cnt'],dtype={0: np.int64,1: np.int64})
    print "df_cor_fea -----> shape",df_cor_fea.shape
    print df_cor_fea.head(1)
    train_a=pd.merge(x_train,df_cor_fea,how='left',on=['uid','aid'])['cnt']
    #valid_a=pd.merge(x_valid,df_cor_fea,how='left',on=['uid','aid'])['cnt']
    test_a=pd.merge(data_test,df_cor_fea,how='left',on=['uid','aid'])['cnt']
    print "train_a after shape -------->",train_a.shape
    print train_a.head(1)
    train_a=sparse.csr_matrix(train_a.values.reshape(-1,1))
    #valid_a=sparse.csr_matrix(valid_a.values.reshape(-1,1))
    test_a=sparse.csr_matrix(test_a.values.reshape(-1,1))
    data_x_train=sparse.hstack((data_x_train,train_a))
    #data_x_valid=sparse.hstack((data_x_valid,valid_a))
    data_x_test=sparse.hstack((data_x_test,test_a))
    #df_tmp=pd.DataFrame(['ctr'],columns=['val'])
    #feature important mapping
    #df_tmp['feature']='%s' %feature
    #df_feature_map=pd.concat([df_feature_map,df_tmp])
    print feature
    print "--------------------"

#用户特征组合广告的其他特征

corr_feature=["marriagestatus","age","consumptionability"]
aid_feature=["creativesize"]
aid_feature_age=["creativesize","adcategoryid"]
#为了labelencoder用的list
Encoder_list=[]
for feature in corr_feature:
    if feature=='age':
        source_feature=aid_feature_age
    else:
        source_feature=aid_feature
    for adfeature in source_feature:
        if feature not in ["os","ct","marriagestatus"]:
            corr_feature_path='/home/heqt/tencent/corr_feature/'+str(feature)+'/'+str(feature)+'_'+str(adfeature)+'_zuhe.csv'
        else :
            corr_feature_path='/home/heqt/tencent/corr_feature/'+str(feature)+'/'+str(feature)+'_'+str(adfeature)+'_aid_zuhe.csv'
        df_cor_fea=pd.read_csv(corr_feature_path,header=None,names=['uid','aid',feature+'_'+adfeature],dtype={0: np.int64,1: np.int64})
        print "df_cor_fea -----> shape",df_cor_fea.shape
        print df_cor_fea.head(1)
        x_train[feature+'_'+adfeature]=pd.merge(x_train,df_cor_fea,how='left',on=['uid','aid'])[feature+'_'+adfeature]
        #x_valid[feature+'_'+adfeature]=pd.merge(x_valid,df_cor_fea,how='left',on=['uid','aid'])[feature+'_'+adfeature]
        data_test[feature+'_'+adfeature]=pd.merge(data_test,df_cor_fea,how='left',on=['uid','aid'])[feature+'_'+adfeature]
        Encoder_list.append(feature+'_'+adfeature)
x_train['qubie']=1
#x_valid['qubie']=0
data_test['qubie']=2
data_x=pd.concat([x_train,data_test])
#data_x=pd.concat([data_x,data_test])
LE=LabelEncoder()
for feature in Encoder_list:
    try:
        data_x[feature]=LE.fit_transform(data_x[feature].map(np.int))
    except:
        data_x[feature]=LE.fit_transform(data_x[feature])
print "LabelEncoder finish"
x_train=data_x[data_x.qubie==1]
#x_valid=data_x[data_x.qubie==0]
data_test=data_x[data_x.qubie==2]
data_x.pop('qubie'),x_train.pop('qubie'),data_test.pop('qubie')
OHE=OneHotEncoder()
#data_x_train_1=data_x_train.copy()
#data_x_valid_1=data_x_valid.copy()
#df_AUC=pd.DataFrame(columns=['var_name','AUC_0','AUC_1'])
for feature in Encoder_list:
    OHE.fit(data_x[feature].values.reshape(-1,1))
    train_a=OHE.transform(x_train[feature].values.reshape(-1,1))
    #valid_a=OHE.transform(x_valid[feature].values.reshape(-1,1))
    test_a=OHE.transform(data_test[feature].values.reshape(-1,1))
    print 'train_a ----> one hot--->shape',train_a.shape
    data_x_train=sparse.hstack((data_x_train,train_a))
    #data_x_valid=sparse.hstack((data_x_valid,valid_a))
    data_x_test=sparse.hstack((data_x_test,test_a))
    print "data_x_train ---->shape",data_x_train.shape
    print feature
    print "----------------"

#用户特征相交,只选了os_gender
corr_feature=['os_gender']
#为了labelencoder用的list
for feature in corr_feature:
    corr_feature_path='/home/heqt/tencent/corr_feature/'+str(feature)+'_zuhe.csv'
    df_cor_fea=pd.read_csv(corr_feature_path,header=None,names=['uid','aid',feature],dtype={0: np.int64,1: np.int64})
    print "df_cor_fea -----> shape",df_cor_fea.shape
    print df_cor_fea.head(1)
    x_train[feature]=pd.merge(x_train,df_cor_fea,how='left',on=['uid','aid'])[feature]
    #x_valid[feature]=pd.merge(x_valid,df_cor_fea,how='left',on=['uid','aid'])[feature]
    data_test[feature]=pd.merge(data_test,df_cor_fea,how='left',on=['uid','aid'])[feature]
x_train['qubie']=1
#x_valid['qubie']=0
data_test['qubie']=2
data_x=pd.concat([x_train,data_test])
#data_x=pd.concat([data_x,data_test])
LE=LabelEncoder()
for feature in corr_feature:
    try:
        data_x[feature]=LE.fit_transform(data_x[feature].map(np.int))
    except:
        data_x[feature]=LE.fit_transform(data_x[feature])
print "LabelEncoder finish"
x_train=data_x[data_x.qubie==1]
#x_valid=data_x[data_x.qubie==0]
data_test=data_x[data_x.qubie==2]
data_x.pop('qubie'),x_train.pop('qubie'),data_test.pop('qubie')
OHE=OneHotEncoder()
#data_x_train_1=data_x_train.copy()
#data_x_valid_1=data_x_valid.copy()
#df_AUC=pd.DataFrame(columns=['var_name','AUC_0','AUC_1'])
for feature in corr_feature:
    OHE.fit(data_x[feature].values.reshape(-1,1))
    train_a=OHE.transform(x_train[feature].values.reshape(-1,1))
    #valid_a=OHE.transform(x_valid[feature].values.reshape(-1,1))
    test_a=OHE.transform(data_test[feature].values.reshape(-1,1))
    print 'train_a ----> one hot--->shape',train_a.shape
    data_x_train=sparse.hstack((data_x_train,train_a))
    #data_x_valid=sparse.hstack((data_x_valid,valid_a))
    data_x_test=sparse.hstack((data_x_test,test_a))

    print "data_x_train ---->shape",data_x_train.shape
    print feature
    print "----------------"


#3交叉组合



corr_feature=["age_gender_aid","age_consumptionability_aid","age_education_aid","gender_house_aid","gender_consumptionability_aid",
             "gender_marriagestatus_aid","gender_education_aid"]
#为了labelencoder用的list
for feature in corr_feature:
    corr_feature_path='/home/heqt/tencent/3corr_feature/'+str(feature)+'_zuhe.csv'
    df_cor_fea=pd.read_csv(corr_feature_path,header=None,names=['uid','aid',feature],dtype={0: np.int64,1: np.int64})
    print "df_cor_fea -----> shape",df_cor_fea.shape
    print df_cor_fea.head(1)
    x_train[feature]=pd.merge(x_train,df_cor_fea,how='left',on=['uid','aid'])[feature]
    #x_valid[feature]=pd.merge(x_valid,df_cor_fea,how='left',on=['uid','aid'])[feature]
    data_test[feature]=pd.merge(data_test,df_cor_fea,how='left',on=['uid','aid'])[feature]
x_train['qubie']=1
#x_valid['qubie']=0
data_test['qubie']=2
data_x=pd.concat([x_train,data_test])
#data_x=pd.concat([data_x,data_test])
LE=LabelEncoder()
for feature in corr_feature:
    try:
        data_x[feature]=LE.fit_transform(data_x[feature].map(np.int))
    except:
        data_x[feature]=LE.fit_transform(data_x[feature])
print "LabelEncoder finish"
x_train=data_x[data_x.qubie==1]
#x_valid=data_x[data_x.qubie==0]
data_test=data_x[data_x.qubie==2]
data_x.pop('qubie'),x_train.pop('qubie'),data_test.pop('qubie')
OHE=OneHotEncoder()
#data_x_train_1=data_x_train.copy()
#data_x_valid_1=data_x_valid.copy()
#df_AUC=pd.DataFrame(columns=['var_name','AUC_0','AUC_1'])
for feature in corr_feature:
    OHE.fit(data_x[feature].values.reshape(-1,1))
    train_a=OHE.transform(x_train[feature].values.reshape(-1,1))
    #valid_a=OHE.transform(x_valid[feature].values.reshape(-1,1))
    test_a=OHE.transform(data_test[feature].values.reshape(-1,1))
    print 'train_a ----> one hot--->shape',train_a.shape
    data_x_train=sparse.hstack((data_x_train,train_a))
    #data_x_valid=sparse.hstack((data_x_valid,valid_a))
    data_x_test=sparse.hstack((data_x_test,test_a))
    print "data_x_train ---->shape",data_x_train.shape
    print feature
    print "----------------"


#尝试age与广告其他的组合特征
corr_feature=["age_gender_creativesize","age_consumptionability_creativesize","age_education_creativesize",
              "age_gender_adcategoryid","age_consumptionability_adcategoryid","age_education_adcategoryid",
              "age_gender_productid","age_consumptionability_productid","age_education_productid","age_gender_producttype",
              "age_consumptionability_producttype","age_education_producttype"]
#为了labelencoder用的list
for feature in corr_feature:
    corr_feature_path='/home/heqt/tencent/3corr_feature/'+str(feature)+'_zuhe.csv'
    df_cor_fea=pd.read_csv(corr_feature_path,header=None,names=['uid','aid',feature],dtype={0: np.int64,1: np.int64})
    print "df_cor_fea -----> shape",df_cor_fea.shape
    print df_cor_fea.head(1)
    x_train[feature]=pd.merge(x_train,df_cor_fea,how='left',on=['uid','aid'])[feature]
    #x_valid[feature]=pd.merge(x_valid,df_cor_fea,how='left',on=['uid','aid'])[feature]
    data_test[feature]=pd.merge(data_test,df_cor_fea,how='left',on=['uid','aid'])[feature]

x_train['qubie']=1
#x_valid['qubie']=0
data_test['qubie']=2
data_x=pd.concat([x_train,data_test])
#data_x=pd.concat([data_x,data_test])
LE=LabelEncoder()
for feature in corr_feature:
    try:
        data_x[feature]=LE.fit_transform(data_x[feature].map(np.int))
    except:
        data_x[feature]=LE.fit_transform(data_x[feature])
print "LabelEncoder finish"
x_train=data_x[data_x.qubie==1]
#x_valid=data_x[data_x.qubie==0]
data_test=data_x[data_x.qubie==2]
data_x.pop('qubie'),x_train.pop('qubie'),data_test.pop('qubie')
OHE=OneHotEncoder()
#data_x_train_1=data_x_train.copy()
#data_x_valid_1=data_x_valid.copy()
#df_AUC_age=pd.DataFrame(columns=['var_name','AUC_0','AUC_1'])
for feature in corr_feature:
    OHE.fit(data_x[feature].values.reshape(-1,1))
    train_a=OHE.transform(x_train[feature].values.reshape(-1,1))
    #valid_a=OHE.transform(x_valid[feature].values.reshape(-1,1))
    test_a=OHE.transform(data_test[feature].values.reshape(-1,1))
    print 'train_a ----> one hot--->shape',train_a.shape
    data_x_train=sparse.hstack((data_x_train,train_a))
    #data_x_valid=sparse.hstack((data_x_valid,valid_a))
    data_x_test=sparse.hstack((data_x_test,test_a))
    print "data_x_train ---->shape",data_x_train.shape
    print feature
    print "----------------"

#尝试gender与广告的其他组合特征:

corr_feature=["gender_house_creativesize","gender_consumptionability_creativesize","gender_marriagestatus_creativesize",
              "gender_education_creativesize"]

for feature in corr_feature:
    corr_feature_path='/home/heqt/tencent/3corr_feature/'+str(feature)+'_zuhe.csv'
    df_cor_fea=pd.read_csv(corr_feature_path,header=None,names=['uid','aid',feature],dtype={0: np.int64,1: np.int64})
    print "df_cor_fea -----> shape",df_cor_fea.shape
    print df_cor_fea.head(1)
    x_train[feature]=pd.merge(x_train,df_cor_fea,how='left',on=['uid','aid'])[feature]
    #x_valid[feature]=pd.merge(x_valid,df_cor_fea,how='left',on=['uid','aid'])[feature]
    data_test[feature]=pd.merge(data_test,df_cor_fea,how='left',on=['uid','aid'])[feature]
x_train['qubie']=1
#x_valid['qubie']=0
data_test['qubie']=2
data_x=pd.concat([x_train,data_test])
#data_x=pd.concat([data_x,data_test])
LE=LabelEncoder()
for feature in corr_feature:
    try:
        data_x[feature]=LE.fit_transform(data_x[feature].map(np.int))
    except:
        data_x[feature]=LE.fit_transform(data_x[feature])
print "LabelEncoder finish"
x_train=data_x[data_x.qubie==1]
#x_valid=data_x[data_x.qubie==0]
data_test=data_x[data_x.qubie==2]
data_x.pop('qubie'),x_train.pop('qubie'),data_test.pop('qubie')
OHE=OneHotEncoder()
#data_x_train_1=data_x_train.copy()
#data_x_valid_1=data_x_valid.copy()
#df_AUC=pd.DataFrame(columns=['var_name','AUC_0','AUC_1'])
for feature in corr_feature:
    OHE.fit(data_x[feature].values.reshape(-1,1))
    train_a=OHE.transform(x_train[feature].values.reshape(-1,1))
    #valid_a=OHE.transform(x_valid[feature].values.reshape(-1,1))
    test_a=OHE.transform(data_test[feature].values.reshape(-1,1))
    print 'train_a ----> one hot--->shape',train_a.shape
    data_x_train=sparse.hstack((data_x_train,train_a))
    #data_x_valid=sparse.hstack((data_x_valid,valid_a))
    data_x_test=sparse.hstack((data_x_test,test_a))
    print "data_x_train ---->shape",data_x_train.shape
    print feature
    print "----------------"


#aid,lbs,campaignid,producttype的历史点击率,不同广告在不同人群的投放量,经过one_hot修改后的

#corr_feature=["aid_toufang","aid","lbs","campaignid","producttype"]
corr_feature=["aid_toufang","aid"]
data_test['aid']=data_test['aid'].map(np.int64)
data_test['uid']=data_test['uid'].map(np.int64)
for feature in corr_feature:
    corr_feature_path='/home/heqt/tencent/corr_feature/'+str(feature)+'_uid_count_onehot.csv'
    if feature=="aid_toufang":
        feature="aid"
        df_cor_fea=pd.read_csv(corr_feature_path,header=None,names=[feature,'corr_ctr'],dtype={0: np.int64})
        train_a=pd.merge(x_train,df_cor_fea,how='left',on=[feature])['corr_ctr']
        #valid_a=pd.merge(x_valid,df_cor_fea,how='left',on=[feature])['corr_ctr']
        test_a=pd.merge(data_test,df_cor_fea,how='left',on=[feature])['corr_ctr']

        train_a=sparse.csr_matrix(train_a.values.reshape(-1,1))
        #valid_a=sparse.csr_matrix(valid_a.values.reshape(-1,1))
        test_a=sparse.csr_matrix(test_a.values.reshape(-1,1))
        data_x_train=sparse.hstack((data_x_train,train_a))
        #data_x_valid=sparse.hstack((data_x_valid,valid_a))
        data_x_test=sparse.hstack((data_x_test,test_a))
    else:
        df_cor_fea=pd.read_csv(corr_feature_path,header=None,names=['uid','aid','corr_ctr'],dtype={0: np.int64,1: np.int64})
        print "df_cor_fea -----> shape",df_cor_fea.shape
        print df_cor_fea.head(1)
        train_a=pd.merge(x_train,df_cor_fea,how='left',on=['uid','aid'])['corr_ctr']
        #valid_a=pd.merge(x_valid,df_cor_fea,how='left',on=['uid','aid'])['corr_ctr']
        test_a=pd.merge(data_test,df_cor_fea,how='left',on=['uid','aid'])['corr_ctr']
        print "train_a after shape -------->",train_a.shape
        print train_a.head(1)
        train_a=sparse.csr_matrix(train_a.values.reshape(-1,1))
        #valid_a=sparse.csr_matrix(valid_a.values.reshape(-1,1))
        test_a=sparse.csr_matrix(test_a.values.reshape(-1,1))
        data_x_train=sparse.hstack((data_x_train,train_a))
        #data_x_valid=sparse.hstack((data_x_valid,valid_a))
        data_x_test=sparse.hstack((data_x_test,test_a))
    #df_tmp=pd.DataFrame(['ctr'],columns=['val'])
    #feature important mapping
    #df_tmp['feature']='%s' %feature
    #df_feature_map=pd.concat([df_feature_map,df_tmp])
    print feature
    print "--------------------"
#最后一波
corr_feature=["advertiserid","campaignid","creativesize","adcategoryid","productid","producttype"]
aid_feature=["age","gender","house","lbs","carrier","education","consumptionability"]
#array([329, 123,  28,   9,   7,   0,   1,   0,   0,   0, 
#       636, 183,  50,   15,   2,   2,   0,   0,   0,   0, 
#       186,  65,  20,  10,   2,   0,0,   0,   0,   0, 
#       298,  81,  13,   9,   6,   0,   0,   0,   0,0, 
#       161,  33,   5,   4,   0,   0,   1,   0,   0,   0, 
#       107,  32, 9,   4,   4,   0,   0,   0,   0,   0])

#为了labelencoder用的list
x_train['uid']=x_train['uid'].map(np.int64)
x_train['aid']=x_train['aid'].map(np.int64)
data_test['aid']=data_test['aid'].map(np.int64)
data_test['uid']=data_test['uid'].map(np.int64)
for feature in corr_feature:
    for adfeature in aid_feature:
        corr_feature_path='/home/heqt/tencent/tongjitezheng/'+str(feature)+'_'+str(adfeature)+'_tongji.csv'
        df_cor_fea=pd.read_csv(corr_feature_path,header=None,names=['uid','aid',feature+'_'+adfeature],dtype={0: np.int64,1: np.int64})
        print "df_cor_fea -----> shape",df_cor_fea.shape
        print df_cor_fea.head(1)
        train_a=pd.merge(x_train,df_cor_fea,how='left',on=['uid','aid'])[feature+'_'+adfeature]
        #valid_a=pd.merge(x_valid,df_cor_fea,how='left',on=['uid','aid'])[feature+'_'+adfeature]
        test_a=pd.merge(data_test,df_cor_fea,how='left',on=['uid','aid'])[feature+'_'+adfeature]
        train_a=sparse.csr_matrix(train_a.values.reshape(-1,1))
        #valid_a=sparse.csr_matrix(valid_a.values.reshape(-1,1))
        test_a=sparse.csr_matrix(test_a.values.reshape(-1,1))
        data_x_train=sparse.hstack((data_x_train,train_a))
        #data_x_valid=sparse.hstack((data_x_valid,valid_a))
        data_x_test=sparse.hstack((data_x_test,test_a))



corr_feature=["lbs","carrier","consumptionability","education","gender","os","ct","marriagestatus"]
#x_train['uid']=x_train['uid'].map(np.int64)
#x_train['aid']=x_train['aid'].map(np.int64)
#x_valid['uid']=x_valid['uid'].map(np.int64)
#x_valid['aid']=x_valid['aid'].map(np.int64)
#data_test['aid']=data_test['aid'].map(np.int64)
#data_test['uid']=data_test['uid'].map(np.int64)
for feature in corr_feature:
    corr_feature_path='/home/heqt/tencent/corr_feature/'+str(feature)+'_aid_count.csv'
    df_cor_fea=pd.read_csv(corr_feature_path,header=None,names=['uid','aid','corr_ctr'],dtype={0: np.int64,1: np.int64})
    print "df_cor_fea -----> shape",df_cor_fea.shape
    print df_cor_fea.head(1)
    train_a=pd.merge(x_train,df_cor_fea,how='left',on=['uid','aid'])['corr_ctr']
    #valid_a=pd.merge(x_valid,df_cor_fea,how='left',on=['uid','aid'])['corr_ctr']
    test_a=pd.merge(data_test,df_cor_fea,how='left',on=['uid','aid'])['corr_ctr']
    print "train_a after shape -------->",train_a.shape
    print train_a.head(1)
    train_a=sparse.csr_matrix(train_a.values.reshape(-1,1))
    valid_a=sparse.csr_matrix(valid_a.values.reshape(-1,1))
    #test_a=sparse.csr_matrix(test_a.values.reshape(-1,1))
    data_x_train=sparse.hstack((data_x_train,train_a))
    #data_x_valid=sparse.hstack((data_x_valid,valid_a))
    data_x_test=sparse.hstack((data_x_test,test_a))
    #df_tmp=pd.DataFrame(['ctr'],columns=['val'])
    #feature important mapping
    #df_tmp['feature']='%s' %feature
    #df_feature_map=pd.concat([df_feature_map,df_tmp])
    print feature
    print "--------------------"