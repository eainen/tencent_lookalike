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


def sub_to_other(x):
    x_split=x.split(" ")
    set_tmp=set(map(int,x_split))
    intersec=list(set_tmp & target_set)
    if intersec:
        for target in intersec:
            x_split[x_split.index(str(target))]='other'  
    return ' '.join(x_split)

#删除缺失率高的值，再建立个模型
save_path="/home/heqt/tencent/remove_hight_na/"

data_x=pd.read_csv('/home/heqt/tencent/data_all.csv')
data_x.drop(labels=['appidaction','appidinstall','interest3','interest4','kw3','topic3'],axis=1, inplace=True )
data_x.fillna('-1',inplace=True)


#fit_model
one_hot_feature=['lbs','age','carrier','consumptionability','education','gender','house','os','ct','marriagestatus',
'advertiserid','campaignid', 'creativeid','adcategoryid','productid', 'producttype']
vector_feature=['interest1','interest2','interest5','kw1','kw2','topic1','topic2']
bin_feature=['kw1','kw2','topic1','topic2']

paramas={
'appidaction':266,
'appidinstall':2875,
'kw1':5690,
'kw2':3698,
'kw3':224,
'topic1':6105,
'topic2':5496,
'topic3':309
}
for feature in bin_feature:
    file_path='/home/heqt/tencent/countvec/'+str(feature)+'.csv'
    df_cut=pd.read_csv(file_path,header=None,names=['feature_value','import_value'])
    df_cut_m1=df_cut.loc[df_cut['feature_value']==-1]
    df_cut_nm1=df_cut.loc[df_cut['feature_value']!=-1]
    top_N=paramas[feature]
    df_cut_indic=df_cut_nm1.sort_values(by='import_value',ascending=False)[top_N:].feature_value.tolist()
    target_set=set(df_cut_indic)
    data_x[feature]=data_x[feature].apply(sub_to_other)
    print feature

data_x.to_csv(save_path+"data_x_cut_drop.csv",index=None)
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

data_fit=data_x[data_x.label!=-1]
data_y=data_fit.pop('label')
data_test=data_x[data_x.label==-1]

x_train,x_valid,y_train,y_valid= train_test_split(data_fit,data_y,test_size=0.3, random_state=20180515)



data_x_train=x_train[['creativesize']]
data_x_valid=x_valid[['creativesize']]
data_x_test=data_test[['creativesize']]
OHE=OneHotEncoder()
for feature in one_hot_feature:
    OHE.fit(data_x[feature].values.reshape(-1,1))
    train_a=OHE.transform(x_train[feature].values.reshape(-1,1))
    valid_a=OHE.transform(x_valid[feature].values.reshape(-1,1))
    test_a=OHE.transform(data_test[feature].values.reshape(-1,1))
    data_x_train=sparse.hstack((data_x_train,train_a))
    data_x_valid=sparse.hstack((data_x_valid,valid_a))
    data_x_test=sparse.hstack((data_x_test,test_a))
print 'one_hot finish'

CVec=CountVectorizer(analyzer='word',token_pattern=r'(?u)\b\w+\b',tokenizer =lambda x: x.split(' '))
#CVec=CountVectorizer()
for feature in vector_feature:
    CVec.fit(data_x[feature])
    train_a=CVec.transform(x_train[feature])
    valid_a=CVec.transform(x_valid[feature])
    test_a=CVec.transform(data_test[feature])
    data_x_train=sparse.hstack((data_x_train,train_a))
    data_x_valid=sparse.hstack((data_x_valid,valid_a))
    data_x_test=sparse.hstack((data_x_test,test_a))
    df_tmp=pd.DataFrame(CVec.get_feature_names(),columns=['val'])
    #feature important mapping
    df_tmp['feature']='%s' %feature
    df_feature_map=pd.concat([df_feature_map,df_tmp])
print ' countvec finish'
df_feature_map.to_csv(save_path+"feature_important_mapping_cut.csv")

sparse.save_npz(save_path+"data_x_train_cut_drop.npz",data_x_train)
x_train.to_csv(save_path+"x_train_cut_drop.csv",index=None)
y_train.to_csv(save_path+"data_y_train_cut_drop.csv",index=None)

sparse.save_npz(save_path+"data_x_valid_cut_drop.npz",data_x_valid)
x_valid.to_csv(save_path+"x_valid_cut_drop.csv",index=None)
y_valid.to_csv(save_path+"data_y_valid_cut_drop.csv",index=None)

sparse.save_npz(save_path+"data_x_test_cut_drop.npz",data_x_test)
data_test.to_csv(save_path+"data_test_cut_drop.csv",index=None)