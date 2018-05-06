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

run_day=datetime.now().strftime('%m%d%H%M%S')
save_path="/home/heqt/tencent/"+run_day+"/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

#data_fit=pd.read_csv('/home/heqt/tencent/train_all.csv')
#data_test=pd.read_csv('/home/heqt/tencent/test1_all.csv')
#data_fit.loc[data_fit['label']==-1,'label']=0
#data_test.insert(2,'label',-1)
#data_x=pd.concat([data_fit,data_test])
#data_x.to_csv('/home/heqt/tencent/data_all.csv',index=None)
data_x=pd.read_csv('/home/heqt/tencent/data_all.csv')
data_x.fillna('-1',inplace=True)


#fit_model
#one_hot_feature=['lbs','age','carrier','consumptionability','education','gender','house','os','ct','marriagestatus','advertiserid','campaignid', 'creativeid',
#      'adcategoryid', 'creativesize','productid', 'producttype']
one_hot_feature=['producttype','age','creativesize','gender','consumptionability','os','marriagestatus','adcategoryid','carrier','ct','education']
#vector_feature=['appidaction','appidinstall','interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3','topic1','topic2','topic3']
vector_feature=['interest2']

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
#data_test=data_x[data_x.label==-1]

x_train,x_valid_2,y_train,y_valid= train_test_split(data_fit,data_y,test_size=0.2, random_state=20180505)
x_train,x_valid,y_train,y_valid= train_test_split(x_valid_2,y_valid,test_size=0.3, random_state=20180506)

data_x_train=pd.DataFrame()
data_x_valid=pd.DataFrame()
#data_x_test=pd.DataFrame()
OHE=OneHotEncoder()
for feature in one_hot_feature:
    OHE.fit(x_valid_2[feature].values.reshape(-1,1))
    train_a=OHE.transform(x_train[feature].values.reshape(-1,1))
    valid_a=OHE.transform(x_valid[feature].values.reshape(-1,1))
    #test_a=OHE.transform(data_test[feature].values.reshape(-1,1))
    data_x_train=sparse.hstack((data_x_train,train_a))
    data_x_valid=sparse.hstack((data_x_valid,valid_a))
    #data_x_test=sparse.hstack((data_x_test,test_a))
print 'one_hot finish'
CVec=CountVectorizer(analyzer='word',token_pattern=r'(?u)\b\w+\b',tokenizer =lambda x: x.split(' '))
#CVec=CountVectorizer()
for feature in vector_feature:
    CVec.fit(x_valid_2[feature])
    train_a=CVec.transform(x_train[feature])
    valid_a=CVec.transform(x_valid[feature])
    #test_a=CVec.transform(data_test[feature])
    data_x_train=sparse.hstack((data_x_train,train_a))
    data_x_valid=sparse.hstack((data_x_valid,valid_a))
    #data_x_test=sparse.hstack((data_x_test,test_a))
    df_tmp=pd.DataFrame(CVec.get_feature_names(),columns=['val'])
    #feature important mapping
    df_tmp['feature']='%s' %feature
    df_feature_map=pd.concat([df_feature_map,df_tmp])
print ' countvec finish'
df_feature_map.to_csv(save_path+"feature_important_mapping.csv")

#sparse.save_npz(save_path+"data_x_train.npz",data_x_train)
#y_train.to_csv(save_path+"data_y_train.csv",index=None)

#sparse.save_npz(save_path+"data_x_valid.npz",data_x_valid)
#y_valid.to_csv(save_path+"data_y_valid.csv",index=None)

#sparse.save_npz(save_path+"data_x_test.npz",data_x_test)
#result=data_test[['aid','uid']]
#result.to_csv(save_path+"result.csv",index=None)

gbm_clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=300, objective='binary',
        subsample_freq=1,
        learning_rate=0.02, min_child_weight=50,random_state=20180505,n_jobs=3)

parama={'bin_feature':'appidaction',
        'top_N':100
        }

def aotu_bin(parama):
    global target_set
    file_path='/home/heqt/tencent/countvec/'+str(parama['bin_feature'])+'.csv'
    df_cut=pd.read_csv(file_path,header=None,names=['feature_value','import_value'])
    df_cut_m1=df_cut.loc[df_cut['feature_value']==-1]
    df_cut_nm1=df_cut.loc[df_cut['feature_value']!=-1]
    #prevente top_N less than default
    top_N=min(parama['top_N'],df_cut_nm1.shape[0])
    print top_N
    steps=0
    df_cut_inf=pd.DataFrame(columns=['cut_num','valid_0','valid_1'])
    while top_N>0:
        if steps==0:
            CVec.fit(x_valid_2[parama['bin_feature']])
            train_a=CVec.transform(x_train[parama['bin_feature']])
            valid_a=CVec.transform(x_valid[parama['bin_feature']])
            data_x_train_top=sparse.hstack((data_x_train,train_a))
            data_x_valid_top=sparse.hstack((data_x_valid,valid_a))
        elif steps ==1:
            df_cut_indic=df_cut_nm1.sort_values(by='import_value',ascending=False)[top_N:].feature_value.tolist()
            target_set=set(df_cut_indic)
            x_valid_2[parama['bin_feature']]=x_valid_2[parama['bin_feature']].apply(sub_to_other)
            x_train[parama['bin_feature']]=x_train[parama['bin_feature']].apply(sub_to_other)
            x_valid[parama['bin_feature']]=x_valid[parama['bin_feature']].apply(sub_to_other)
            CVec.fit(x_valid_2[parama['bin_feature']])
            train_a=CVec.transform(x_train[parama['bin_feature']])
            valid_a=CVec.transform(x_valid[parama['bin_feature']])
            data_x_train_top=sparse.hstack((data_x_train,train_a))
            data_x_valid_top=sparse.hstack((data_x_valid,valid_a))
        else:
            df_cut_indic=df_cut_nm1.sort_values(by='import_value',ascending=False).iloc[top_N].feature_value
            parten='\\b'+str(df_cut_indic)+'\\b'
            x_valid_2[parama['bin_feature']]=x_valid_2[parama['bin_feature']].apply(lambda x: re.sub(parten,'other',x))
            x_train[parama['bin_feature']]=x_train[parama['bin_feature']].apply(lambda x: re.sub(parten,'other',x))
            x_valid[parama['bin_feature']]=x_valid[parama['bin_feature']].apply(lambda x: re.sub(parten,'other',x))
            CVec.fit(x_valid_2[parama['bin_feature']])
            train_a=CVec.transform(x_train[parama['bin_feature']])
            valid_a=CVec.transform(x_valid[parama['bin_feature']])
            data_x_train_top=sparse.hstack((data_x_train,train_a))
            data_x_valid_top=sparse.hstack((data_x_valid,valid_a))

        eval_list=[(data_x_train_top,y_train),(data_x_valid_top,y_valid)]
        gbm_clf.fit(data_x_train,y_train,eval_set=eval_list,eval_metric ='auc',early_stopping_rounds =100)

        if top_N%10==0:
            print 'gmb_finish','----->',top_N
        #save cut_information
        df_inf_tmp=pd.DataFrame([top_N+1],columns=['cut_num'])
        df_inf_tmp['valid_0']=gbm_clf.best_score_['valid_0']['auc']
        df_inf_tmp['valid_1']=gbm_clf.best_score_['valid_1']['auc']
        df_cut_inf=pd.concat([df_cut_inf,df_inf_tmp])
        steps+=1
        top_N-=1
    df_cut_inf.to_csv(save_path+str(parama['bin_feature'])+"cut_information.csv",index=None)


#after cut,feature ennerging

def sub_to_other(x):
    x_split=x.split(" ")
    set_tmp=set(map(int,x_split))
    intersec=list(set_tmp & target_set)
    if intersec:
        for target in intersec:
            x_split[x_split.index(str(target))]='other'  
    return ' '.join(x_split)


save_path="/home/heqt/tencent/"+"ennerging_feature"+"/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

data_fit=pd.read_csv('/home/heqt/tencent/train_all.csv')
data_fit.loc[data_fit['label']==-1,'label']=0
data_y=data_fit.pop('label')
x_train,x_valid_2,y_train,y_valid= train_test_split(data_fit,data_y,test_size=0.1, random_state=201805)

x_valid_2.insert(2,'label',y_valid)
data_fit=x_valid_2
data_test=pd.read_csv('/home/heqt/tencent/test1_all.csv')
data_test.insert(2,'label',-1)
data_x=pd.concat([data_fit,data_test])
data_x.fillna('-1',inplace=True)
data_x.to_csv(save_path+"data_x_80W.csv",index=None)

one_hot_feature=['lbs','age','carrier','consumptionability','education','gender','house','os','ct','marriagestatus','advertiserid','campaignid', 'creativeid',
       'adcategoryid', 'creativesize','productid', 'producttype']
vector_feature=['appidaction','appidinstall','interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3','topic1','topic2','topic3']
bin_feature=['appidaction','appidinstall','kw1','kw2','kw3','topic1','topic2','topic3']

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


data_x.to_csv(save_path+"data_x_cut_80W.csv",index=None)


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

x_train,x_valid,y_train,y_valid= train_test_split(data_fit,data_y,test_size=0.3, random_state=20180506)

data_x_train=pd.DataFrame()
data_x_valid=pd.DataFrame()
data_x_test=pd.DataFrame()
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

sparse.save_npz(save_path+"data_x_train_cut.npz",data_x_train)
x_train.to_csv(save_path+"x_train_cut.csv",index=None)
y_train.to_csv(save_path+"data_y_train_cut.csv",index=None)

sparse.save_npz(save_path+"data_x_valid_cut.npz",data_x_valid)
x_valid.to_csv(save_path+"x_valid_cut.csv",index=None)
y_valid.to_csv(save_path+"data_y_valid_cut.csv",index=None)

sparse.save_npz(save_path+"data_x_test_cut.npz",data_x_test)
result=data_test[['aid','uid']]
result.to_csv(save_path+"result_cut.csv",index=None)
print 'save_cut_finish'

eval_list=[(data_x_train,y_train),(data_x_valid,y_valid)]

gbm_clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=1000, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.02, min_child_weight=50,random_state=20180506,n_jobs=7)

gbm_clf.fit(data_x_train,y_train,eval_set=eval_list,eval_metric ='auc',early_stopping_rounds =100)
joblib.dump(gbm_clf, '/home/heqt/jupyter_project/model/gbm_clf_cnt_80W.pkl')


#交叉类特征

#house特征的-1取值情况影响不大
corr_feature=["lbs","age","carrier","education","gender","os","ct","marriagestatus","house"]

for feature in corr_feature:
    corr_feature_path='/home/heqt/tencent/corr_feature/'+str(feature)+'_corr_aid.csv'
    df_cor_fea=pd.read_csv(corr_feature_path,header=None,names=[feature,'aid','label','feacnt','cnt','corr_ctr'])
    df_cor_fea['corr_ctr']=df_cor_fea['corr_ctr']*100.0
    print df_cor_fea.shape
    print df_cor_fea.head(1)
    df_cor_fea=df_cor_fea.drop(labels=['label','feacnt','cnt'],axis=1)
    train_a=pd.merge(x_train,df_cor_fea,how='left',on=[feature,'aid'])['corr_ctr']
    valid_a=pd.merge(x_valid,df_cor_fea,how='left',on=[feature,'aid'])['corr_ctr']
    print train_a.shape
    print train_a.head(1)
    train_a=sparse.csr_matrix(train_a.values.reshape(-1,1))
    valid_a=sparse.csr_matrix(valid_a.values.reshape(-1,1))
    data_x_train=sparse.hstack((data_x_train,train_a))
    data_x_valid=sparse.hstack((data_x_valid,valid_a))
    df_tmp=pd.DataFrame(['ctr'],columns=['val'])
    #feature important mapping
    df_tmp['feature']='%s' %feature
    df_feature_map=pd.concat([df_feature_map,df_tmp])
    print feature




df_feature_map.to_csv(save_path+"feature_important_mapping_cut_corr.csv")
sparse.save_npz(save_path+"data_x_train_cut_corr.npz",data_x_train)
sparse.save_npz(save_path+"data_x_valid_cut_corr.npz",data_x_valid)

eval_list=[(data_x_train,y_train),(data_x_valid,y_valid)]

gbm_clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=1000, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.02, min_child_weight=50,random_state=20180506,n_jobs=7)

gbm_clf.fit(data_x_train,y_train,eval_set=eval_list,eval_metric ='auc',early_stopping_rounds =100)
joblib.dump(gbm_clf, '/home/heqt/jupyter_project/model/gbm_clf_cnt_80W_corr.pkl')