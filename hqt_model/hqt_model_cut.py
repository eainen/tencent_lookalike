#注意，我们读的所有数据文件都是有header的，如果从数据库中导入数据跟读取label时，就没有header

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
import matplotlib.pyplot as plt

def sub_to_other(x):
    x_split=x.split(" ")
    set_tmp=set(map(int,x_split))
    intersec=list(set_tmp & target_set)
    if intersec:
        for target in intersec:
            x_split[x_split.index(str(target))]='other'  
    return ' '.join(x_split)

def count_value(x):
    if x=='-1' or x==-1:
        return int(x)
    else:
        split_tmp=x.split(' ')
        return len(split_tmp)


run_day=datetime.now().strftime('%Y%m%d')
save_path="/home/heqt/tencent/"+run_day+"/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

hive_engine = create_engine('hive://heqt@192.168.254.107:10000',
                        connect_args={'auth':'KERBEROS','kerberos_service_name':'hive',
                        'configuration':{'mapred.job.queue.name':  'root.anti_fraud'}})

save_path='/home/heqt/tencent/20180506/'
data_fit=pd.read_csv('/home/heqt/tencent/train_all.csv')
#data_test=pd.read_csv('/home/heqt/tencent/test1_all.csv')
data_test=pd.read_csv('/home/heqt/tencent/test2_all.csv')
data_fit.loc[data_fit['label']==-1,'label']=0
data_test.insert(2,'label',-1)
data_x=pd.concat([data_fit,data_test])
data_x.fillna('-1',inplace=True)
data_x.to_csv('/home/heqt/tencent/data_all.csv',index=None)

#data_x=pd.read_csv('/home/heqt/tencent/data_all.csv')
data_x.fillna('-1',inplace=True)


#fit_model
one_hot_feature=['lbs','age','carrier','consumptionability','education','gender','house','advertiserid','campaignid', 'creativeid',
       'adcategoryid', 'creativesize','productid', 'producttype']
vector_feature=['appidaction','appidinstall','interest1','interest2','interest3','interest4','interest5',
'kw1','kw2','kw3','topic1','topic2','topic3','os','ct','marriagestatus']
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

#data_x=pd.read_csv('/home/heqt/tencent/20180506/data_all_cut.csv')
#data_x.to_csv('/home/heqt/tencent/20180506/data_all_cut.csv',index=None)

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
    valid_a=CVec.transform(x_valid[feature])
    test_a=CVec.transform(data_test[feature])
    data_x_train=sparse.hstack((data_x_train,train_a))
    #data_x_valid=sparse.hstack((data_x_valid,valid_a))
    data_x_test=sparse.hstack((data_x_test,test_a))
    df_tmp=pd.DataFrame(CVec.get_feature_names(),columns=['val'])
    #feature important mapping
    df_tmp['feature']='%s' %feature
    df_feature_map=pd.concat([df_feature_map,df_tmp])
print ' countvec finish'
save_path="home/heqt/tencent/20180506/"
df_feature_map.to_csv(save_path+"feature_important_mapping_cut.csv")

sparse.save_npz(save_path+"data_x_train_cut.npz",data_x_train)
x_train.to_csv(save_path+"x_train_cut.csv",index=None)
y_train.to_csv(save_path+"data_y_train_cut.csv",index=None)

sparse.save_npz(save_path+"data_x_valid_cut.npz",data_x_valid)
x_valid.to_csv(save_path+"x_valid_cut.csv",index=None)
y_valid.to_csv(save_path+"data_y_valid_cut.csv",index=None)

sparse.save_npz(save_path+"data_x_test_cut.npz",data_x_test)
data_test.to_csv(save_path+"data_test_cut.csv",index=None)
#result=data_test[['aid','uid']]
#result.to_csv(save_path+"result_cut.csv",index=None)
print 'save_cut_finish'
#"os","ct","marriagestatus"影响不大，但是也放进去跑模型了
"""
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
    test_a=pd.merge(data_test,df_cor_fea,how='left',on=[feature,'aid'])['corr_ctr']
    print train_a.shape
    print train_a.head(1)
    train_a=sparse.csr_matrix(train_a.values.reshape(-1,1))
    valid_a=sparse.csr_matrix(valid_a.values.reshape(-1,1))
    test_a=sparse.csr_matrix(test_a.values.reshape(-1,1))
    data_x_train=sparse.hstack((data_x_train,train_a))
    data_x_valid=sparse.hstack((data_x_valid,valid_a))
    data_x_test=sparse.hstack((data_x_test,test_a))
    df_tmp=pd.DataFrame(['ctr'],columns=['val'])
    #feature important mapping
    df_tmp['feature']='%s' %feature
    df_feature_map=pd.concat([df_feature_map,df_tmp])
    print feature

#上述考虑贝叶斯平滑的情况
#
corr_feature=["lbs","age","carrier","education","consumptionability","gender","os","ct","marriagestatus","house"]

for feature in corr_feature:
    corr_feature_path='/home/heqt/tencent/corr_feature/repair_aid/'+str(feature)+'_ctr_smoot.csv'
    df_cor_fea=pd.read_csv(corr_feature_path,header=None,names=[feature,'aid',feature+'dianji',feature+'baoguang','corr_ctr'])
    df_cor_fea['corr_ctr']=df_cor_fea['corr_ctr']
    print "df_cor_fea -----> shape",df_cor_fea.shape
    print df_cor_fea.head(1)
    train_a=pd.merge(x_train,df_cor_fea,how='left',on=[feature,'aid'])['corr_ctr']
    valid_a=pd.merge(x_valid,df_cor_fea,how='left',on=[feature,'aid'])['corr_ctr']
    test_a=pd.merge(data_test,df_cor_fea,how='left',on=[feature,'aid'])['corr_ctr']
    print "train_a after shape -------->",train_a.shape
    print train_a.head(1)
    train_a=sparse.csr_matrix(train_a.values.reshape(-1,1))
    valid_a=sparse.csr_matrix(valid_a.values.reshape(-1,1))
    test_a=sparse.csr_matrix(test_a.values.reshape(-1,1))
    data_x_train=sparse.hstack((data_x_train,train_a))
    data_x_valid=sparse.hstack((data_x_valid,valid_a))
    data_x_test=sparse.hstack((data_x_test,test_a))
    #df_tmp=pd.DataFrame(['ctr'],columns=['val'])
    #feature important mapping
    #df_tmp['feature']='%s' %feature
    #df_feature_map=pd.concat([df_feature_map,df_tmp])
    print feature
    print "--------------------"
"""
#上面没有考虑labelencode的情况,下面考虑了,并且用了平滑的数据
corr_feature=["lbs","age","carrier","education","consumptionability","gender","os","ct","marriagestatus","house"]

for feature in corr_feature:
    corr_feature_path='/home/heqt/tencent/corr_feature/repair_aid/'+str(feature)+'_ctr_smoot_onehot.csv'
    df_cor_fea=pd.read_csv(corr_feature_path,header=None,names=['uid','aid','corr_ctr'])
    print "df_cor_fea -----> shape",df_cor_fea.shape
    print df_cor_fea.head(1)
    train_a=pd.merge(x_train,df_cor_fea,how='left',on=['uid','aid'])['corr_ctr']
    valid_a=pd.merge(x_valid,df_cor_fea,how='left',on=['uid','aid'])['corr_ctr']
    test_a=pd.merge(data_test,df_cor_fea,how='left',on=['uid','aid'])['corr_ctr']
    print "train_a after shape -------->",train_a.shape
    print train_a.head(1)
    train_a=sparse.csr_matrix(train_a.values.reshape(-1,1))
    valid_a=sparse.csr_matrix(valid_a.values.reshape(-1,1))
    test_a=sparse.csr_matrix(test_a.values.reshape(-1,1))
    data_x_train=sparse.hstack((data_x_train,train_a))
    data_x_valid=sparse.hstack((data_x_valid,valid_a))
    data_x_test=sparse.hstack((data_x_test,test_a))
    #df_tmp=pd.DataFrame(['ctr'],columns=['val'])
    #feature important mapping
    #df_tmp['feature']='%s' %feature
    #df_feature_map=pd.concat([df_feature_map,df_tmp])
    print feature
    print "--------------------"
#统计兴趣的特征
intersec_feature=["interest1","interest2","interest3","interest4","interest5"]
for feature in intersec_feature:

    #intersec_feature_path='/home/heqt/tencent/countvec/interest/'+str(feature)+'_count.csv'
    #df_cor_fea=pd.read_csv(intersec_feature_path,header=None,names=['uid',feature,'cnt'])
    #df_cor_fea[feature]=df_cor_fea[feature].map(str)
    #df_cor_fea['cnt']=df_cor_fea['cnt'].map(np.float)
    #print df_cor_fea.shape
    #print df_cor_fea.head(1)
    train_a=x_train[feature].map(count_value)
    valid_a=x_valid[feature].map(count_value)
    test_a=data_test[feature].map(count_value)
    print train_a.shape
    print train_a.head(1)
    train_a=sparse.csr_matrix(train_a.values.reshape(-1,1))
    valid_a=sparse.csr_matrix(valid_a.values.reshape(-1,1))
    test_a=sparse.csr_matrix(test_a.values.reshape(-1,1))
    data_x_train=sparse.hstack((data_x_train,train_a))
    data_x_valid=sparse.hstack((data_x_valid,valid_a))
    data_x_test=sparse.hstack((data_x_test,test_a))
    #df_tmp=pd.DataFrame(['ctr'],columns=['val'])
    #feature important mapping
    #df_tmp['feature']='%s' %feature
    #df_feature_map=pd.concat([df_feature_map,df_tmp])
    print feature
#统计广告交叉特征
intersec_feature=["interest1","interest2","interest3","interest4","interest5"]
for feature in intersec_feature:
    #intersec_count_path='/home/heqt/tencent/countvec/interest/'+str(feature)+'_count.csv'
    intersec_feature_path='/home/heqt/tencent/corr_feature/interst/'+str(feature)+'_corr_aid.csv'
    #df_cnt_fea=pd.read_csv(intersec_count_path,header=None,names=['uid',feature,feature+'cnt'])
    print 'x_train ----> shape',x_train.shape
    x_train[feature+'cnt']=x_train[feature].map(count_value)
    x_valid[feature+'cnt']=x_valid[feature].map(count_value)
    data_test[feature+'cnt']=data_test[feature].map(count_value)
    #x_train=pd.merge(x_train,df_cnt_fea,how='left',on=['uid',feature])
    #x_valid=pd.merge(x_valid,df_cnt_fea,how='left',on=['uid',feature])
    #data_test=pd.merge(data_test,df_cnt_fea,how='left',on=['uid',feature])
    print 'x_train after merge ----> shape',x_train.shape
    print 'x_valid after merge ----> shape',x_valid.shape
    print 'data_test after merge ----> shape',data_test.shape
    df_cor_fea=pd.read_csv(intersec_feature_path,header=None,names=[feature+'cnt','aid','label','feacnt','to_talcnt','corr_ctr'])
    df_cor_fea=df_cor_fea.drop(labels=['label','feacnt','to_talcnt'],axis=1)
    print df_cor_fea.shape
    print df_cor_fea.head(1)
    train_a=pd.merge(x_train,df_cor_fea,how='left',on=[feature+'cnt','aid'])['corr_ctr']
    valid_a=pd.merge(x_valid,df_cor_fea,how='left',on=[feature+'cnt','aid'])['corr_ctr']
    test_a=pd.merge(data_test,df_cor_fea,how='left',on=[feature+'cnt','aid'])['corr_ctr']
    print train_a.shape
    print train_a.head(1)
    train_a=sparse.csr_matrix(train_a.values.reshape(-1,1))
    valid_a=sparse.csr_matrix(valid_a.values.reshape(-1,1))
    test_a=sparse.csr_matrix(test_a.values.reshape(-1,1))
    data_x_train=sparse.hstack((data_x_train,train_a))
    data_x_valid=sparse.hstack((data_x_valid,valid_a))
    data_x_test=sparse.hstack((data_x_test,test_a))
    #df_tmp=pd.DataFrame(['ctr'],columns=['val'])
    #feature important mapping
    #df_tmp['feature']='%s' %feature
    #df_feature_map=pd.concat([df_feature_map,df_tmp])
    print feature
    print '-----------------------------'
 #   "os","ct","marriagestatus","advertiserId","campaignId","creativeSize","productType"效果不理想，所以去掉
var_features=["gender","age","lbs","carrier","education","house"]
for var_feature in var_features:
    corr_feature=["advertiserId","campaignId","creativeSize","adCategoryId","productId","productType"]
    for feature in corr_feature:
        corr_feature_path="/home/heqt/tencent/corr_feature/"+var_feature+"/"+var_feature+"_corr_"+str(feature)+".csv"
        df_cor_fea=pd.read_csv(corr_feature_path,header=None,names=[var_feature,feature.lower(),'label','feacnt','cnt','corr_ctr'])
        df_cor_fea['corr_ctr']=df_cor_fea['corr_ctr']*100.0
        print "df_cor_fea ------>shape",df_cor_fea.shape
        print df_cor_fea.head(1)
        df_cor_fea=df_cor_fea.drop(labels=['label','feacnt','cnt'],axis=1)
        train_a=pd.merge(x_train,df_cor_fea,how='left',on=[var_feature,feature.lower()])['corr_ctr']
        valid_a=pd.merge(x_valid,df_cor_fea,how='left',on=[var_feature,feature.lower()])['corr_ctr']
        test_a=pd.merge(data_test,df_cor_fea,how='left',on=[var_feature,feature.lower()])['corr_ctr']
        print "train_a ---> shape",train_a.shape
        print train_a.head(1)
        train_a=sparse.csr_matrix(train_a.values.reshape(-1,1))
        valid_a=sparse.csr_matrix(valid_a.values.reshape(-1,1))
        test_a=sparse.csr_matrix(test_a.values.reshape(-1,1))
        data_x_train=sparse.hstack((data_x_train,train_a))
        data_x_valid=sparse.hstack((data_x_valid,valid_a))
        data_x_test=sparse.hstack((data_x_test,test_a))
        #df_tmp=pd.DataFrame(['ctr'],columns=['val'])
        #feature important mapping
        #df_tmp['feature']='%s' %feature
        #df_feature_map=pd.concat([df_feature_map,df_tmp])
        print feature
        print "-----------------------"
#3交叉 模型效果：9个交叉+gender+age共20个[324, 979, 678, 795, 1112, 0, 0,0,232,304, 874,819, 840,1064,207,341,906,1139，962, 313]
source_feature=["gender"]
corr_feature=["lbs","age","carrier","education","consumptionability","house"]
for source in source_feature:
    for feature in corr_feature:
        corr_feature_path='/home/heqt/tencent/3corr_feature/'+str(source)+"_"+str(feature)+'_corr_aid.csv'
        if feature in ["interest1","interest2","interest3","interest4","interest5"]:
            df_cor_fea=pd.read_csv(corr_feature_path,header=None,names=[source,feature+"cnt",'aid','label','feacnt','cnt','corr_ctr'])
        else:
            df_cor_fea=pd.read_csv(corr_feature_path,header=None,names=[source,feature,'aid','label','feacnt','cnt','corr_ctr'])
        df_cor_fea['corr_ctr']=df_cor_fea['corr_ctr']*100.0
        print "df_cor_fea -----> shape",df_cor_fea.shape
        print df_cor_fea.head(1)
        df_cor_fea=df_cor_fea.drop(labels=['label','feacnt','cnt'],axis=1)
        if feature in ["interest1","interest2","interest3","interest4","interest5"]:
            train_a=pd.merge(x_train,df_cor_fea,how='left',on=[source,feature+"cnt",'aid'])['corr_ctr']
            valid_a=pd.merge(x_valid,df_cor_fea,how='left',on=[source,feature+"cnt",'aid'])['corr_ctr']

            print "excute interest"
        else:
            train_a=pd.merge(x_train,df_cor_fea,how='left',on=[source,feature,'aid'])['corr_ctr']
            valid_a=pd.merge(x_valid,df_cor_fea,how='left',on=[source,feature,'aid'])['corr_ctr']
            test_a=pd.merge(data_test,df_cor_fea,how='left',on=[source,feature,'aid'])['corr_ctr']
        print "train_a after shape -------->",train_a.shape
        print train_a.head(1)
        train_a=sparse.csr_matrix(train_a.values.reshape(-1,1))
        valid_a=sparse.csr_matrix(valid_a.values.reshape(-1,1))
        test_a=sparse.csr_matrix(test_a.values.reshape(-1,1))
        data_x_train=sparse.hstack((data_x_train,train_a))
        data_x_valid=sparse.hstack((data_x_valid,valid_a))
        data_x_test=sparse.hstack((data_x_test,test_a))
        #df_tmp=pd.DataFrame(['ctr'],columns=['val'])
        #feature important mapping
        #df_tmp['feature']='%s' %feature
        #df_feature_map=pd.concat([df_feature_map,df_tmp])
        print feature
        print "--------------------"
source_feature=["age"]
corr_feature=["lbs","carrier","education","consumptionability","house"]
for source in source_feature:
    for feature in corr_feature:
        corr_feature_path='/home/heqt/tencent/3corr_feature/'+str(source)+"_"+str(feature)+'_corr_aid.csv'
        if feature in ["interest1","interest2","interest3","interest4","interest5"]:
            df_cor_fea=pd.read_csv(corr_feature_path,header=None,names=[source,feature+"cnt",'aid','label','feacnt','cnt','corr_ctr'])
        else:
            df_cor_fea=pd.read_csv(corr_feature_path,header=None,names=[source,feature,'aid','label','feacnt','cnt','corr_ctr'])
        df_cor_fea['corr_ctr']=df_cor_fea['corr_ctr']*100.0
        print "df_cor_fea -----> shape",df_cor_fea.shape
        print df_cor_fea.head(1)
        df_cor_fea=df_cor_fea.drop(labels=['label','feacnt','cnt'],axis=1)
        if feature in ["interest1","interest2","interest3","interest4","interest5"]:
            train_a=pd.merge(x_train,df_cor_fea,how='left',on=[source,feature+"cnt",'aid'])['corr_ctr']
            valid_a=pd.merge(x_valid,df_cor_fea,how='left',on=[source,feature+"cnt",'aid'])['corr_ctr']
            print "excute interest"
        else:
            train_a=pd.merge(x_train,df_cor_fea,how='left',on=[source,feature,'aid'])['corr_ctr']
            valid_a=pd.merge(x_valid,df_cor_fea,how='left',on=[source,feature,'aid'])['corr_ctr']
            test_a=pd.merge(data_test,df_cor_fea,how='left',on=[source,feature,'aid'])['corr_ctr']
        print "train_a after shape -------->",train_a.shape
        print train_a.head(1)
        train_a=sparse.csr_matrix(train_a.values.reshape(-1,1))
        valid_a=sparse.csr_matrix(valid_a.values.reshape(-1,1))
        test_a=sparse.csr_matrix(test_a.values.reshape(-1,1))
        data_x_train=sparse.hstack((data_x_train,train_a))
        data_x_valid=sparse.hstack((data_x_valid,valid_a))
        data_x_test=sparse.hstack((data_x_test,test_a))
        #df_tmp=pd.DataFrame(['ctr'],columns=['val'])
        #feature important mapping
        #df_tmp['feature']='%s' %feature
        #df_feature_map=pd.concat([df_feature_map,df_tmp])
        print feature
        print "--------------------"
#使用贝叶斯平滑：因为"appidaction","appidinstall","interest1","interest2","interest3","interest4","interest5"影响不大，没有添加进去,
#这里是没有考虑如果没有就补0跟填充-1的情况
sigle_interest=["kw1","kw2","kw3","topic1","topic2","topic3"]
for feature in sigle_interest:
    intersec_feature_path='/home/heqt/tencent/corr_feature/single_corr/for_smoot/smooted_ctr/'+feature+'_uid_aid_ctr.csv'
    df_cor_fea=pd.read_csv(intersec_feature_path,header=None,names=['uid','aid',feature+'ctr'])
    #because smoe label=-1 will cause null,need df_cor_count to fill null
    print "df_cor_fea---->shape",df_cor_fea.shape
    #print df_cor_fea.head(1)
    train_a=pd.merge(x_train,df_cor_fea,how='left',on=['uid','aid'])[feature+'ctr']
    valid_a=pd.merge(x_valid,df_cor_fea,how='left',on=['uid','aid'])[feature+'ctr']
    test_a=pd.merge(data_test,df_cor_fea,how='left',on=['uid','aid'])[feature+'ctr']
    print 'train_a ---> shape',train_a.shape
    #print 'train_a --->head(1)',train_a.head(1)
    train_a=sparse.csr_matrix(train_a.values.reshape(-1,1))
    valid_a=sparse.csr_matrix(valid_a.values.reshape(-1,1))
    test_a=sparse.csr_matrix(test_a.values.reshape(-1,1))
    data_x_train=sparse.hstack((data_x_train,train_a))
    data_x_valid=sparse.hstack((data_x_valid,valid_a))
    data_x_test=sparse.hstack((data_x_test,test_a))
    print feature
#使用贝叶斯平滑：因为"appidaction","appidinstall","interest1","interest2","interest3","interest4","interest5"影响不大，没有添加进去,
#这里是考虑了补0跟填充-1的情况
sigle_interest=["kw1","kw2","kw3","topic1","topic2","topic3"]
for feature in sigle_interest:
    intersec_feature_path='/home/heqt/tencent/corr_feature/single_corr/for_smoot/smooted_ctr/'+feature+'_uid_aid_ctr_rv_0.csv'
    df_cor_fea=pd.read_csv(intersec_feature_path,header=None,names=['uid','aid',feature+'ctr'])
    #because smoe label=-1 will cause null,need df_cor_count to fill null
    print "df_cor_fea---->shape",df_cor_fea.shape
    #print df_cor_fea.head(1)
    train_a=pd.merge(x_train,df_cor_fea,how='left',on=['uid','aid'])[feature+'ctr']
    valid_a=pd.merge(x_valid,df_cor_fea,how='left',on=['uid','aid'])[feature+'ctr']
    test_a=pd.merge(data_test,df_cor_fea,how='left',on=['uid','aid'])[feature+'ctr']
    train_a[np.where(x_train[feature]=='-1')[0]]=-1
    valid_a[np.where(x_valid[feature]=='-1')[0]]=-1
    test_a[np.where(data_test[feature]=='-1')[0]]=-1
    print 'train_a ---> shape',train_a.shape
    #print 'train_a --->head(1)',train_a.head(1)
    train_a=sparse.csr_matrix(train_a.values.reshape(-1,1))
    valid_a=sparse.csr_matrix(valid_a.values.reshape(-1,1))
    test_a=sparse.csr_matrix(test_a.values.reshape(-1,1))
    data_x_train=sparse.hstack((data_x_train,train_a))
    data_x_valid=sparse.hstack((data_x_valid,valid_a))
    data_x_test=sparse.hstack((data_x_test,test_a))
    print feature

#组合统计量
#corr_feature=["age","gender","house"]
corr_feature=["age","gender","campaignid"]
x_train['uid']=x_train['uid'].map(np.int64)
x_train['aid']=x_train['aid'].map(np.int64)
x_valid['uid']=x_valid['uid'].map(np.int64)
data_test['aid']=data_test['aid'].map(np.int64)
data_test['uid']=data_test['uid'].map(np.int64)

for feature in corr_feature:
    corr_feature_path='/home/heqt/tencent/corr_feature/'+str(feature)+'_aid_count.csv'
    df_cor_fea=pd.read_csv(corr_feature_path,header=None,names=['uid','aid','cnt'],dtype={0: np.int64,1: np.int64})
    print "df_cor_fea -----> shape",df_cor_fea.shape
    print df_cor_fea.head(1)
    train_a=pd.merge(x_train,df_cor_fea,how='left',on=['uid','aid'])['cnt']
    valid_a=pd.merge(x_valid,df_cor_fea,how='left',on=['uid','aid'])['cnt']
    test_a=pd.merge(data_test,df_cor_fea,how='left',on=['uid','aid'])['cnt']
    print "train_a after shape -------->",train_a.shape
    print train_a.head(1)
    train_a=sparse.csr_matrix(train_a.values.reshape(-1,1))
    valid_a=sparse.csr_matrix(valid_a.values.reshape(-1,1))
    test_a=sparse.csr_matrix(test_a.values.reshape(-1,1))
    data_x_train=sparse.hstack((data_x_train,train_a))
    data_x_valid=sparse.hstack((data_x_valid,valid_a))
    data_x_test=sparse.hstack((data_x_test,test_a))
    #df_tmp=pd.DataFrame(['ctr'],columns=['val'])
    #feature important mapping
    #df_tmp['feature']='%s' %feature
    #df_feature_map=pd.concat([df_feature_map,df_tmp])
    print feature
    print "--------------------"

#与广告的2交叉组合,选取了3个交叉变量

corr_feature=["age","gender","house"]

x_train['uid']=x_train['uid'].map(np.int64)
x_train['aid']=x_train['aid'].map(np.int64)
x_valid['uid']=x_valid['uid'].map(np.int64)
x_valid['aid']=x_valid['aid'].map(np.int64)
data_test['aid']=data_test['aid'].map(np.int64)
data_test['uid']=data_test['uid'].map(np.int64)

for feature in corr_feature:
    corr_feature_path='/home/heqt/tencent/corr_feature/'+str(feature)+'_aid_zuhe.csv'
    df_cor_fea=pd.read_csv(corr_feature_path,header=None,names=['uid','aid',feature+'_aid'],dtype={0: np.int64,1: np.int64})
    print "df_cor_fea -----> shape",df_cor_fea.shape
    print df_cor_fea.head(1)
    x_train[feature+'_aid']=pd.merge(x_train,df_cor_fea,how='left',on=['uid','aid'])[feature+'_aid']
    x_valid[feature+'_aid']=pd.merge(x_valid,df_cor_fea,how='left',on=['uid','aid'])[feature+'_aid']
    data_test[feature+'_aid']=pd.merge(data_test,df_cor_fea,how='left',on=['uid','aid'])[feature+'_aid']
x_train['qubie']=1
x_valid['qubie']=0
data_test['qubie']=2
data_x=pd.concat([x_train,x_valid])
data_x=pd.concat([data_x,data_test])
LE=LabelEncoder()
for feature in corr_feature:
    try:
        data_x[feature+'_aid']=LE.fit_transform(data_x[feature+'_aid'].map(np.int))
    except:
        data_x[feature+'_aid']=LE.fit_transform(data_x[feature+'_aid'])
print "LabelEncoder finish"
x_train=data_x[data_x.qubie==1]
x_valid=data_x[data_x.qubie==0]
data_test=data_x[data_x.qubie==2]
data_x.pop('qubie'),x_train.pop('qubie'),x_valid.pop('qubie'),data_test.pop('qubie')
OHE=OneHotEncoder()
for feature in corr_feature:
    OHE.fit(data_x[feature+'_aid'].values.reshape(-1,1))
    train_a=OHE.transform(x_train[feature+'_aid'].values.reshape(-1,1))
    valid_a=OHE.transform(x_valid[feature+'_aid'].values.reshape(-1,1))
    test_a=OHE.transform(data_test[feature+'_aid'].values.reshape(-1,1))
    print 'train_a ----> one hot--->shape',train_a.shape
    data_x_train=sparse.hstack((data_x_train,train_a))
    data_x_valid=sparse.hstack((data_x_valid,valid_a))
    data_x_test=sparse.hstack((data_x_test,test_a))
    print "data_x_train ---->shape",data_x_train.shape
    print feature
    print "----------------"

#
#用户特征组合广告的其他特征

corr_feature=["marriagestatus","age","consumptionability"]
aid_feature=["creativesize"]
aid_feature_age=["creativesize","adcategoryid"]
#为了labelencoder用的list
Encoder_list=[]
x_train['uid']=x_train['uid'].map(np.int64)
x_train['aid']=x_train['aid'].map(np.int64)
x_valid['uid']=x_valid['uid'].map(np.int64)
x_valid['aid']=x_valid['aid'].map(np.int64)
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
        x_valid[feature+'_'+adfeature]=pd.merge(x_valid,df_cor_fea,how='left',on=['uid','aid'])[feature+'_'+adfeature]
        data_test[feature+'_'+adfeature]=pd.merge(data_test,df_cor_fea,how='left',on=['uid','aid'])[feature+'_'+adfeature]
        Encoder_list.append(feature+'_'+adfeature)
x_train['qubie']=1
x_valid['qubie']=0
data_test['qubie']=2
data_x=pd.concat([x_train,x_valid])
data_x=pd.concat([data_x,data_test])
LE=LabelEncoder()
for feature in Encoder_list:
    try:
        data_x[feature]=LE.fit_transform(data_x[feature].map(np.int))
    except:
        data_x[feature]=LE.fit_transform(data_x[feature])
print "LabelEncoder finish"
x_train=data_x[data_x.qubie==1]
x_valid=data_x[data_x.qubie==0]
data_test=data_x[data_x.qubie==2]
data_x.pop('qubie'),x_train.pop('qubie'),x_valid.pop('qubie'),data_test.pop('qubie')
OHE=OneHotEncoder()
#data_x_train_1=data_x_train.copy()
#data_x_valid_1=data_x_valid.copy()
#df_AUC=pd.DataFrame(columns=['var_name','AUC_0','AUC_1'])
for feature in Encoder_list:
    OHE.fit(data_x[feature].values.reshape(-1,1))
    train_a=OHE.transform(x_train[feature].values.reshape(-1,1))
    valid_a=OHE.transform(x_valid[feature].values.reshape(-1,1))
    test_a=OHE.transform(data_test[feature].values.reshape(-1,1))
    print 'train_a ----> one hot--->shape',train_a.shape
    data_x_train=sparse.hstack((data_x_train,train_a))
    data_x_valid=sparse.hstack((data_x_valid,valid_a))
    data_x_test=sparse.hstack((data_x_test,test_a))
    print "data_x_train ---->shape",data_x_train.shape
    print feature
    print "----------------"

#用户特征相交,只选了os_gender
corr_feature=['os_gender']
#为了labelencoder用的list
x_train['uid']=x_train['uid'].map(np.int64)
x_train['aid']=x_train['aid'].map(np.int64)
x_valid['uid']=x_valid['uid'].map(np.int64)
x_valid['aid']=x_valid['aid'].map(np.int64)
for feature in corr_feature:
    corr_feature_path='/home/heqt/tencent/corr_feature/'+str(feature)+'_zuhe.csv'
    df_cor_fea=pd.read_csv(corr_feature_path,header=None,names=['uid','aid',feature],dtype={0: np.int64,1: np.int64})
    print "df_cor_fea -----> shape",df_cor_fea.shape
    print df_cor_fea.head(1)
    x_train[feature]=pd.merge(x_train,df_cor_fea,how='left',on=['uid','aid'])[feature]
    x_valid[feature]=pd.merge(x_valid,df_cor_fea,how='left',on=['uid','aid'])[feature]
    data_test[feature]=pd.merge(data_test,df_cor_fea,how='left',on=['uid','aid'])[feature]
x_train['qubie']=1
x_valid['qubie']=0
data_test['qubie']=2
data_x=pd.concat([x_train,x_valid])
data_x=pd.concat([data_x,data_test])
LE=LabelEncoder()
for feature in corr_feature:
    try:
        data_x[feature]=LE.fit_transform(data_x[feature].map(np.int))
    except:
        data_x[feature]=LE.fit_transform(data_x[feature])
print "LabelEncoder finish"
x_train=data_x[data_x.qubie==1]
x_valid=data_x[data_x.qubie==0]
data_test=data_x[data_x.qubie==2]
data_x.pop('qubie'),x_train.pop('qubie'),x_valid.pop('qubie'),data_test.pop('qubie')
OHE=OneHotEncoder()
#data_x_train_1=data_x_train.copy()
#data_x_valid_1=data_x_valid.copy()
#df_AUC=pd.DataFrame(columns=['var_name','AUC_0','AUC_1'])
for feature in corr_feature:
    OHE.fit(data_x[feature].values.reshape(-1,1))
    train_a=OHE.transform(x_train[feature].values.reshape(-1,1))
    valid_a=OHE.transform(x_valid[feature].values.reshape(-1,1))
    test_a=OHE.transform(data_test[feature].values.reshape(-1,1))
    print 'train_a ----> one hot--->shape',train_a.shape
    data_x_train=sparse.hstack((data_x_train,train_a))
    data_x_valid=sparse.hstack((data_x_valid,valid_a))
    data_x_test=sparse.hstack((data_x_test,test_a))

    print "data_x_train ---->shape",data_x_train.shape
    print feature
    print "----------------"
#3交叉组合



corr_feature=["age_gender_aid","age_consumptionability_aid","age_education_aid","gender_house_aid","gender_consumptionability_aid",
             "gender_marriagestatus_aid","gender_education_aid"]
#为了labelencoder用的list
x_train['uid']=x_train['uid'].map(np.int64)
x_train['aid']=x_train['aid'].map(np.int64)
x_valid['uid']=x_valid['uid'].map(np.int64)
x_valid['aid']=x_valid['aid'].map(np.int64)
data_test['aid']=data_test['aid'].map(np.int64)
data_test['uid']=data_test['uid'].map(np.int64)
for feature in corr_feature:
    corr_feature_path='/home/heqt/tencent/3corr_feature/'+str(feature)+'_zuhe.csv'
    df_cor_fea=pd.read_csv(corr_feature_path,header=None,names=['uid','aid',feature],dtype={0: np.int64,1: np.int64})
    print "df_cor_fea -----> shape",df_cor_fea.shape
    print df_cor_fea.head(1)
    x_train[feature]=pd.merge(x_train,df_cor_fea,how='left',on=['uid','aid'])[feature]
    x_valid[feature]=pd.merge(x_valid,df_cor_fea,how='left',on=['uid','aid'])[feature]
    data_test[feature]=pd.merge(data_test,df_cor_fea,how='left',on=['uid','aid'])[feature]
x_train['qubie']=1
x_valid['qubie']=0
data_test['qubie']=2
data_x=pd.concat([x_train,x_valid])
data_x=pd.concat([data_x,data_test])
LE=LabelEncoder()
for feature in corr_feature:
    try:
        data_x[feature]=LE.fit_transform(data_x[feature].map(np.int))
    except:
        data_x[feature]=LE.fit_transform(data_x[feature])
print "LabelEncoder finish"
x_train=data_x[data_x.qubie==1]
x_valid=data_x[data_x.qubie==0]
data_test=data_x[data_x.qubie==2]
data_x.pop('qubie'),x_train.pop('qubie'),x_valid.pop('qubie'),data_test.pop('qubie')
OHE=OneHotEncoder()
#data_x_train_1=data_x_train.copy()
#data_x_valid_1=data_x_valid.copy()
#df_AUC=pd.DataFrame(columns=['var_name','AUC_0','AUC_1'])
for feature in corr_feature:
    OHE.fit(data_x[feature].values.reshape(-1,1))
    train_a=OHE.transform(x_train[feature].values.reshape(-1,1))
    valid_a=OHE.transform(x_valid[feature].values.reshape(-1,1))
    test_a=OHE.transform(data_test[feature].values.reshape(-1,1))
    print 'train_a ----> one hot--->shape',train_a.shape
    data_x_train=sparse.hstack((data_x_train,train_a))
    data_x_valid=sparse.hstack((data_x_valid,valid_a))
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
x_train['uid']=x_train['uid'].map(np.int64)
x_train['aid']=x_train['aid'].map(np.int64)
x_valid['uid']=x_valid['uid'].map(np.int64)
x_valid['aid']=x_valid['aid'].map(np.int64)
for feature in corr_feature:
    corr_feature_path='/home/heqt/tencent/3corr_feature/'+str(feature)+'_zuhe.csv'
    df_cor_fea=pd.read_csv(corr_feature_path,header=None,names=['uid','aid',feature],dtype={0: np.int64,1: np.int64})
    print "df_cor_fea -----> shape",df_cor_fea.shape
    print df_cor_fea.head(1)
    x_train[feature]=pd.merge(x_train,df_cor_fea,how='left',on=['uid','aid'])[feature]
    x_valid[feature]=pd.merge(x_valid,df_cor_fea,how='left',on=['uid','aid'])[feature]
    data_test[feature]=pd.merge(data_test,df_cor_fea,how='left',on=['uid','aid'])[feature]

x_train['qubie']=1
x_valid['qubie']=0
data_test['qubie']=2
data_x=pd.concat([x_train,x_valid])
data_x=pd.concat([data_x,data_test])
LE=LabelEncoder()
for feature in corr_feature:
    try:
        data_x[feature]=LE.fit_transform(data_x[feature].map(np.int))
    except:
        data_x[feature]=LE.fit_transform(data_x[feature])
print "LabelEncoder finish"
x_train=data_x[data_x.qubie==1]
x_valid=data_x[data_x.qubie==0]
data_test=data_x[data_x.qubie==2]
data_x.pop('qubie'),x_train.pop('qubie'),x_valid.pop('qubie'),data_test.pop('qubie')
OHE=OneHotEncoder()
#data_x_train_1=data_x_train.copy()
#data_x_valid_1=data_x_valid.copy()
#df_AUC_age=pd.DataFrame(columns=['var_name','AUC_0','AUC_1'])
for feature in corr_feature:
    OHE.fit(data_x[feature].values.reshape(-1,1))
    train_a=OHE.transform(x_train[feature].values.reshape(-1,1))
    valid_a=OHE.transform(x_valid[feature].values.reshape(-1,1))
    test_a=OHE.transform(data_test[feature].values.reshape(-1,1))
    print 'train_a ----> one hot--->shape',train_a.shape
    data_x_train=sparse.hstack((data_x_train,train_a))
    data_x_valid=sparse.hstack((data_x_valid,valid_a))
    data_x_test=sparse.hstack((data_x_test,test_a))
    print "data_x_train ---->shape",data_x_train.shape
    print feature
    print "----------------"

#尝试gender与广告的其他组合特征:

corr_feature=["gender_house_creativesize","gender_consumptionability_creativesize","gender_marriagestatus_creativesize",
              "gender_education_creativesize"]

x_train['uid']=x_train['uid'].map(np.int64)
x_train['aid']=x_train['aid'].map(np.int64)
x_valid['uid']=x_valid['uid'].map(np.int64)
x_valid['aid']=x_valid['aid'].map(np.int64)
for feature in corr_feature:
    corr_feature_path='/home/heqt/tencent/3corr_feature/'+str(feature)+'_zuhe.csv'
    df_cor_fea=pd.read_csv(corr_feature_path,header=None,names=['uid','aid',feature],dtype={0: np.int64,1: np.int64})
    print "df_cor_fea -----> shape",df_cor_fea.shape
    print df_cor_fea.head(1)
    x_train[feature]=pd.merge(x_train,df_cor_fea,how='left',on=['uid','aid'])[feature]
    x_valid[feature]=pd.merge(x_valid,df_cor_fea,how='left',on=['uid','aid'])[feature]
    data_test[feature]=pd.merge(data_test,df_cor_fea,how='left',on=['uid','aid'])[feature]
x_train['qubie']=1
x_valid['qubie']=0
data_test['qubie']=2
data_x=pd.concat([x_train,x_valid])
data_x=pd.concat([data_x,data_test])
LE=LabelEncoder()
for feature in corr_feature:
    try:
        data_x[feature]=LE.fit_transform(data_x[feature].map(np.int))
    except:
        data_x[feature]=LE.fit_transform(data_x[feature])
print "LabelEncoder finish"
x_train=data_x[data_x.qubie==1]
x_valid=data_x[data_x.qubie==0]
data_test=data_x[data_x.qubie==2]
data_x.pop('qubie'),x_train.pop('qubie'),x_valid.pop('qubie'),data_test.pop('qubie')
OHE=OneHotEncoder()
#data_x_train_1=data_x_train.copy()
#data_x_valid_1=data_x_valid.copy()
df_AUC=pd.DataFrame(columns=['var_name','AUC_0','AUC_1'])
for feature in corr_feature:
    OHE.fit(data_x[feature].values.reshape(-1,1))
    train_a=OHE.transform(x_train[feature].values.reshape(-1,1))
    valid_a=OHE.transform(x_valid[feature].values.reshape(-1,1))
    test_a=OHE.transform(data_test[feature].values.reshape(-1,1))
    print 'train_a ----> one hot--->shape',train_a.shape
    data_x_train=sparse.hstack((data_x_train,train_a))
    data_x_valid=sparse.hstack((data_x_valid,valid_a))
    data_x_test=sparse.hstack((data_x_test,test_a))
    print "data_x_train ---->shape",data_x_train.shape
    print feature
    print "----------------"

#尝试逻辑回归把数值型变量拟合成一个值给GBDT训练
df_LR_train=pd.DataFrame()
df_LR_valid=pd.DataFrame()
df_LR_test=pd.DataFrame()
#sigle_interest=["appidaction","kw1","kw2","kw3","topic1","topic2","topic3"]
sigle_interest=["appidaction","appidinstall","interest1","interest2","interest3","interest4","interest5","topic1","topic2","topic3","kw1","kw2","kw3"]
for feature in sigle_interest:
    intersec_feature_path='/home/heqt/tencent/corr_feature/single_corr/for_smoot/smooted_ctr/'+feature+'_uid_aid_ctr_rv_0.csv'
    df_cor_fea=pd.read_csv(intersec_feature_path,header=None,names=['uid','aid',feature+'ctr'],dtype={0: np.int64,1: np.int64})
    #because smoe label=-1 will cause null,need df_cor_count to fill null
    print "df_cor_fea---->shape",df_cor_fea.shape
    #print df_cor_fea.head(1)
    train_a=pd.merge(x_train,df_cor_fea,how='left',on=['uid','aid'])[feature+'ctr']
    valid_a=pd.merge(x_valid,df_cor_fea,how='left',on=['uid','aid'])[feature+'ctr']
    test_a=pd.merge(data_test,df_cor_fea,how='left',on=['uid','aid'])[feature+'ctr']
    df_LR_train=pd.concat([df_LR_train,train_a],axis=1)
    df_LR_valid=pd.concat([df_LR_valid,valid_a],axis=1)
    df_LR_test=pd.concat([df_LR_test,test_a],axis=1)
    #train_a[np.where(x_train[feature]=='-1')[0]]=-1
    #valid_a[np.where(x_valid[feature]=='-1')[0]]=-1
    #test_a[np.where(data_test[feature]=='-1')[0]]=-1
    print 'train_a ---> shape',train_a.shape
    #print 'train_a --->head(1)',train_a.head(1)
    #train_a=sparse.csr_matrix(train_a.values.reshape(-1,1))
    #valid_a=sparse.csr_matrix(valid_a.values.reshape(-1,1))
    #test_a=sparse.csr_matrix(test_a.values.reshape(-1,1))
    #data_x_train=sparse.hstack((data_x_train,train_a))
    #data_x_valid=sparse.hstack((data_x_valid,valid_a))
    #data_x_test=sparse.hstack((data_x_test,test_a))
    print feature
#调试LR
LR_clf=LogisticRegression(penalty='l2', tol=0.0001, C=500,random_state=20180521, solver='lbfgs', max_iter=200,n_jobs=10)
LR_clf.fit(df_LR_train,y_train)
LR_pre_train=LR_clf.predict_proba(df_LR_train)
LR_pre_valid=LR_clf.predict_proba(df_LR_valid)
LR_pre_test=LR_clf.predict_proba(df_LR_test)

train_a=LR_pre_train[:,1]
valid_a=LR_pre_valid[:,1]
test_a=LR_pre_test[:,1]
train_a=sparse.csr_matrix(train_a.reshape(-1,1))
valid_a=sparse.csr_matrix(valid_a.reshape(-1,1))
test_a=sparse.csr_matrix(test_a.reshape(-1,1))
data_x_train=sparse.hstack((data_x_train,train_a))
data_x_valid=sparse.hstack((data_x_valid,valid_a))
data_x_test=sparse.hstack((data_x_test,test_a))


#aid,lbs,campaignid,producttype的历史点击率,不同广告在不同人群的投放量,经过one_hot修改后的

#corr_feature=["aid_toufang","aid","lbs","campaignid","producttype"]
corr_feature=["aid_toufang","aid"]
x_train['uid']=x_train['uid'].map(np.int64)
x_train['aid']=x_train['aid'].map(np.int64)
x_valid['uid']=x_valid['uid'].map(np.int64)
x_valid['aid']=x_valid['aid'].map(np.int64)
data_test['aid']=data_test['aid'].map(np.int64)
data_test['uid']=data_test['uid'].map(np.int64)
for feature in corr_feature:
    corr_feature_path='/home/heqt/tencent/corr_feature/'+str(feature)+'_uid_count_onehot.csv'
    if feature=="aid_toufang":
        feature="aid"
        df_cor_fea=pd.read_csv(corr_feature_path,header=None,names=[feature,'corr_ctr'],dtype={0: np.int64})
        train_a=pd.merge(x_train,df_cor_fea,how='left',on=[feature])['corr_ctr']
        valid_a=pd.merge(x_valid,df_cor_fea,how='left',on=[feature])['corr_ctr']
        test_a=pd.merge(data_test,df_cor_fea,how='left',on=[feature])['corr_ctr']

        train_a=sparse.csr_matrix(train_a.values.reshape(-1,1))
        valid_a=sparse.csr_matrix(valid_a.values.reshape(-1,1))
        test_a=sparse.csr_matrix(test_a.values.reshape(-1,1))
        data_x_train=sparse.hstack((data_x_train,train_a))
        data_x_valid=sparse.hstack((data_x_valid,valid_a))
        data_x_test=sparse.hstack((data_x_test,test_a))
    else:
        df_cor_fea=pd.read_csv(corr_feature_path,header=None,names=['uid','aid','corr_ctr'],dtype={0: np.int64,1: np.int64})
        print "df_cor_fea -----> shape",df_cor_fea.shape
        print df_cor_fea.head(1)
        train_a=pd.merge(x_train,df_cor_fea,how='left',on=['uid','aid'])['corr_ctr']
        valid_a=pd.merge(x_valid,df_cor_fea,how='left',on=['uid','aid'])['corr_ctr']
        test_a=pd.merge(data_test,df_cor_fea,how='left',on=['uid','aid'])['corr_ctr']
        print "train_a after shape -------->",train_a.shape
        print train_a.head(1)
        train_a=sparse.csr_matrix(train_a.values.reshape(-1,1))
        valid_a=sparse.csr_matrix(valid_a.values.reshape(-1,1))
        test_a=sparse.csr_matrix(test_a.values.reshape(-1,1))
        data_x_train=sparse.hstack((data_x_train,train_a))
        data_x_valid=sparse.hstack((data_x_valid,valid_a))
        data_x_test=sparse.hstack((data_x_test,test_a))
    #df_tmp=pd.DataFrame(['ctr'],columns=['val'])
    #feature important mapping
    #df_tmp['feature']='%s' %feature
    #df_feature_map=pd.concat([df_feature_map,df_tmp])
    print feature
    print "--------------------"

#最后一波组合特征
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
x_valid['uid']=x_valid['uid'].map(np.int64)
x_valid['aid']=x_valid['aid'].map(np.int64)
data_test['aid']=data_test['aid'].map(np.int64)
data_test['uid']=data_test['uid'].map(np.int64)
for feature in corr_feature:
    for adfeature in aid_feature:
        corr_feature_path='/home/heqt/tencent/tongjitezheng/'+str(feature)+'_'+str(adfeature)+'_tongji.csv'
        df_cor_fea=pd.read_csv(corr_feature_path,header=None,names=['uid','aid',feature+'_'+adfeature],dtype={0: np.int64,1: np.int64})
        print "df_cor_fea -----> shape",df_cor_fea.shape
        print df_cor_fea.head(1)
        train_a=pd.merge(x_train,df_cor_fea,how='left',on=['uid','aid'])[feature+'_'+adfeature]
        valid_a=pd.merge(x_valid,df_cor_fea,how='left',on=['uid','aid'])[feature+'_'+adfeature]
        test_a=pd.merge(data_test,df_cor_fea,how='left',on=['uid','aid'])[feature+'_'+adfeature]
        train_a=sparse.csr_matrix(train_a.values.reshape(-1,1))
        valid_a=sparse.csr_matrix(valid_a.values.reshape(-1,1))
        test_a=sparse.csr_matrix(test_a.values.reshape(-1,1))
        data_x_train=sparse.hstack((data_x_train,train_a))
        data_x_valid=sparse.hstack((data_x_valid,valid_a))
        data_x_test=sparse.hstack((data_x_test,test_a))


#之前没入模的统计变量
#尝试之前没入模的的组合特征,与aid组合
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
    valid_a=pd.merge(x_valid,df_cor_fea,how='left',on=['uid','aid'])['corr_ctr']
    test_a=pd.merge(data_test,df_cor_fea,how='left',on=['uid','aid'])['corr_ctr']
    print "train_a after shape -------->",train_a.shape
    print train_a.head(1)
    train_a=sparse.csr_matrix(train_a.values.reshape(-1,1))
    valid_a=sparse.csr_matrix(valid_a.values.reshape(-1,1))
    test_a=sparse.csr_matrix(test_a.values.reshape(-1,1))
    data_x_train=sparse.hstack((data_x_train,train_a))
    data_x_valid=sparse.hstack((data_x_valid,valid_a))
    data_x_test=sparse.hstack((data_x_test,test_a))
    #df_tmp=pd.DataFrame(['ctr'],columns=['val'])
    #feature important mapping
    #df_tmp['feature']='%s' %feature
    #df_feature_map=pd.concat([df_feature_map,df_tmp])
    print feature
    print "--------------------"

sparse.save_npz(save_path+"data_x_train_cut_corr_23.npz",data_x_train)
x_train.to_csv(save_path+"x_train_cut_corr_23.csv",index=None)
y_train.to_csv(save_path+"data_y_train_cut_corr_23.csv",index=None,header=None)

sparse.save_npz(save_path+"data_x_valid_cut_corr_23.npz",data_x_valid)
x_valid.to_csv(save_path+"x_valid_cut_corr_23.csv",index=None)
y_valid.to_csv(save_path+"data_y_valid_cut_corr_23.csv",index=None,header=None)

sparse.save_npz(save_path+"data_x_test_cut_corr_23.npz",data_x_test)
data_test.to_csv(save_path+"data_test_cut_corr_23.csv",index=None,)
df_feature_map.to_csv(save_path+"feature_important_mapping_cut_corr_23.csv")




#xgboost
clf=xgb.XGBClassifier(max_depth=5, learning_rate=0.05, n_estimators=1500, 
    objective='reg:logistic', nthread=7, min_child_weight=50, max_delta_step=0, 
    subsample=0.7, colsample_bytree=0.7, reg_lambda=1, 
    scale_pos_weight=1,seed=2018, missing='-1')

eval_list=[(data_x_train,y_train),(data_x_test,y_test)]
clf.fit(data_x_train, y_train, eval_set=eval_list, eval_metric='auc', early_stopping_rounds=100)

#sklearn 中的逻辑回归，跑不出来

LR_clf=LogisticRegression(penalty='l2', tol=0.00001, C=0.01,random_state=20180517, solver='lbfgs', max_iter=100)

LR_clf.fit(data_x_train,y_train)
joblib.dump(LR_clf, '/home/heqt/jupyter_project/model/LR_clf.pkl')

y_pre_LR=LR_clf.predict(data_x_test)
roc_auc_score(y_test,y_pre_LR)

plot_roc_curve(y_test,y_pre_LR)
plt.show()


#sklearn 中的随机梯度下降  

SGDLR_clf=SGDClassifier(loss='log', penalty='l1', alpha=1.0, l1_ratio=0.15, 
     random_state=20150511,learning_rate='optimal',class_weight='balanced',n_jobs=15)
SGDLR_clf.fit(data_x_train,y_train)
joblib.dump(gbm_clf, '/home/heqt/jupyter_project/model/feature_SGDLR_clf.pkl')

y_pre_SGDLR=SGDLR_clf.predict(data_x_test)
roc_auc_score(y_test,y_pre_SGDLR)

#lightgbm
eval_list=[(data_x_train,y_train),(data_x_valid,y_valid)]

gbm_clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=1000, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.02, min_child_weight=50,random_state=20183,n_jobs=7)

gbm_clf.fit(data_x_train,y_train,eval_set=eval_list,eval_metric ='auc',early_stopping_rounds =100)
joblib.dump(gbm_clf, '/home/heqt/jupyter_project/model/gbm_clf_cnt.pkl')
pre_gbm=gbm_clf.predict_proba(data_x_test)
result=data_test[['aid','uid']]
result['score']=pre_gbm[:,1]
result['score']=result['score'].apply(lambda x : '{:.8f}'.format(x))
result.to_csv('submission.csv',index=None)