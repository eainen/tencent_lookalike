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
run_day=datetime.now().strftime('%Y%m%d')
save_path="/home/heqt/tencent/"+run_day+"/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

hive_engine = create_engine('hive://heqt@192.168.254.107:10000',
                        connect_args={'auth':'KERBEROS','kerberos_service_name':'hive',
                        'configuration':{'mapred.job.queue.name':  'root.anti_fraud'}})

data_fit=pd.read_csv('/home/heqt/tencent/train_all.csv')
data_test=pd.read_csv('/home/heqt/tencent/test1_all.csv')
data_fit.loc[data_fit['label']==-1,'label']=0
data_test.insert(2,'label',-1)
data_x=pd.concat([data_fit,data_test])
data_x.fillna('-1',inplace=True)


#fit_model
one_hot_feature=['lbs','age','carrier','consumptionability','education','gender','house','os','ct','marriagestatus','advertiserid','campaignid', 'creativeid',
       'adcategoryid', 'creativesize','productid', 'producttype']
vector_feature=['appidaction','appidinstall','interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3','topic1','topic2','topic3']
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

x_train,x_valid,y_train,y_valid= train_test_split(data_fit,data_y,test_size=0.3, random_state=2018)

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
#CVec=CountVectorizer(analyzer='word',token_pattern=r'(?u)\b\w+\b',tokenizer =lambda x: x.split(' '))
CVec=CountVectorizer()
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
df_feature_map.to_csv(save_path+"feature_important_mapping.csv")

sparse.save_npz(save_path+"data_x_train.npz",data_x_train)
x_train.to_csv(save_path+"x_train.csv",index=None)
y_train.to_csv(save_path+"data_y_train.csv",index=None)

sparse.save_npz(save_path+"data_x_valid.npz",data_x_valid)
x_valid.to_csv(save_path+"x_valid.csv",index=None)
y_valid.to_csv(save_path+"data_y_valid.csv",index=None)

sparse.save_npz(save_path+"data_x_test.npz",data_x_test)
data_test.to_csv(save_path+"data_test.csv",index=None)
result=data_test[['aid','uid']]
result.to_csv(save_path+"result.csv",index=None)





#xgboost
clf=xgb.XGBClassifier(max_depth=5, learning_rate=0.05, n_estimators=1500, 
    objective='reg:logistic', nthread=7, min_child_weight=50, max_delta_step=0, 
    subsample=0.7, colsample_bytree=0.7, reg_lambda=1, 
    scale_pos_weight=1,seed=2018, missing='-1')

eval_list=[(data_x_train,y_train),(data_x_test,y_test)]
clf.fit(data_x_train, y_train, eval_set=eval_list, eval_metric='auc', early_stopping_rounds=100)

#sklearn 中的逻辑回归，跑不出来

LR_clf=LogisticRegression(penalty='l1', tol=0.0001, C=1.0,class_weight='balanced', 
    random_state=20181, solver='liblinear', max_iter=100, n_jobs=7)

LR_clf.fit(data_x_train,y_train)
joblib.dump(LR_clf, '/home/heqt/jupyter_project/model/LR_clf.pkl')

y_pre_LR=LR_clf.predict(data_x_test)
roc_auc_score(y_test,y_pre_LR)

plot_roc_curve(y_test,y_pre_LR)
plt.show()


#sklearn 中的随机梯度下降  

SGDLR_clf=SGDClassifier(loss='log', penalty='l1', alpha=0.0001, l1_ratio=0.15, 
     n_jobs=3, random_state=20182,learning_rate='optimal',class_weight='balanced')
SGDLR_clf.fit(data_x_train,y_train)
joblib.dump(gbm_clf, '/home/heqt/jupyter_project/model/SGDLR_clf.pkl')
y_pre_SGDLR=SGDLR_clf.predict(data_x_test)
roc_auc_score(y_test,y_pre_SGDLR)

#lightgbm
eval_list=[(data_x_train,y_train),(data_x_valid,y_valid)]

gbm_clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=1000, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.02, min_child_weight=50,random_state=20183,n_jobs=7,tree_learner='feature')

gbm_clf.fit(data_x_train,y_train,eval_set=eval_list,eval_metric ='auc',early_stopping_rounds =100)
joblib.dump(gbm_clf, '/home/heqt/jupyter_project/model/gbm_clf_2.pkl')
pre_gbm=gbm_clf.predict_proba(data_x_test)
result=data_test[['aid','uid']]
result['score']=pre_gbm[:,1]
result['score']=result['score'].apply(lambda x : '{:.8f}'.format(x))
result.to_csv('submission.csv',index=None)