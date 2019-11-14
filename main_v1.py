# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 20:43:07 2019

@author: 12107
"""
import time
import pandas as pd
import xgboost as xgb
import numpy as np
import seaborn as sns
import tensorflow as tf
import sklearn as skt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import dump_svmlight_file
import pickle
import json
class data_prepare(object):
    def __init__(self,path='./data/'):
        self.building_metadata=pd.read_csv(path+'building_metadata.csv')
        self.sample_submission=pd.read_csv(path+'sample_submission.csv')
        self.test=pd.read_csv(path+'test.csv')
        self.train=pd.read_csv(path+'train.csv')
        self.weather_test=pd.read_csv(path+'weather_test.csv')
        self.weather_train=pd.read_csv(path+'weather_train.csv')
        print('data reading done')
    def get_train_data(self,building_id=0,site_id=0,meter=0):
        meter_data_train=self.train[(self.train['building_id']==building_id)&(self.train['meter']==meter)]
        weather_data_train=self.weather_train[self.weather_train['site_id']==site_id]
        train_data=pd.merge(left=meter_data_train,right=weather_data_train)
        train_data['wday']=train_data['timestamp'].apply(lambda x:time.strptime(x,'%Y-%m-%d %H:%M:%S')[-3]) 
        train_data['month']=train_data['timestamp'].apply(lambda x:x.split(' ')[0].split('-')[1]).apply(int)
        train_data['day']=train_data['timestamp'].apply(lambda x:x.split(' ')[0].split('-')[2]).apply(int)
        return train_data
    def get_train_data_by_use(self,primary_use='Education',meter=0):
        train_data=pd.merge(left=self.train,right=self.building_metadata,how='left')
        train_data=pd.merge(left=train_data,right=self.weather_train,how='left')
        train_data=train_data[(train_data['primary_use']==primary_use)&(train_data['meter']==meter)]
        del train_data['primary_use'],train_data['meter']
        train_data['wday']=train_data['timestamp'].apply(lambda x:time.strptime(x,'%Y-%m-%d %H:%M:%S')[-3]) 
        train_data['month']=train_data['timestamp'].apply(lambda x:x.split(' ')[0].split('-')[1]).apply(int)
        train_data['day']=train_data['timestamp'].apply(lambda x:x.split(' ')[0].split('-')[2]).apply(int)
        train_data['hour']=train_data['timestamp'].apply(lambda x:x.split(' ')[1].split(':')[0]).apply(int)
        train_data['is_holiday']=0

        train_data.loc[(train_data['wday']) == 5 | (train_data['wday'] == 6) , 'is_holiday'] = 1
        print(train_data['is_holiday'].mean())
        del train_data['timestamp']
        return train_data   
    def get_test_data_by_use(self,primary_use='Education',meter=0):
        bm=self.building_metadata[self.building_metadata['primary_use']==primary_use]
        test_data=self.test[self.test['meter']==meter]
        test_data=test_data[test_data['building_id'].isin(bm['building_id'].values)]
        test_data=pd.merge(left=test_data,right=bm,how='left')
        test_data=pd.merge(left=test_data,right=self.weather_test,how='left')
        #test_data=test_data[(test_data['primary_use']==primary_use)&(test_data['meter']==meter)]
        del test_data['primary_use'],test_data['meter']
        test_data['wday']=test_data['timestamp'].apply(lambda x:time.strptime(x,'%Y-%m-%d %H:%M:%S')[-3]) 
        test_data['month']=test_data['timestamp'].apply(lambda x:x.split(' ')[0].split('-')[1]).apply(int)
        test_data['day']=test_data['timestamp'].apply(lambda x:x.split(' ')[0].split('-')[2]).apply(int)
        test_data['hour']=test_data['timestamp'].apply(lambda x:x.split(' ')[1].split(':')[0]).apply(int)
        test_data['is_holiday']=0
        test_data.loc[(test_data['wday'] == 5) | (test_data['wday'] == 6) , 'is_holiday'] = 1
        print(test_data['is_holiday'].mean())
        test_data.index=test_data['row_id']
        del test_data['timestamp'],test_data['row_id']
        return test_data        
    
    def get_train_data_all(self):
        train_data=pd.merge(left=self.train,right=self.building_metadata,how='left')
        train_data=pd.merge(left=train_data,right=self.weather_train,how='left')
        train_data['primary_use']=pd.factorize(train_data.primary_use)[0]
        train_data['wday']=train_data['timestamp'].apply(lambda x:time.strptime(x,'%Y-%m-%d %H:%M:%S')[-3]) 
        train_data['month']=train_data['timestamp'].apply(lambda x:x.split(' ')[0].split('-')[1]).apply(int)
        train_data['day']=train_data['timestamp'].apply(lambda x:x.split(' ')[0].split('-')[2]).apply(int)
        return train_data  

    def get_test_data_all(self):
        test_data=pd.merge(left=self.test,right=self.building_metadata,how='left')
        test_data=pd.merge(left=test_data,right=self.weather_test,how='left')
        test_data['primary_use']=pd.factorize(test_data.primary_use)[0]
        test_data['wday']=test_data['timestamp'].apply(lambda x:time.strptime(x,'%Y-%m-%d %H:%M:%S')[-3]) 
        test_data['month']=test_data['timestamp'].apply(lambda x:x.split(' ')[0].split('-')[1]).apply(int)
        test_data['day']=test_data['timestamp'].apply(lambda x:x.split(' ')[0].split('-')[2]).apply(int)
        return test_data      
    def get_train_data_daily(self,train_data):
        train_data['date']=train_data['timestamp'].apply(lambda x:x.split(' ')[0])
        train_data_daily=pd.DataFrame()
        train_data_daily['date']=train_data['date'].drop_duplicates()
        for i in ['air_temperature', 'cloud_coverage', 'dew_temperature',
                  'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction',
                  'wind_speed','wday','month','day']:
            train_data_daily[i]=train_data_daily['date'].apply(lambda x:np.mean(train_data[train_data['date']==x][i]))
        train_data_daily['meter_reading']=train_data_daily['date'].apply(lambda x:np.sum(train_data[train_data['date']==x]['meter_reading']))
        return train_data_daily

    def find_optimal_split_point(self,meter_reading=[1,1,1,1,1,2,2,2,2,2]):
        '''
        由于一年之中，一个大楼可能同时存在多种能耗模式，随季节变换进行切换，因此需要对训练数据进行切割、分段；
        不同时间段内的数据分别作为单独的样本进行训练；
        分段原则为，分段后，各段内能耗表读数的平均值的标准差最大；
        如10天内能耗表的读数为[1,1,1,1,1,2,2,2,2,2],那么最优分割点的数量为1，分割位置为5，标准差为np.std([1,2]),即0.5；
        PS1:执行效率太低了，增加变量月/日，通过模型自动学习到该信息
        或按周进行分割
        '''
        optimal_split_point_nums=0
        optimal_split_position=[]
        max_std=0
        
        #1个分割点
        for i in range(1,len(meter_reading)-1):
            if np.std([np.mean(meter_reading[:i]),np.mean(meter_reading[i:])])>max_std:
                optimal_split_position=[i]
                optimal_split_point_nums=1
                max_std=np.std([np.mean(meter_reading[:i]),np.mean(meter_reading[i:])])
            print(i,optimal_split_point_nums,optimal_split_position,max_std)
        
        #2个分割点            
        for i in range(1,len(meter_reading)-2):
            for j in range(2,len(meter_reading)-1):
                if np.std([np.mean(meter_reading[:i]),np.mean(meter_reading[i:j]),np.mean(meter_reading[j:])])>max_std:
                    optimal_split_position=[i,j]
                    optimal_split_point_nums=2  
                    max_std=np.std([np.mean(meter_reading[:i]),np.mean(meter_reading[i:j]),np.mean(meter_reading[j:])])
            print(i,optimal_split_point_nums,optimal_split_position,max_std)           
        #3个分割点            
        for i in range(1,len(meter_reading)-3):
            for j in range(2,len(meter_reading)-2):
                for k in range(3,len(meter_reading)-1):
                    if np.std([np.mean(meter_reading[:i]),np.mean(meter_reading[i:j]),np.mean(meter_reading[j:k]),np.mean(meter_reading[k:])])>max_std:
                        optimal_split_position=[i,j,k]
                        optimal_split_point_nums=3      
                        max_std=np.std([np.mean(meter_reading[:i]),np.mean(meter_reading[i:j]),np.mean(meter_reading[j:k]),np.mean(meter_reading[k:])])
                print(i,j,optimal_split_point_nums,optimal_split_position,max_std)
        return optimal_split_point_nums,optimal_split_position,max_std

class model_train(object):
    def __init__(self):
        self.init_common_params= {
                'booster': 'gbtree',

                'gamma': 0,                  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
                'max_depth': 7,  
                'alpha': 0.45,
                'lambda': 0.45,                   # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
                'subsample': 0.6,              # 随机采样训练样本
                'colsample_bytree': 0.4,       # 生成树时进行的列采样
                'min_child_weight': 1.5,
                'silent': 0,                   # 设置成1则没有运行信息输出，最好是设置为0.
                #'eta': 0.1,                  # 如同学习率
                #'eval_metric':'rmsle'
                #'seed': 1000,
                #'nthread': 4,                  # cpu 线程数，默认为最大可用线程数
                }  
        self.class_params_dict={}
        #同一类大楼采用相同参数作为自动调参的起点
    def base_train_cell_v1(self,train_data,params):
        #params=self.init_common_params
        y=train_data.meter_reading
        x=train_data.drop(labels=['building_id','meter','meter_reading','timestamp','site_id'],axis=1)
        x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=1)
        xgb_val = xgb.DMatrix(x_test,label=y_test)
        xgb_train = xgb.DMatrix(x_train, label=y_train)
        
        plst = params.items()
        num_rounds = 5000 # 迭代次数
        watchlist = [(xgb_train, 'train'),(xgb_val, 'val')]
        model = xgb.train(plst, xgb_train, num_rounds, watchlist,early_stopping_rounds=50)
        print(model.eval(xgb_val))
        y_predict=pd.DataFrame(model.predict(xgb_val),columns=['meter_reading'])
        y_predict['type']='predict'
        y_test.index=range(len(y_test))
        Y_test=pd.DataFrame(y_test,columns=['meter_reading'])
        Y_test['type']='test'   
        data=pd.concat([y_predict,Y_test])
        data['index']=data.index
        sns.relplot(x='index', y='meter_reading', kind='line',hue='type', data=data[(data['index']<1000)&(data['index']>800)])
        pass
    def train_by_use(self,train_data,params,version=1,primary_use='Education',meter=0,learning_rates=[0.2]*300+[0.1]*200+[0.07]*100,num_rounds = 600 ):
        y=train_data.meter_reading
        x=train_data.drop(labels=['meter_reading'],axis=1)
    
        del train_data
        x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=1)
        xgb_val = xgb.DMatrix(x_test,label=y_test)

        xgb_train = xgb.DMatrix(x_train, label=y_train)
        del x_train, x_test, y_train, y_test,x,y

        init_common_params= {
                'booster': 'gbtree',

                'gamma': 0.1,                  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
                'max_depth': 5,               # 构建树的深度，越大越容易过拟合
                'lambda': 0.2,                   # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
                'subsample': 0.7,              # 随机采样训练样本
                'colsample_bytree': 0.7,       # 生成树时进行的列采样
                'min_child_weight': 3,
                'silent': 0,                   # 设置成1则没有运行信息输出，最好是设置为0.
                #'eta': 0.1,                  # 如同学习率
                #'eval_metric':'rmsle'
                #'seed': 1000,
                #'nthread': 4,                  # cpu 线程数，默认为最大可用线程数
                }       
        #learning_rates=[0.3]*300+[0.2]*200+[0.1]*100
        params=init_common_params
        plst = params.items()
        #num_rounds = 5000 # 迭代次数
        #num_rounds = 600 # 迭代次数
        watchlist = [(xgb_train, 'train'),(xgb_val, 'val')]
        model = xgb.train(plst, xgb_train, num_rounds, watchlist,obj=squarederrorobj,feval=eval_metric,learning_rates=learning_rates,early_stopping_rounds=50)
        r=model.eval_set(watchlist,feval=eval_metric)
        del xgb_train,xgb_val,watchlist
        date=time.strftime('%m%d',time.localtime())
        f_path='./model/{}_{}_{}_{}.txt'.format(primary_use,meter,date,version)
        model.dump_model(f_path)
        with open('./params/{}_{}_{}_{}.txt'.format(primary_use,meter,date,version),'w') as f:
            f.write(json.dumps(params))
                
            
        return model,r
    
    def predict_by_use(self,model,test_data):
        row_id_list=test_data.index
     
        xgb_test=xgb.DMatrix(test_data)
        del test_data
        y_predict=model.predict(xgb_test)
        y_predict=y_predict*(1+np.sign(y_predict))/2
        y_predict=pd.DataFrame(y_predict,columns=['meter_reading'],index=row_id_list)
        return y_predict
    
    

def eval_metric(preds, dtrain):
    labels = dtrain.get_label()
    preds=preds*(1+np.sign(preds))/2
    #preds=np.array([i if i>0 else 0 for i in preds])
    l=np.log(labels+1)
    p=np.log(preds+1)
    loss=np.mean((l-p)**2)**0.5
    return 'rmsle',loss

def squarederrorobj(preds, dtrain):
    labels = dtrain.get_label()
    preds=preds*(1+np.sign(preds))/2
    #preds=np.array([i if i>0 else 0 for i in preds])
    l=np.log(labels+1)
    p=np.log(preds+1)    
    grad = (p-l)/(preds+1)
    hess = (1+l+p)/(preds+1)**2
    return grad, hess



def main(version=1,date='1114'):
    dp=data_prepare()
    primary_use_list=dp.building_metadata.primary_use.drop_duplicates().values
    meter_list=[0,1,2,3]
    mt=model_train()
    params=mt.init_common_params.copy()
    params['nthread']=3
    eval_history={}
    predict_result=[]
    
    for p in primary_use_list:
        eval_history[p]={}
        for m in meter_list:
            train_data=dp.get_train_data_by_use(primary_use=p,meter=m)
            if len(train_data)>0:
                model,r=mt.train_by_use(train_data,mt.init_common_params)
                del train_data
                eval_history[p][m]=r
                test_data=dp.get_test_data_by_use(primary_use=p,meter=m)
                predict_df=mt.predict_by_use(model,test_data)
                del test_data
                predict_result.append(predict_df)
    predict_result=pd.concat(predict_result)
    predict_result.sort_index(inplace=True)
    predict_result.to_csv('./submission/{}_v{}.csv'.format(date,version))
    eval_history_df=pd.DataFrame(eval_history).T
    eval_history_df.to_csv('./eval_history/eh{}_v{}.csv'.format(date,version))        


def test3():
     self=data_prepare()
     train_data=self.get_train_data_by_use(primary_use='Education',meter=1)
     

     y=train_data.meter_reading
     x=train_data.drop(labels=['meter_reading','timestamp'],axis=1)
     #x=x.fillna(-999)
     del train_data
     x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=1)
     #del x,y
     #dump_svmlight_file(x_test,y_test,'xgb_test.lib')
     #del x_test, y_test
     #dump_svmlight_file(x_train,y_train,'xgb_train.lib')
     
     #del x_train, y_train
     xgb_val = xgb.DMatrix(x_test,label=y_test)

     #del x_test,y_test,xgb_val
     
     xgb_train = xgb.DMatrix(x_train, label=y_train)
     del x_train, x_test, y_train, y_test
     #xgb_train.save_binary('xgb_train')
     #del x_train,y_train,xgb_train
     #xgb_val=xgb.DMatrix(data = 'xgb_test.lib#xgb_val.cache')
     #xgb_train=xgb.DMatrix(data = 'xgb_train.lib#xgb_train.cache')
     '''
     init_common_params= {
                'booster': 'gblinear',
                'lambda': 0.2,                   # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。

                'silent': 0,                   # 设置成1则没有运行信息输出，最好是设置为0.
                
                'feature_selector':'shuffle'

                }  
     '''
     init_common_params= {
                'booster': 'gbtree',

                'gamma': 0.1,                  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
                'max_depth': 5,               # 构建树的深度，越大越容易过拟合
                'lambda': 0.2,                   # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
                'subsample': 0.7,              # 随机采样训练样本
                'colsample_bytree': 0.7,       # 生成树时进行的列采样
                'min_child_weight': 3,
                'silent': 0,                   # 设置成1则没有运行信息输出，最好是设置为0.
                #'eta': 0.1,                  # 如同学习率
                #'eval_metric':'rmsle'
                #'seed': 1000,
                #'nthread': 4,                  # cpu 线程数，默认为最大可用线程数
                }       
     #learning_rates=[0.3]*500+[0.25]*500+[0.2]*500+[0.15]*500+[0.1]*500+[0.05]*950+[0.3]*50+[0.03]*950+[0.2]*50+[0.01]*450+[0.1]*50
     learning_rates=[0.3]*200+[0.2]*100+[0.1]*100
     params=init_common_params
     plst = params.items()
     #num_rounds = 5000 # 迭代次数
     num_rounds = 400 # 迭代次数
     watchlist = [(xgb_train, 'train'),(xgb_val, 'val')]
     model = xgb.train(plst, xgb_train, num_rounds, watchlist,obj=squarederrorobj,feval=eval_metric,learning_rates=learning_rates,early_stopping_rounds=100)
     
     test_data=self.get_test_data_by_use(primary_use='Education',meter=1)
     row_id_list=test_data.index
     
     xgb_test=xgb.DMatrix(test_data)
     del test_data
     y_predict=model.predict(xgb_test)
     y_predict=y_predict*(1+np.sign(y_predict))/2
     y_predict=pd.DataFrame(y_predict,columns=['meter_reading'],index=row_id_list)
     
     #y_test.index=range(len(y_test))
     #Y_test=pd.DataFrame(y_test,columns=['meter_reading'])
     #Y_test['type']='test'   
     #data=pd.concat([y_predict,Y_test])
     #data['index']=data.index
     #sns.relplot(x='index', y='meter_reading', kind='line',hue='type', data=data[(data['index']<1000)&(data['index']>800)])
                  