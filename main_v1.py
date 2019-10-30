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
        return train_data
    def get_train_data_daily(self,train_data):
        train_data['date']=train_data['timestamp'].apply(lambda x:x.split(' ')[0])
        train_data['wday']=train_data['timestamp'].apply(lambda x:time.strptime(x,'%Y-%m-%d %H:%M:%S')[-3])
        train_data_daily=pd.DataFrame()
        train_data_daily['date']=train_data['date'].drop_duplicates()
        for i in ['air_temperature', 'cloud_coverage', 'dew_temperature',
                  'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction',
                  'wind_speed','wday']:
            train_data_daily[i]=train_data_daily['date'].apply(lambda x:np.mean(train_data[train_data['date']==x][i]))
        train_data_daily['meter_reading']=train_data_daily['date'].apply(lambda x:np.sum(train_data[train_data['date']==x]['meter_reading']))
        return train_data_daily

def test():#测试函数，保存实验中间过程
    self=data_prepare()
    bm=self.building_metadata
    print(bm.max())
    for i in bm.columns:
        print(i,'\n',pd.value_counts(bm[i]),'\n\n')
        
    #注意，一共只有15个site_id，各大楼分布在15个地区。各地区大楼数量不均衡；
    wtrain=self.weather_train
    train=self.train
    building_id=0
    site_id=0
    meter=0
    #0号楼，教育用途（寒暑假影响？）
    
    train_data=self.get_train_data(building_id=0,site_id=0,meter=0)
    
    #plt.figure()
    sns.relplot(x='timestamp', y='meter_reading',
            kind='line',
            #data=train_data[(train_data.timestamp<'2016-06-30')&(train_data.timestamp>'2016-05-30')])
            data=train_data[(train_data.timestamp<'2016-06-30')&(train_data.timestamp>'2016-06-20')])

    for i in ['air_temperature', 'cloud_coverage', 'dew_temperature',
       'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction',
       'wind_speed']:
        sns.relplot(x='timestamp', y=i ,
            kind='line',
            #data=train_data[(train_data.timestamp<'2016-06-30')&(train_data.timestamp>'2016-05-30')])
            data=train_data[(train_data.timestamp<'2016-06-30')&(train_data.timestamp>'2016-06-20')])
    #plt.show()
    train_data_daily=self.get_train_data_daily(train_data)
    sns.relplot(x='date', y='meter_reading',
            kind='line',
            #data=train_data[(train_data.timestamp<'2016-06-30')&(train_data.timestamp>'2016-05-30')])
            data=train_data_daily[(train_data_daily.date<'2016-08-30')&(train_data_daily.date>'2016-06-20')])

    for i in ['air_temperature', 'cloud_coverage', 'dew_temperature',
       'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction',
       'wind_speed','wday']:
        sns.relplot(x='date', y=i ,
            kind='line',
            
            data=train_data_daily[(train_data_daily.date<'2016-08-30')&(train_data_daily.date>'2016-06-20')])
        
        
        