# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 21:15:35 2019

@author: 12107
"""

'''
def fun(a=[1,1,1,1,1,2,2,2,2,2],x=[0,0,0],y=0,n=3,max_std=0,split_position=[]):
    if y>=n:
        return max_std=0,split_position=[]
    else:
        up=True
        for i in range(y,len(x)):
            if
'''            
        
    

def test():#测试函数，保存实验中间过程
    self=data_prepare()
    bm=self.building_metadata
    print(bm.max())
    #for i in bm.columns:
    #    print(i,'\n',pd.value_counts(bm[i]),'\n\n')
        
    #注意，一共只有15个site_id，各大楼分布在15个地区。各地区大楼数量不均衡；
    wtrain=self.weather_train
    train=self.train
    building_id=1
    site_id=0
    meter=0
    #0号楼，教育用途（寒暑假影响？）
    
    train_data=self.get_train_data(building_id=building_id,site_id=0,meter=0)
    
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
            data=train_data_daily[(train_data_daily.date<'2016-12-30')&(train_data_daily.date>'2016-01-01')])

    for i in ['air_temperature', 'cloud_coverage', 'dew_temperature',
       'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction',
       'wind_speed','wday']:
        sns.relplot(x='date', y=i ,
            kind='line',
            
            data=train_data_daily[(train_data_daily.date<'2016-08-30')&(train_data_daily.date>'2016-06-20')])
        
def test2():
     self=data_prepare()
     train_data=self.get_train_data_all()
     del self
     init_common_params= {
                'booster': 'gblinear',
                'objective': 'reg:linear',  
                #'num_class': 10,               # 类别数，与 multisoftmax 并用
                'gamma': 0.1,                  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
                'max_depth': 3,               # 构建树的深度，越大越容易过拟合
                'lambda': 0.5,                   # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
                'subsample': 0.7,              # 随机采样训练样本
                'colsample_bytree': 0.7,       # 生成树时进行的列采样
                'min_child_weight': 3,
                'silent': 0,                   # 设置成1则没有运行信息输出，最好是设置为0.
                'eta': 0.007,                  # 如同学习率
                #'seed': 1000,
                #'nthread': 4,                  # cpu 线程数，默认为最大可用线程数
                }  
     y=train_data.meter_reading
     x=train_data.drop(labels=['meter_reading','timestamp'],axis=1)
     x=x.fillna(-999)
     del train_data
     x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=1)
     del x,y
     dump_svmlight_file(x_test,y_test,'xgb_test.lib')
     del x_test, y_test
     dump_svmlight_file(x_train,y_train,'xgb_train.lib')
     
     del x_train, y_train
     #xgb_val = xgb.DMatrix(x_test,label=y_test)

     #del x_test,y_test,xgb_val
     
     #xgb_train = xgb.DMatrix(x_train, label=y_train)
     #xgb_train.save_binary('xgb_train')
     #del x_train,y_train,xgb_train
     xgb_val=xgb.DMatrix(data = 'xgb_test.lib#xgb_val.cache')
     xgb_train=xgb.DMatrix(data = 'xgb_train.lib#xgb_train.cache')
     params=init_common_params
     plst = params.items()
     num_rounds = 5000 # 迭代次数
     watchlist = [(xgb_train, 'train'),(xgb_val, 'val')]
     model = xgb.train(plst, xgb_train, num_rounds, watchlist,early_stopping_rounds=50)
     print(model.eval(xgb_val))
     y_predict=pd.DataFrame(model.predict(xgb_val),columns=['meter_reading'])
     y_predict['type']='predict'
     #y_test.index=range(len(y_test))
     #Y_test=pd.DataFrame(y_test,columns=['meter_reading'])
     #Y_test['type']='test'   
     #data=pd.concat([y_predict,Y_test])
     #data['index']=data.index
     #sns.relplot(x='index', y='meter_reading', kind='line',hue='type', data=data[(data['index']<1000)&(data['index']>800)])
