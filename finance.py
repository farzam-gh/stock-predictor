#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 20:14:48 2018

@author: farzam
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fix_yahoo_finance as yf
from sklearn.model_selection import train_test_split
import seaborn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
import datetime

#data = yf.download('AMZN',start='2008-01-01',end='2018-09-18')
#data.Close.plot()
#plt.show()

#import pandas_datareader.data as pdweb
pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import data as pdr
import fix_yahoo_finance # must pip install first
import app

stock='MSFT'

def get_data():
    global data
    now=datetime.datetime.now()
    curr=now.strftime('%Y-%m-%d')
    data = pdr.get_data_yahoo(stock,'2008-01-01',curr)


def eval_fold(i):
    y_train_pred=clf.predict(X_train)
    y_test_pred=clf.predict(X_test)
    r2_train=r2_score(y_train,y_train_pred)
    r2_test=r2_score(y_test,y_test_pred)
    mse_train=mean_squared_error(y_train,y_train_pred)
    mse_test=mean_squared_error(y_test,y_test_pred)
#    print('\nFold ',i)
#    print('\nR2_score_train: %.2f\tr2_score_test: %.2f'%(r2_test,r2_train))
#    print('mse_train: %.2f\tmse_test: %.2f'%(mse_train,mse_test))



def calc(X_today):

    np.set_printoptions(precision=3)
    n=int(np.shape(X_train)[0]/folds)
    for i in range(folds):
        begining=i*n
        end=(i+1)*n-(int(n/folds))
        clf.fit(X_train[begining:end,:],y_train[begining:end])
        r2_scores.append(r2_score(y_train[end:end+int(n/folds)],
                  clf.predict(X_train[end:end+int(n/folds),:])))
        mse_scores.append(mean_squared_error(y_train[end:end+int(n/folds)],
                  clf.predict(X_train[end:end+int(n/folds),:])))
        eval_fold(i)

    #clf.fit(X_train,y_train)
#    y_pred=clf.predict(X_test)
#    print('\nr2_scores:\n',r2_scores,
#          '\nmean-r2_scores:\n',np.mean(r2_scores))
#    print('\nmse_scores: \n',mse_scores,
#          '\nmean_mse_scores:\n',np.mean(mse_scores))
#    print(clf.predict(X_today))
    return clf.predict(X_today)


get_data()
r2_scores=[]
mse_scores=[]
folds=2
forest=RandomForestRegressor(n_estimators=1000,criterion='mse',
                     max_depth=5,n_jobs=-1)
lr=LinearRegression(normalize=True)
clf=lr
temp=np.empty((5,np.shape(data)[1]))
columns=data.columns
rows=['Next','2nd','3rd','forth','fifth']
predictions=np.empty((5,6))


#predictions=pd.DataFrame(temp,index=rows,columns=columns)
def weekly_predict():
    global X_test,X_train,y_test,y_train
    for i in range(1,6):
        for j in range(np.shape(data)[1]):
            cols=[]
            for k in range(np.shape(data)[1]):
                if(k!=j):
                    cols.append(k)
            global X
            X=data.iloc[:-1*i,cols].values
            global y
            y=data.iloc[i:,j].values
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,
                                                       random_state=0)
            X_today=data.iloc[-1:,cols].values

            predictions[i-1][j]='%.3f'%calc(X_today)
    return predictions
    #        print('X_today:\n',X_today)
    #        print('calc:\n',calc(X_today))

weekly_predict()
#print('\nPredictions:\n',predictions)


