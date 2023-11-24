# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 17:42:50 2023

@author: Rehaman shaik
"""

import pandas as pd 
import numpy as np
df=pd.read_csv("Hitters_final.csv")
df
y=df["Salary"]
x=df.iloc[:,1:17]
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
ss_x=ss.fit_transform(x)
train_error=[]
test_error=[]

for i in range(1000):
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(ss_x,y,test_size=0.30,random_state=i)
    from sklearn.linear_model import LinearRegression
    le=LinearRegression()
    le.fit(x_train,y_train)
    y_pred_train=le.predict(x_train)
    y_pred_test=le.predict(x_test)
    from sklearn.metrics import mean_squared_error
    train_error.append(np.sqrt(mean_squared_error(y_train, y_pred_train)))
    test_error.append(np.sqrt(mean_squared_error(y_test, y_pred_test)))
print("average training error: " , np.mean(train_error).round(3))
print("average test error:", np.mean(test_error).round(3))

#=================================================================================
#losso regression
list(x)
ss_x_new=pd.DataFrame(ss_x)
ss_x_new.columns=list(x)
from sklearn.linear_model import Lasso
#lass=Lasso(alpha=1.0)
lass=Lasso(alpha=5.0)
lass.fit(x_train,y_train)
lass.coef_
#a1=pd.DataFrame(lass.coef_)
a5=pd.DataFrame(lass.coef_)
lin_reg=pd.DataFrame(le.coef_)
xnames=pd.DataFrame(list(x))
d1=pd.concat([xnames,lin_reg,a5], axis=1)
d1.columns = ["xnames","lin_reg","a1"]
xnames_new=ss_x_new.drop(ss_x_new.columns[11],axis=1)
training_error=[]
test_error=[]
for i in range(100):
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(xnames_new,y,test_size=0.30,random_state=i)
    from sklearn.linear_model import LinearRegression
    le=LinearRegression()
    le.fit(x_train,y_train)
    y_pred_train=le.predict(x_train)
    y_pred_test=le.predict(x_test)
    from sklearn.metrics import mean_squared_error
    training_error.append(np.sqrt(mean_squared_error(y_train,y_pred_train)))#rmse
    test_error.append(np.sqrt(mean_squared_error(y_test, y_pred_test)))
print("average training error :", np.mean(training_error))
print("average test error: ", np.mean(test_error))