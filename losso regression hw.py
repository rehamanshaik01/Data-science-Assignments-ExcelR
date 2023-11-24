# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 13:07:17 2023

@author: Rehaman shaik
"""

import pandas as pd
df=pd.read_csv("Hitters_final.csv")
df
y=df["Salary"]
x=df.iloc[:,1:17]

from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
ss_x=ss.fit_transform(x)
training_error=[]
test_error=[]
for i in range(100):
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(ss_x,y,test_size=0.30,random_state=i)
    from sklearn.linear_model import LinearRegression
    le=LinearRegression()
    le.fit(x_train,y_train)
    y_pred_train=le.predict(x_train)
    y_pred_test=le.predict(x_test)
    from sklearn.metrics import mean_squared_error
    training_error.append(np.sqrt(mean_squared_error(y_train, y_pred_train)))
    test_error.append(np.sqrt(mean_squared_error(y_test, y_pred_test )))
    
print("average training error: ",np.mean(training_error).round(3))
print("average test error: ", np.mean(test_error).round(3))
#alpha=1.0
list(x)
ss_x_new=pd.DataFrame(ss_x)
ss_x_new.columns=list(x)
from sklearn.linear_model import Lasso
lass=Lasso(alpha=1.0)
lass.fit(x_train,y_train)
lass.coef_
a1=pd.DataFrame(lass.coef_)
le.coef_
lin_reg=pd.DataFrame(le.coef_)
xnames=pd.DataFrame(list(x))
d1=pd.concat([xnames,lin_reg,a1],axis=1)
list(d1)
d1.columns=["xanmes","lin_reg","a1"]
d1
x_new=ss_x_new.drop(ss_x_new.columns[[2,7,9]],axis=1)
training_error=[]
test_error=[]
for i in range(100):
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x_new, y,test_size=0.30,random_state=i)
    from sklearn.linear_model import LinearRegression
    le=LinearRegression()
    le.fit(x_train,y_train)
    y_pred_train=le.predict(x_train)
    y_pred_test=le.predict(x_test)
    from sklearn.metrics import mean_squared_error
    training_error.append(np.sqrt(mean_squared_error(y_train, y_pred_train  )))
    test_error.append(np.sqrt(mean_squared_error(y_test, y_pred_test)))
print("average training error: ",np.mean(training_error).round(3))
print("average test error: ", np.mean(test_error).round(3))

#alpha=5.0
from sklearn.linear_model import Lasso
lass=Lasso(alpha=5.0)
lass.fit(x_train,y_train)
lass.coef_
a5=pd.DataFrame(lass.coef_)
le.coef_
lin_reg=pd.DataFrame(le.coef_)
xnames=pd.DataFrame(list(x))
d1=pd.concat([xnames,lin_reg,a5],axis=1)
list(d1)
d1.columns=["xnames","lin_reg","a5"]
d1
x_new=ss_x_new.drop(ss_x_new.columns[[2,3,7,9,10,11,15]],axis=1)
training_error=[]
test_error=[]
for i in range(100):
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x_new, y,test_size=0.30,random_state=i)
    from sklearn.linear_model import LinearRegression
    le=LinearRegression()
    le.fit(x_train,y_train)
    y_pred_train=le.predict(x_train)
    y_pred_test=le.predict(x_test)
    from sklearn.metrics import mean_squared_error
    training_error.append(np.sqrt(mean_squared_error(y_train, y_pred_train  )))
    test_error.append(np.sqrt(mean_squared_error(y_test, y_pred_test )))
print("average training error: ",np.mean(training_error).round(3))
print("average test error: ", np.mean(test_error).round(3))
#alpha=7.0
list(x)
ss_x_new=pd.DataFrame(ss_x)
ss_x_new.columns=list(x)
from sklearn.linear_model import Lasso
lass=Lasso(alpha=7.0)
lass.fit(x_train,y_train)
lass.coef_
a3=pd.DataFrame(lass.coef_)
le.coef_
lin_reg=pd.DataFrame(le.coef_)
xnames=pd.DataFrame(list(x))
d3=pd.concat([xnames,lin_reg,a3],axis=1)
d3.columns=["xnames","lin_reg","a3"]
xnames=ss_x_new.drop(ss_x_new.columns[[2,3,6,7,9,10,11,12,15]],axis=1)
training_error=[]
test_error=[]
for i in range(100):
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x_new, y,test_size=0.30,random_state=i)
    from sklearn.linear_model import LinearRegression
    le=LinearRegression()
    le.fit(x_train,y_train)
    y_pred_train=le.predict(x_train)
    y_pred_test=le.predict(x_test)
    from sklearn.metrics import mean_squared_error
    training_error.append(np.sqrt(mean_squared_error(y_train, y_pred_train )))
    test_error.append(np.sqrt(mean_squared_error(y_test, y_pred_test)))
print("average training error: ",np.mean(training_error).round(3))
print("average test error: ", np.mean(test_error).round(3))
'''     train             test
       297.614             338.474
1.0    299.651             333.487
5.0    306                  331#we select alpha =5.0
7.0    268                  408'''
import seaborn as sns
import matplotlib.pyplot as plt

list(xnames)
variable=["AtBat","Hits","RBI","Walks","CHits","PutOuts","Assists"]
for i in variable:
    plt.hist(xnames[i])
    plt.xlabel(i)
    plt.ylabel("frequency")
    plt.title("histogram")
    plt.show( )
for i in variable:
    sns.boxplot(x=i,data=xnames)
    plt.title("boxplot")
    plt.show()
list(xnames)
xnames.columns
x_new_1=xnames.drop(xnames.columns[[4,5,6]],axis=1)
x_new_1
training_error=[]
test_error=[]
for i in range(100):
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x_new_1, y,test_size=0.30,random_state=i)
    from sklearn.linear_model import LinearRegression
    le=LinearRegression()
    le.fit(x_train,y_train)
    y_pred_train=le.predict(x_train)
    y_pred_test=le.predict(x_test)
    from sklearn.metrics import mean_squared_error
    training_error.append(np.sqrt(mean_squared_error(y_train, y_pred_train)))
    test_error.append(np.sqrt(mean_squared_error(y_test, y_pred_test)))
print("average training error after removing unnecessary variables: ", np.mean(training_error).round(3))
print("average test error after removing unnecessary variables:", np.mean(test_error).round(3))