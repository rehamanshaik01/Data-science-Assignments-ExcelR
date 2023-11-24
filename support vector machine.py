# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 17:51:17 2023

@author: Rehaman shaik
"""
import numpy as np
import pandas as pd
d1=pd.read_csv("createdata.csv")
d1
y=d1["Y"]
x=d1.iloc[:,:2]
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
ss_x=ss.fit_transform(x)
x=pd.DataFrame(ss_x)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(ss_x, y,train_size=0.75)
from sklearn.svm import SVC
svc=SVC(C=1.0,kernel="linear")
svc.fit(x_train,y_train)
y_pred_train=svc.predict(x_train)
y_pred_test=svc.predict(x_test)
from sklearn.metrics import accuracy_score
print("average training accuracy score: ",accuracy_score(y_train,y_pred_train))
print("average testing accuracy score: ",accuracy_score(y_test, y_pred_test))

import pandas as pd
d1=pd.read_csv("createdata.csv")
d1
y=d1["Y"]
x=d1.iloc[:,:3]

from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
ss_x=ss.fit_transform(x)
x=pd.DataFrame(ss_x)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(ss_x, y,train_size=0.75)
#=========================
#linear
#=====================
from sklearn.svm import SVC
svc=SVC(C=1.0,kernel="linear")
svc.fit(x_train,y_train)
y_pred_train=svc.predict(x_train)
y_pred_test=svc.predict(x_test)
from sklearn.metrics import accuracy_score
print("average training accuracy score: ",accuracy_score(y_train,y_pred_train))
print("average testing accuracy score: ",accuracy_score(y_test, y_pred_test))


from mlxtend.plotting import plot_decision_regions
plot_decision_regions(X=x.values,
                      y=y.values, 
                      clf=svc, 
                      legend=4)
#polynomial
from sklearn.svm import SVC
svc=SVC(kernel="poly",degree=3)
svc.fit(x_train,y_train)
y_pred_train=svc.predict(x_train)
y_pred_test=svc.predict(x_test)
from sklearn.metrics import accuracy_score
print("average training accuracy score: ",accuracy_score(y_train,y_pred_train))
print("average testing accuracy score: ",accuracy_score(y_test, y_pred_test))

#rbf
from sklearn.svm import SVC
svc=SVC(kernel="linear")
svc.fit(x_train,y_train)
y_pred_train=svc.predict(x_train)
y_pred_test=svc.predict(x_test)
from sklearn.metrics import accuracy_score
print("average training accuracy score: ",accuracy_score(y_train,y_pred_train))
print("average testing accuracy score: ",accuracy_score(y_test, y_pred_test))