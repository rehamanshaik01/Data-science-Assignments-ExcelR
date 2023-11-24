# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 18:29:56 2023

@author: Rehaman shaik
"""

#k-fold cross validation 
import pandas as pd
d1=pd.read_csv("breast_cancer.csv")
d1
y=d1["Class"]
x=d1.iloc[:,1:10]
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder() 
le.fit_transform(y)
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
ss_x=ss.fit_transform(X)
training_accuracy=[]
test_accuracy=[]
from sklearn.model_selection import KFold
kf=KFold(n_splits=5)
for train_index,test_index in kf.split(ss_x):
 
    x_train,x_test=ss_x[train_index],ss_x[test_index]
    y_train,y_test=y[train_index],y[test_index]
    from sklearn.linear_model import LogisticRegression
    logreg=LogisticRegression()
    logreg.fit(x_train,y_train)
    y_pred_train=logreg.predict(x_train)
    y_pred_test=logreg.predict(x_test)
    from sklearn.metrics import accuracy_score
    training_accuracy.append(accuracy_score(y_train, y_pred_train))
    test_accuracy.append(accuracy_score(y_test, y_pred_test))
print("average training acuracy: ",np.mean(training_accuracy).round(3))
print("average test acuracy: ",np.mean(test_accuracy).round(3))
