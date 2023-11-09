# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 18:08:07 2023

@author: Rehaman shaik
"""
import numpy as np

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
ss_x=ss.fit_transform(x)
from sklearn.model_selection import train_test_split
training_accuracy=[]
test_accuracy=[]

for i in range(1000):
    x_train,x_test,y_train,y_test=train_test_split(ss_x,y,test_size=0.30,random_state=i)
    from sklearn.linear_model import LogisticRegression
    logreg=LogisticRegression()
    logreg.fit(x_train,y_train)
    y_pred_train=logreg.predict(x_train)
    y_pred_test=logreg.predict(x_test)
    from sklearn.metrics import accuracy_score
    training_accuracy.append(accuracy_score(y_train, y_pred_train))
    test_accuracy.append(accuracy_score(y_test, y_pred_test))
print("average training accuracy",np.mean(training_accuracy).round(3))

print("average test accuracy",np.mean(test_accuracy).round(3))