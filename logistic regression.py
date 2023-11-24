# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 18:43:41 2023

@author: Rehaman shaik
"""
#1. impport file
import pandas as pd
d1=pd.read_csv("breast_cancer.csv")
d1
d1.columns
list(d1)


#2. EDA
d1.hist()

#3. split x and y variables
y=d1["Class"]
x=d1.iloc[:,1:10]


#4. data transformation
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)
     #standardzation data transformation for variables in x 
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
ss_x=ss.fit_transform(x)


#5. model fittimg
from sklearn.linear_model import LogisticRegression
logr=LogisticRegression()
logr.fit(x,y) ||  logr.fit(ss_x,y)
y_pred=logr.predict(ss_x)


#6. matrix
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y, y_pred)
cm
print("accuraruy score", accuracy_score(y, y_pred).round(2))