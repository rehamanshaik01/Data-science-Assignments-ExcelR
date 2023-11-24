# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 18:44:58 2023

@author: Rehaman shaik
"""
import numpy as np
import pandas as pd
d1=pd.read_csv("mushroom.csv")


for columns in d1.columns:
    from sklearn.preprocessing import LabelEncoder
    le=LabelEncoder()
    d1[columns]=le.fit_transform(d1[columns])
print(d1)
y=d1.iloc[:,:1].values
y = y.ravel()
y.ndim
x=d1.iloc[:,1:]
x
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x, y,test_size=0.30)
from sklearn.naive_bayes import MultinomialNB
ga= MultinomialNB()
ga.fit(x_train,y_train)
y_pred_train=ga.predict(x_train)
y_pred_test=ga.predict(x_test)
from sklearn.metrics import accuracy_score
print("average accuracy trraining score: ",accuracy_score(y_train, y_pred_train))

print("average accuracy test score: ",accuracy_score(y_test, y_pred_test))

accuracy_train=[]
accuracy_test=[]
for i in range(1,100):
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x, y,test_size=0.30,random_state=i)
    from sklearn.naive_bayes import MultinomialNB
    ga= MultinomialNB()
    ga.fit(x_train,y_train)
    y_pred_train=ga.predict(x_train)
    y_pred_test=ga.predict(x_test)
    from sklearn.metrics import accuracy_score
    accuracy_train.append(accuracy_score(y_train, y_pred_train))
    
    accuracy_test.append(accuracy_score(y_test, y_pred_test))
        
print("average accuracy trraining score: ",np.mean(accuracy_train).round(3))

print("average accuracy test score: ",np.mean(accuracy_test).round(3))