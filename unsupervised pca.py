# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 22:56:19 2023

@author: Rehaman shaik
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 18:34:15 2023

@author: Rehaman shaik
"""
import numpy as np
import pandas as pd
d1=pd.read_csv("breast-cancer-wisconsin-data.csv")
d1
y=d1["diagnosis"]
x=d1.iloc[:,2:]
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
le.fit_transform(y)
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
ss_x=ss.fit_transform(x)
training_error=[]
test_error=[]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(ss_x, y,test_size=0.30)
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(x_train,y_train)
y_pred_train=logreg.predict(x_train)
y_pred_test=logreg.predict(x_test)
from sklearn.metrics import accuracy_score
training_error.append((accuracy_score(y_train, y_pred_train)))
test_error.append((accuracy_score(y_test, y_pred_test)))
print("average training  accuracy score: ",np.mean(training_error).round(3))
print("average test accuracy score: ",np.mean(test_error).round(3))

#--------------------------------------------------------------------
#validation
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
#---------------------------------------------------------------------------------
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=15)

#k1_train = []
#k1_test = []

from sklearn.model_selection import train_test_split

training_accuracy = []
test_accuracy = []

for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(SS_X,Y,test_size=0.30,random_state=i)
    KNN.fit(X_train,Y_train)
    Y_pred_train = KNN.predict(X_train)
    Y_pred_test  = KNN.predict(X_test)
    training_accuracy.append(accuracy_score(Y_train ,Y_pred_train))
    test_accuracy.append(accuracy_score(Y_test ,Y_pred_test))
    
import numpy as np
print("Average training accuracy:",np.mean(training_accuracy).round(3))
print("Average test accuracy:",np.mean(test_accuracy).round(3))

k1_train.append(np.mean(training_accuracy).round(3))
k1_test.append(np.mean(test_accuracy).round(3))

print(k1_train)
print(k1_test)


import matplotlib.pyplot as plt
plt.scatter(x=range(5,17,2),y=k1_train,color='blue')
plt.scatter(x=range(5,17,2),y=k1_test,color='red')
plt.plot(range(5,17,2),k1_train,color='red')
plt.plot(range(5,17,2),k1_test,color='black')
plt.show()
#=================pca=============
ss_x
from sklearn.decomposition import PCA
pca=PCA()
pc_df=pca.fit_transform(ss_x)
x_new=pc_df[:,0:5]
training_accuracy=[]
test_accuracy=[]

for i in range(1,101):
    x_train,x_test,y_train,y_test=train_test_split(x_new,y,test_size=0.30,random_state=i)
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