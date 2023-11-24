# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 19:02:20 2023

@author: Rehaman shaik
"""

import pandas as pd
df = pd.read_csv("breast_cancer.csv")
df.shape
df.head()
list(df)
# step2: EDA

# step3: split as X and Y variable
Y = df["Class"]
X = df.iloc[:,1:10]

# step4: Data Transformation
# label encoding
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
Y = LE.fit_transform(Y)
Y

# standardization
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_X = SS.fit_transform(X)
pd.DataFrame(SS_X)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(SS_X,Y,test_size=0.30, random_state=10)
print("X_train data size: ", x_train.shape)
print("x_test data size: ", x_test.shape)
print("Y_train data size: ",y_train.shape)
print("Y_test data size: ",y_test.shape)
#step5 model fitting
from sklearn.linear_model import LogisticRegression
logreg= LogisticRegression()
logreg.fit(x_train,y_train)
#prediction
y_pred_train=logreg.predict(x_train)
y_pred_test=logreg.predict(x_test)
#step 6 : metrics
from sklearn.metrics import accuracy_score
ac1=accuracy_score(y_train,y_pred_train)
print(ac1.round(3))
ac2=accuracy_score(y_test, y_pred_test)
print(ac2.round(3))
 