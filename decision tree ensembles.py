import pandas as pd
import numpy as np
d1=pd.read_csv("Boston.csv")
d1
y=d1["medv"]
x=d1.iloc[:,1:14]
x.columns
training_error=[]
test_error=[]
for i in range(1,100):
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x, y,test_size=0.30,random_state=i)
    from sklearn.tree import DecisionTreeRegressor
    dtr=DecisionTreeRegressor()
    dtr.fit(x_train, y_train)
    y_pred_train=dtr.predict(x_train)
    y_pred_test=dtr.predict(x_test)
    from sklearn.metrics import mean_squared_error
    training_error.append(np.sqrt(mean_squared_error(y_train, y_pred_train)))
    test_error.append(np.sqrt(mean_squared_error(y_test, y_pred_test)))
print("TRAINING RMSE=",np.mean(training_error).round(3))
print("TEST RMSE=",np.mean(test_error).round(3))
print("Variance:",np.mean(test_error).round(3)-np.mean(training_error).round(3))
#============BAGGING==========
for i in range(100):
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x, y,test_size=0.30,random_state=i)
    from sklearn.ensemble import BaggingRegressor
    bag=BaggingRegressor(estimator=DecisionTreeRegressor() ,
                        n_estimators=100,
                        max_samples=0.6,
                        max_features=0.7)
    bag.fit(x_train, y_train)
    y_pred_train=bag.predict(x_train)
    y_pred_test=bag.predict(x_test)
    from sklearn.metrics import mean_squared_error
    training_error.append(np.sqrt(mean_squared_error(y_train, y_pred_train)))
    test_error.append(np.sqrt(mean_squared_error(y_test, y_pred_test)))
print("TRAINING RMSE=",np.mean(training_error).round(3))
print("TEST RMSE=",np.mean(test_error).round(3))
print("Variance:",np.mean(test_error).round(3)-np.mean(training_error).round(3))
#==========Random forest========
for i in range(1,100):
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x, y,test_size=0.30)
    from sklearn.ensemble import RandomForestRegressor
    rr=RandomForestRegressor(n_estimators=100,
                             max_samples=0.6,
                             max_features=0.7,max_depth=8)
    rr.fit(x_train, y_train)
    y_pred_train=rr.predict(x_train)
    y_pred_test=rr.predict(x_test)
    from sklearn.metrics import mean_squared_error
    training_error.append(np.sqrt(mean_squared_error(y_train, y_pred_train)))
    test_error.append(np.sqrt(mean_squared_error(y_test, y_pred_test)))
print("TRAINING RMSE=",np.mean(training_error).round(3))
print("TEST RMSE=",np.mean(test_error).round(3))
print("Variance:",np.mean(test_error).round(3)-np.mean(training_error).round(3))
#======================gradeint boosting=======
for i in range(1,100):#estimor we can increase to maintain varience and learning rate
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x, y,test_size=0.30)
    from sklearn.ensemble import GradientBoostingRegressor
    gbr=GradientBoostingRegressor(n_estimators=100,learning_rate=0.1,max_features=0.7)
    gbr.fit(x_train, y_train)
    y_pred_train=gbr.predict(x_train)
    y_pred_test=gbr.predict(x_test)
    from sklearn.metrics import mean_squared_error
    training_error.append(np.sqrt(mean_squared_error(y_train, y_pred_train)))
    test_error.append(np.sqrt(mean_squared_error(y_test, y_pred_test)))
print("TRAINING RMSE=",np.mean(training_error).round(3))
print("TEST RMSE=",np.mean(test_error).round(3))
print("Variance:",np.mean(test_error).round(3)-np.mean(training_error).round(3))
#============================adaBoost=====================
from sklearn.ensemble import AdaBoostRegressor
ABR = AdaBoostRegressor(n_estimators=500,learning_rate=0.01,
                                random_state=100)
training_error = []
test_error = []

for i in range(1,100):
    from sklearn.model_selection import train_test_split
    X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.30,random_state=i)
    ABR.fit(X_train,Y_train)
    Y_pred_train = ABR.predict(X_train)
    Y_pred_test  = ABR.predict(X_test)
    from sklearn.metrics import mean_squared_error
    training_error.append(np.sqrt(mean_squared_error(Y_train ,Y_pred_train)))
    test_error.append(np.sqrt(mean_squared_error(Y_test ,Y_pred_test)))

import numpy as np
print("Average training Error:",np.mean(training_error).round(3))
print("Average test Error:",np.mean(test_error).round(3))
print("ABR-variance:",np.mean(test_error).round(3)-np.mean(training_error).round(3))
#==================XGBOOST========================================

from xgboost import XGBRegressor
XGB=XGBRegressor(gamma=40,learning_rate=0.1,reg_lambda=0.5,n_estimators=400) 
training_error = []
test_error = []

for i in range(1,100):
    from sklearn.model_selection import train_test_split
    X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.30,random_state=i)
    XGB.fit(X_train,Y_train)
    Y_pred_train = XGB.predict(X_train)
    Y_pred_test  = XGB.predict(X_test)
    from sklearn.metrics import mean_squared_error
    training_error.append(np.sqrt(mean_squared_error(Y_train ,Y_pred_train)))
    test_error.append(np.sqrt(mean_squared_error(Y_test ,Y_pred_test)))

import numpy as np
print("Average training Error:",np.mean(training_error).round(3))
print("Average test Error:",np.mean(test_error).round(3))
print("XGB-variance:",np.mean(test_error).round(3)-np.mean(training_error).round(3))