# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 17:09:07 2023

@author: Rehaman shaik
"""
#import file
import pandas as pd
import numpy as np
d1=pd.read_csv("Cars_4vars.csv")
d1
d1.head()
d1["HP"]
d1["MPG"]
d1["VOL"]
d1["SP"]
d1["WT"]
#construct EDA
#compare all the variable with MPG
import matplotlib.pyplot as plt
plt.scatter(d1["HP"],d1["MPG"])
plt.show()
plt.scatter(d1["VOL"],d1["MPG"])
plt.show()
 plt.scatter(d1["SP"],d1["MPG"])
plt.show()
plt.scatter(d1["WT"],d1["MPG"])
plt.show()
#box plot
d1.boxplot(column="MPG",vert=False)
#histogram
d1["MPG"].hist()
#histogram and scatter plot together
import seaborn as sns
sns.pairplot(d1)
plt.show()
#correlation matrix
d1.corr()
#split x and y variables
y=d1["MPG"]
#x=d1[["HP"]]
#x=d1[["HP","VOL"]]
#x=d1[["HP","VOL","SP"]]
x=d1[["HP","VOL","SP","WT"]]
#fit the model 
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(x,y)
y_pred=LR.predict(x)
#metrics
from sklearn.metrics import mean_squared_error,r2_score
mse=mean_squared_error(y, y_pred)
print("mean square error: ", mse.round(3))#how muche errors we made 
print("root mean squared error : " , np.sqrt((mse).round(3)))
print("R square : ", r2_score(y, y_pred).round(3))#how much prediction we made correct
#building model
import statsmodels.formula.api as smf
model=smf.ols('MPG ~ HP + VOL',data=d1).fit()

print(model.summary())
model.fittedvalues
model.resid
import numpy as np
mse1=np.mean(model.resid **2)
mse1
====================
import statsmodels.formula.api as smf
model=smf.ols('MPG ~ HP + WT',data=d1).fit()

print(model.summary())
model.fittedvalues
model.resid
import numpy as np
mse1=np.mean(model.resid **2)
mse1
====================
import statsmodels.formula.api as smf
model=smf.ols('MPG ~ SP + VOL',data=d1).fit()

print(model.summary())
model.fittedvalues
model.resid
import numpy as np
mse1=np.mean(model.resid **2)
mse1
====================
import statsmodels.formula.api as smf
model=smf.ols('MPG ~ HP + WT',data=d1).fit()

print(model.summary())
model.fittedvalues
model.resid
import numpy as np
mse1=np.mean(model.resid **2)
mse1
====================
import statsmodels.formula.api as smf
model=smf.ols('MPG ~ SP + WT',data=d1).fit()

print(model.summary())
model.fittedvalues
model.resid
import numpy as np
mse1=np.mean(model.resid **2)
mse1

import statsmodels.formula.api as smf
model=smf.ols('SP ~ WT',data=d1).fit()

print(model.summary())
R=model.rsquared
vif=1/(1-R)
print(vif)