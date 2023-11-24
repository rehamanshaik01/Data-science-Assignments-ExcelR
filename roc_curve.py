"""
Created on Fri Oct  6 18:44:51 2023
"""

# step1: import the file
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

# step5: Model fitting
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

logreg.fit(SS_X,Y)
Y_pred = logreg.predict(SS_X)

# step6: Metrics
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(Y,Y_pred)
cm
print("Accuacy score:", accuracy_score(Y,Y_pred).round(2))

#------------------------------------------------------
from sklearn.metrics import recall_score,precision_score,f1_score
print("Sensitivity score:", recall_score(Y,Y_pred).round(3))
print("Precision score:", precision_score(Y,Y_pred).round(3))
print("F1 score:", f1_score(Y,Y_pred).round(3))

cm
TN = cm[0,0]
FP = cm[0,1]
TNR = TN/(TN + FP)
print("Specificity:", TNR.round(3))

#------------------------------------------------------

# exact probabilities
logreg.predict_proba(SS_X) # 1- prob, prob
Y_probabilities = logreg.predict_proba(SS_X)[:,1:]


#------------------------------------------------------
# ROC CURVE

from sklearn.metrics import roc_curve,roc_auc_score
fpr,tpr,dummy = roc_curve(Y,Y_probabilities)

import matplotlib.pyplot as plt
plt.scatter(x = fpr,y=tpr)
plt.plot(fpr,tpr,color='red')
plt.ylabel("True positive Rate")
plt.xlabel("False positive Rate")
plt.show()

print("AUC score:", roc_auc_score(Y,Y_probabilities).round(3))