import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split # train and test 
from sklearn import metrics
from sklearn.metrics import classification_report

Election = pd.read_csv("C:/Users/SHAJIUDDIN MOHAMMED/Desktop/Data sets/election_data.csv")

Election.columns = 'Election-id','Result','Year','AS','PR'

Election1 = Election.drop("Election-id", axis = 1)

Election1.isna().sum()

# Imputation of NA values by median
Election1.fillna(Election.median(), inplace=True)
Election1.isna().sum()
Election.info()

# Creating a new column for output
Election1['Result'] = np.where(Election.Result > 0,1,0)

#Dropping the original output column
Election1.drop("Result", axis = 1)
Election1.columns

### Splitting the data into train and test data 
# from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(Election1, test_size = 0.2) # 20% test data

y = Election1.iloc[:, 0]
x = Election.iloc[:, 1:]
# Model building 
# import statsmodels.formula.api as sm


logit_model = sm.logit('Result ~ Year+AS+PR' , data = Election1).fit()

#summary
logit_model.summary2() # for AIC
logit_model.summary()

pred = logit_model.predict(Election1.iloc[:, 1 :])

#ROC CURVE AND AUC
fpr, tpr, thresholds = metrics.roc_curve(Election1.Result, pred)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
optimal_threshold # 0.7937672180919426

Election1["pred"] = np.zeros(11)

Election1.loc[pred > optimal_threshold, "pred"] = 1

classification = classification_report(Election1["Result"], Election1["pred"])
classification

#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")


roc_auc = metrics.auc(fpr, tpr) 
roc_auc # 0.9666666666666668

train_data,test_data = train_test_split(Election1, test_size = 0.2)


model = sm.logit('Result ~  AS + PR', data = train_data).fit()
model.summary()

#test data

test_pred = model.predict(test_data)

test_data["test_pred"] = np.zeros(3)
test_data.loc[test_pred > optimal_threshold, "test_pred"] = 1

confusion_matrix = pd.crosstab(test_data.test_pred, test_data.Result)
confusion_matrix

accuracy_test = (2+1)/(3)
accuracy_test # 1

#roc for test data

fpr, tpr, thresholds = metrics.roc_curve(test_data.Result,test_data.pred )

roc_auc = metrics.auc(fpr, tpr) 
roc_auc
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")


# train data

train_pred = model.predict(train_data)

train_data["train_pred"] = np.zeros(8)
train_data.loc[train_pred > optimal_threshold, "train_pred"] = 1

confusion_matrix = pd.crosstab(train_data.train_pred, train_data.Result)
confusion_matrix

accuracy_test = (5+3)/(8)
accuracy_test # 1

#roc for test data

fpr, tpr, thresholds = metrics.roc_curve(train_data.Result,train_data.pred )
roc_auc = metrics.auc(fpr, tpr) 
roc_auc # 0.8333333333333333

plt.plot(fpr, tpr)

 


