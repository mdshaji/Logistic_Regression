import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split # train and test 
from sklearn import metrics
from sklearn.metrics import classification_report

Affairs = pd.read_csv("C:/Users/SHAJIUDDIN MOHAMMED/Desktop/Data sets/Affairs.csv")

Affairs.columns = 'S.No','naffairs', 'kids', 'vryunhap', 'unhap', 'avgmarr','hapavg', 'vryhap', 'antirel', 'notrel', 'slghtrel', 'smerel', 'vryrel','yrsmarr1', 'yrsmarr2', 'yrsmarr3', 'yrsmarr4', 'yrsmarr5', 'yrsmarr6'

Affairs1 = Affairs.drop("S.No", axis = 1)

Affairs1.isna().sum() # No NA values found

Affairs.info()

Affairs1['affairs'] = np.where(Affairs.naffairs > 0,1,0)

Affairs1.drop("naffairs", axis = 1)
Affairs

### Splitting the data into train and test data 
# from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(Affairs1, test_size = 0.3) # 30% test data

y = Affairs1.iloc[:, 0]
x = Affairs.iloc[:, 1:]
# Model building 
# import statsmodels.formula.api as sm


logit_model = sm.logit('affairs ~ kids + vryunhap + unhap + avgmarr + hapavg +vryhap + antirel + notrel + slghtrel + smerel + vryrel + yrsmarr1 +yrsmarr2 +yrsmarr3 +yrsmarr4 + yrsmarr5 +yrsmarr6' , data = Affairs1).fit()

#summary
logit_model.summary2() # for AIC
logit_model.summary()

pred = logit_model.predict(Affairs1.iloc[:, 1 :])

#ROC CURVE AND AUC
fpr, tpr, thresholds = metrics.roc_curve(Affairs1.affairs, pred)

optimal_idx = np.argmax(tpr - fpr)

optimal_threshold = thresholds[optimal_idx]

optimal_threshold # 0.2521571570135329

Affairs1["pred"] = np.zeros(601)

Affairs1.loc[pred > optimal_threshold, "pred"] = 1

classification = classification_report(Affairs1["affairs"], Affairs1["pred"])

classification
#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")


roc_auc = metrics.auc(fpr, tpr)  #AUC = 0.720879526977088
roc_auc

train_data,test_data = train_test_split(Affairs1, test_size = 0.3)

model = sm.logit('affairs ~ kids + vryunhap + unhap + avgmarr + hapavg +vryhap + antirel + notrel + slghtrel + smerel + vryrel + yrsmarr1 +yrsmarr2 +yrsmarr3 +yrsmarr4 + yrsmarr5 +yrsmarr6', data = train_data).fit()

model.summary()

#test data

test_pred = model.predict(test_data)

test_data["test_pred"] = np.zeros(181)
test_data.loc[test_pred> optimal_threshold, "test_pred"] = 1

confusion_matrix = pd.crosstab(test_data.test_pred, test_data.affairs)
confusion_matrix

accuracy_test = (88+32)/(181)
accuracy_test # 0.6629834254143646

#roc for test data
fpr, tpr, thresholds = metrics.roc_curve(test_data.affairs,test_data.pred )
roc_auc = metrics.auc(fpr, tpr) 
roc_auc # 0.7018082282680822

plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")


# train data
train_pred = model.predict(train_data)

train_data["train_pred"] = np.zeros(420)
train_data.loc[train_pred > optimal_threshold, "train_pred"] = 1

confusion_matrix = pd.crosstab(train_data.train_pred, train_data.affairs)
confusion_matrix

accuracy_test = (208+73)/(420)
accuracy_test # 0.669047619047619

#roc for test data
fpr, tpr, thresholds = metrics.roc_curve(train_data.affairs,train_data.pred )
roc_auc = metrics.auc(fpr, tpr) 
roc_auc # 0.6694207426991949

plt.plot(fpr, tpr)


