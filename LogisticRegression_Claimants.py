
import pandas as pd
import numpy as np
# import seaborn as sb
import matplotlib.pyplot as plt

import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split # train and test 
from sklearn import metrics
from sklearn.metrics import classification_report

#Importing Data
claimants1 = pd.read_csv("C:/Datasets_BA/Logistic regression/claimants.csv",sep=",")


#removing CASENUM
claimants1 = claimants1.drop('CASENUM', axis = 1)
claimants1.head(11)
claimants1.isna().sum()

# Imputating the missing values           
########## Median Imputation ############
claimants1.fillna(claimants1.median(), inplace=True)
claimants1.isna().sum()


### Splitting the data into train and test data 
# from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(claimants1, test_size = 0.3) # 30% test data

# Model building 
# import statsmodels.formula.api as sm
logit_model = sm.logit('ATTORNEY ~ CLMAGE + LOSS + CLMINSUR + CLMSEX + SEATBELT', data = train_data).fit()

#summary
logit_model.summary2() # for AIC
logit_model.summary()

#prediction
train_pred = logit_model.predict(train_data.iloc[ :, 1: ])

# Creating new column 
# filling all the cells with zeroes
train_data["train_pred"] = np.zeros(938)

# taking threshold value as 0.5 and above the prob value will be treated as correct value 
train_data.loc[train_pred > 0.5, "train_pred"] = 1

# classification report
classification = classification_report(train_data["train_pred"], train_data["ATTORNEY"])
classification

# confusion matrix
confusion_matrx = pd.crosstab(train_data.train_pred, train_data['ATTORNEY'])
confusion_matrx

accuracy_train = (308 + 358)/(938)
print(accuracy_train)

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(train_data["ATTORNEY"], train_pred)

#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")


roc_auc = metrics.auc(fpr, tpr)  #AUC = 0.76


# Prediction on Test data set
test_pred = logit_model.predict(test_data)

# Creating new column for storing predicted class of Attorney
# filling all the cells with zeroes
test_data["test_pred"] = np.zeros(402)

# taking threshold value as 0.5 and above the prob value will be treated as correct value 
test_data.loc[test_pred > 0.5, "test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['ATTORNEY'])

confusion_matrix
accuracy_test = (124 + 158)/(402) 
accuracy_test

# Based on ROC curv we can say that cut-off value should be 0.60, We can select it and check the acccuracy again.
