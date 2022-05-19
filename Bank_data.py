import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split # train and test 
from sklearn import metrics
from sklearn.metrics import classification_report

bank = pd.read_csv("C:/Users/SHAJIUDDIN MOHAMMED/Desktop/Data sets/bank_data.csv")

bank.columns = 'age', 'default', 'balance', 'housing', 'loan', 'duration', 'campaign','pdays', 'previous', 'poutfailure', 'poutother', 'poutsuccess','poutunknown', 'con_cellular', 'con_telephone', 'con_unknown','divorced', 'married', 'single', 'joadmin', 'joblue','joentrepreneur', 'johousemaid', 'jomanagement', 'joretired','joselfemployed', 'joservices', 'jostudent', 'jotechnician','jounemployed', 'jounknown', 'y'

bank.isna().sum() # No NA Values

bank['y'] = np.where(bank.y > 0,1,0)

bank.drop("y", axis = 1)

bank = bank.iloc[:, [31, 0,1, 2, 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]]
bank.columns

#bank = bank.rename({"joadmin.":"jodamin","joblue.collar":"joblue_collar","joself.employed":"joself_employed"},axis=1)

train_data, test_data = train_test_split(bank, test_size = 0.3) # 30% test data

y = bank.iloc[:, 0]
x = bank.iloc[:, 1:]
# Model building 

logit_model = sm.logit('y ~age + default + balance + housing + loan + duration + campaign + pdays + previous + poutfailure + poutother + poutsuccess + poutunknown + con_cellular + con_telephone + con_unknown + divorced+married+single+joadmin+joblue+joentrepreneur+johousemaid+jomanagement+joretired+joselfemployed+joservices+jostudent+jotechnician+jounemployed+jounknown', data = bank).fit()

logit_model.summary2() # for AIC
logit_model.summary()

pred = logit_model.predict(bank.iloc[:, 1 :])

fpr, tpr, thresholds = metrics.roc_curve(bank.y, pred)

optimal_idx = np.argmax(tpr - fpr)

optimal_threshold = thresholds[optimal_idx]

optimal_threshold # 0.1147422078766177

train_data,test_data = train_test_split(bank, test_size = 0.3)

model = sm.logit('y ~ age + default + balance + housing + loan + duration + campaign + pdays+ previous + poutfailure + poutother + poutsuccess + con_cellular + con_telephone + divorced + married + joadmin + joblue + joentrepreneur + johousemaid + jomanagement + joselfemployed + joservices + jostudent + jotechnician + jounemployed', data = train_data).fit()

pred = model.predict(bank)

model.summary()

#test data

test_pred = model.predict(test_data)

test_data["test_pred"] = np.zeros(13564)

test_data.loc[test_pred > optimal_threshold, "test_pred"] = 1

confusion_matrix = pd.crosstab(test_data.test_pred, test_data.y)
confusion_matrix

accuracy_test = (9799+1294)/(13564)
accuracy_test # 0.817826599823061

# train data
train_pred = model.predict(train_data)

train_data["train_pred"] = np.zeros(31647)
train_data.loc[train_pred > optimal_threshold, "train_pred"] = 1

confusion_matrix = pd.crosstab(train_data.train_pred, train_data.y)
confusion_matrix

accuracy_test = (22822+3022)/(31647)
accuracy_test #  0.8166334881663349

#roc for test data
fpr, tpr, thresholds = metrics.roc_curve(train_data.y,train_data.train_pred )
roc_auc = metrics.auc(fpr, tpr) 
roc_auc # 0.8148783789206663

plt.plot(fpr, tpr)
