# Load the Dataset

Election <- read.csv(file.choose())

sum(is.na(Election))
View(Election)

Election1 <- na.omit(Election) # Omitting NA values from the Data 
# na.omit => will omit the rows which has atleast 1 NA value
dim(Election1)

summary(Election1)

sum(is.na(Election1))
dim(Election1)


# Preparing a linear regression 
mod_lm <- lm(Result ~ ., data = Election1)
summary(mod_lm)

pred1 <- predict(mod_lm, Election1)
pred1
# plot(claimants$CLMINSUR, pred1)

# We can also include NA values but where ever it finds NA value
# probability values obtained using the glm will also be NA 
# So they can be either filled using imputation technique or
# exlclude those values 


# GLM function use sigmoid curve to produce desirable results 
# The output of sigmoid function lies in between 0-1

model <- glm(Result ~ ., data = Election1, family = "binomial")
summary(model)

# To calculate the odds ratio manually we going r going to take exp of coef(model)
exp(coef(model))

# Predicition to check model validation
prob <- predict(model, Election1, type = "response")
prob
# We are going to use NULL and Residual Deviance to compare the between different models

# Confusion matrix and considering the threshold value as 0.5 
confusion <- table(prob > 0.5, Election1$Result)
confusion

# Model Accuracy 
Acc <- sum(diag(confusion)/sum(confusion))
Acc 


# Data Partitioning
n <-  nrow(Election1)
n1 <-  n * 0.85
n2 <-  n - n1
train_index <-  sample(1:n, n1)
train <- Election1[train_index, ]
test <-  Election1[-train_index, ]

# Train the model using Training data
finalmodel <- glm(Result ~ ., data = train, family = "binomial")
summary(finalmodel)

# Prediction on test data
prob_test <- predict(model, newdata = test, type = "response")
prob_test

# Confusion matrix 
confusion <- table(prob_test > 0.5, test$Result)
confusion

# Model Accuracy 
Accuracy <- sum(diag(confusion)/sum(confusion))
Accuracy 

# Creating empty vectors to store predicted classes based on threshold value
pred_values <- NULL

pred_values <- ifelse(prob_test > 0.5, 1, 0)

# Creating new column to store the above values
test[,"prob"] <- prob_test
test[,"pred_values"] <- pred_values

table(test$Result, test$pred_values)


# Compare the model performance on Train data
# Prediction on test data
prob_train <- predict(model, newdata = train, type = "response")
prob_train

# Confusion matrix 
confusion_train <- table(prob_train > 0.5, train$Result)
confusion_train

# Model Accuracy 
Acc_train <- sum(diag(confusion_train)/sum(confusion_train))
Acc_train



# Additional metrics
# Calculate the below metrics
# precision | recall | True Positive Rate | False Positive Rate | Specificity | Sensitivity

# ROC Curve => used to evaluate the betterness of the logistic model
# more area under ROC curve better is the model 
# We will use ROC curve for any classification technique not only for logistic

install.packages(ROCR)
library(ROCR)
rocrpred <- prediction(prob, Election1$Result)
rocrperf <- performance(rocrpred, 'tpr', 'fpr')

str(rocrperf)

plot(rocrperf, colorize=T, text.adj=c(-0.2,1.7))
# More area under the ROC Curve better is the logistic regression model obtained

## Getting cutt off or threshold value along with true positive and false positive rates in a data frame 
rocr_cutoff <- data.frame(cut_off = rocrperf@alpha.values[[1]], fpr=rocrperf@x.values, tpr=rocrperf@y.values)

colnames(rocr_cutoff) <- c("cut_off","FPR","TPR")
View(rocr_cutoff)

library(dplyr)
rocr_cutoff$cut_off <- round(rocr_cutoff$cut_off, 6)
# Sorting data frame with respect to tpr in decreasing order 
rocr_cutoff <- arrange(rocr_cutoff, desc(TPR))
View(rocr_cutoff)


