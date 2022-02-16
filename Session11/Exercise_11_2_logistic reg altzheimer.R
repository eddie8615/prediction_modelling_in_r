###### Session 11: Regularized Regression III ##########
###### Logistic regularized logistic regression - Practical 2  ####


################################################################################
### Data set from Applied Predictive Modeling (2013) by Kuhn and Johnson.
### Copyright 2013 Kuhn and Johnson
### Web Page: http://www.appliedpredictivemodeling.com
### Section 19.6 Case Study: Predicting Cognitive Impairment
### from Chapter 19: An Introduction to Feature Selection
### Data used: The Alzheimer disease data from the AppliedPredictiveModeling 

# Schapiro et al. (2011) Multiplexed immunoassay panel identifies novel CSF biomarkers for
# Alzheimer's disease diagnosis and prognosis. PlosOne 19(4). Clinicopathological studies 
# suggest that Alzheimer's disease (AD) pathology begins ~10-15 years before the resulting
# cognitive impairment draws medical attention. 
# Biomarkers that can detect AD pathology in its early stages and predict dementia onset would,
# therefore, be invaluable for patient care and efficient clinical trial design. 
# The authors utilized a targeted proteomics approach to discover novel cerebrospinal fluid (CSF)
# biomarkers that can augment the diagnostic and prognostic accuracy of current leading 
# CSF biomarkers (A?42, tau, p-tau181).
# Using a multiplexed Luminex platform, 130 analytes (features variables) were measured in
# 333 CSF samples from cognitively normal (Clinical Dementia Rating [CDR] 0) and
# mildly demented (CDR 1) individuals.
##########################################################################################
# Research question: 
# Do identified novel candidate biomarkers together with the best current CSF biomarkers 
# reliable distinguish mildly demented from cognitively normal individuals?
#########################################################################################



 
################################################################################
##### 1. Regularized logistic regression using glmnet
library(glmnet)
library(psych)
library(caret)


# Load the library Applied Prediction modelling with data set
library(AppliedPredictiveModeling)
data(AlzheimerDisease)
# There is one file "predictors" with 130 potential predictors of AD diagnosis
summary(predictors)
ncol(predictors)
# and one data frame called diagnosis with the diagnosis of AD (Impaired=1) and not impaired "Control"=2
summary(diagnosis)

# The factor variable Genotype has got 6 alleles: E2E2, E2E3, E2E4,E3E2, E3E2, E3E3.
# They need to be dummy coded meaning that each value in Genotype variable will be a separate binary variable 
# Model matrix can be used to create dummy coded variables
predictors2<-model.matrix( ~ .-1, data=predictors)  # this formula creates 6 dummy coded variables for genotype

# Create an x and y file
x<-predictors2
y<-as.factor(diagnosis)
as.numeric(y)
#  1= Ad and 2 = control,
# In this case, we need to change the values to 0 and 1 rather than 1 and 2.
# This leads to fail to fit logistic regression because logistic regression treats the outcome only 0 and 1

# Manipulating the outcome variable by subtracting by 1
y <- as.factor(as.numeric(y)-1)

## Now start with logistic regularized regressions 
## You should present at the end sensitivity, specificity, accuracy and AUC
# glmnet package is used in this session

# Regularised logistic regression model (LASSO)
model <- glmnet(x, y, family="binomial", alpha=1)

plot(model, "lambda")
print(model)

set.seed(122)

# Regularised Logistic regression is fitted through cross-validation 
cv.model <- cv.glmnet(x,y, family="binomial", alpha=1)
plot(cv.model, xvar="lambda")

# Regularising parameter that minimises MSE 
cv.model$lambda.min #0.1217499
# Minimum lambda + 1SE (Standard Error) -> penalise more
cv.model$lambda.1se # 0.03387764

# Regularised coefficients
coef(cv.model, s="lambda.min")  # 37 coefficients left
coef(cv.model, s="lambda.1se")  # 17 coefficients left

y_prob<-predict(cv.model,type="response",newx=x, s = "lambda.min")

y_pred<-as.numeric(predict(cv.model,type="class",newx=x, s = "lambda.min"))

table(y, y_pred)

# Result
confusionMatrix(as.factor(y_pred), y)

# Accuracy: 0.9309 [0.8982-0.9557]
# Sensitivity: 0.8132
# Specificity: 0.9752

roc(y, y_pred)
plot(roc(y, y_pred))

# AUC: 0.8942
# The model is really good at accuracy as well as sensitivity and specificity


# If we use lambda.1se instead lambda.min
y_prob<-predict(cv.model,type="response",newx=x, s = "lambda.1se")

y_pred<-as.numeric(predict(cv.model,type="class",newx=x, s = "lambda.1se"))

table(y, y_pred)

confusionMatrix(y, as.factor(y_pred))

# Accuracy: 0.9099 [0.8737-0.9384]
# Sensitivity: 0.9296
# Specificity: 0.9046

# In this model the overall accuracy is slightly decreased by 3% compared to the previous model
# Also, the specificity is also dropped by about 7%
# However, the sensitivity increases by about 11%
# If the research does really take account to more on TP than TN, the latter model would be more plausible choice.
###################################################################################################
###### 2. Optional: you may want to do a repeated CV using the caret package. 
library(caret)
library(glmnet)
library(doParallel)
# Load the library Applied Prediction modelling with data set
library(AppliedPredictiveModeling)
data(AlzheimerDisease)

# Create an x and y file
# There is no need to dummy code your categorical predictors as long as they are defined as factors
x<-predictors
y<-as.numeric(diagnosis)-1

# Comment: There is no need to dummy code categorical variables using caret!
# Merge x and y in a new dataframe object called dat
dat<-as.data.frame(cbind(y,x))
# The outcome needs to be defined as a factor
dat$y<-as.factor(y) 
levels(dat$y) <- c("AD", "Control") 
summary(dat)

## Now start with logistic regularized regressions using repeated CV to identify the best model
cl=makeCluster(7);registerDoParallel(cl)

trControl <- trainControl(method = "repeatedcv", # repeated CV 
                          repeats = 10,          # number of repeated CV
                          number = 10   ,         # Number of folds
                          summaryFunction = twoClassSummary,  #function to compute performance metrics across resamples.AUC for binary outcomes
                          classProbs = TRUE, 
                          savePredictions = "all",
                          allowParallel = TRUE,
                          selectionFunction = "best" ) # best - minimum lambda, oneSE for minimum lambda + 1 Se, Tolerancwe for minimum + 3%

# Grid hyperparameter search
# We consider Ridge, LASSO, and Elastic net
params = expand.grid(alpha=seq(1,0,by=-0.1),   # L1 & L2 mixing parameter
                     lambda=2^seq(1,-10, by=-0.1)) # regularization parameter

glmnet.obj <- train(y ~ .,             # model formula (. means all features)  <---
                    data = dat,           # data.frame containing training set   <-!!!
                    method = "glmnet",     # model to use
                    metric ="ROC",         # Optimizes AUC, best with deviance for unbalanced outcomes 
                    family="binomial",     # logistic regression
                    trControl = trControl, # set training settings
                    tuneGrid = params)     # set grid of params to test over, if not specified defualt gris is used (not always the best)
stopCluster(cl) # Stop the use of cores!

# Plot performance for different parameters
plot(glmnet.obj, xTrans=log, xlab="log(lambda)")




# Plot regularization paths for the best model
plot(glmnet.obj$finalModel, xvar="lambda", label=T)


# Summary of main results 
print(glmnet.obj)

# See the content of the object 
summary(glmnet.obj)


get_best_result = function(caret_fit) {
  best = which(rownames(caret_fit$results) == rownames(caret_fit$bestTune))
  best_result = caret_fit$results[best, ]
  rownames(best_result) = NULL
  best_result
}

# Get some accuracy measures best on the estimates of hold-out data, still over-optimistic due to model selection  
get_best_result(glmnet.obj)

best_alpha <-get_best_result(glmnet.obj)$alpha
best_alpha # shpuld be 1
best_lambda <- get_best_result(glmnet.obj)$lambda
best_lambda  # 

# Finally, we get LASSO model with lambda = 0.008974206

predictions_prob<-(predict(glmnet.obj,s=best_lambda, alpha=best_alpha, type="prob",newx=x))
predictions_class<-(predict(glmnet.obj,s=best_lambda, alpha=best_alpha, type="raw",newx=x))

confusionMatrix(as.factor(predictions_class), dat$y, positive="AD")
###################################################################################################
# 3. Optional: Try to estimate the internal validity of the model by nested cross-validation
# What's the difference between internal and apparent AUC?



######################################################################################################################
# The data needs to have categorical and binary variables as factors, and continous and ordinal variables as numeric #
######################################################################################################################

# OUTCOME: is the last variable ##############################
# Check levels of outcome: the lowest level should be "no event"
# Use levels(outcome) to check that the first vector component here is the lowest level
# x and y need to be in one object (unlike glmnet)
# y needs to be defined as a factor
# the data set needs to be a dataframe (unlike glmnet)


# Load the library Applied Prediction modelling with data set
# install.packages("AppliedPredictiveModeling")

# Importing data file from package
library(AppliedPredictiveModeling)
# There is one file "predictors" with 130 potential predictors of AD diagnosis
data(AlzheimerDisease)
predictors
summary(predictors)
ncol(predictors)
# and one data frame called diagnosis with the diagnosis of AD (Impaired=1) and not impaired "Control"=2
summary(diagnosis)

# The factor variable Genotype has got 6 alleles: E2E2, E2E3, E2E4,E3E2, E3E2, E3E3.
# They need to be dummy coded
# Model matrix can be used to create dummy coded variables
predictors2<-model.matrix( ~ .-1, data=predictors)  # this formula creates 6 dummy coded variables for genotype

# The script needs y in the last column
# We need to use the combined data set with x and y
# The outcome needs to be defined as a factor

# Create an x and y file
x<-predictors2
y<-as.numeric(diagnosis)
y=y-1  # change outcomes into 0 and 1
# Merge x and y in a new dataframe object called dat
data<-as.data.frame(cbind(x,y))
# The outcome needs to be defined as a factor
data$y<-as.factor(y) 
levels(data$y) <- c("AD", "Controls")  # levels 0 and 1 aren't valid names in R and you need to label your two levels

summary(data)


