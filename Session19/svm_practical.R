## R Code for the SVM (introduction) Practical
###########################################################
############################################################

## Load the required R libraries
library(e1071)
library(caret)

## Read the data into R
data<-read.csv("heart.csv")

## See first few rows of the data
head(data)

## See the dimension of data
dim(data)
## rows=462, cols=10

## Check if the class variable is a factor (required by R function svm)
is.factor(data$class)

## Convert to factor (if not already)
data$class<-as.factor(data$class)
is.factor(data$class)

## See the number of people in each class
table(data$class)  

## Randomly split the data into training and test sets
## (consider 2/3:1/3 split)
set.seed(123)
n<-nrow(data) 
train<- ifelse(runif(n)<0.67,1,0)
table(train)

## Separate the training and test data
traindata<-data[train==1,]
dim(traindata)
## rows=304, cols=10
table(traindata$class)/nrow(traindata)
##  CHD      Healthy 
## 0.3123028 0.6876972

testdata<-data[train==0,]
dim(testdata)
## rows=158, cols=10
table(testdata$class)/nrow(testdata)
##   CHD Healthy 
## 0.4206897 0.5793103

## Train a linear kernel SVM with default cost (C=1)
####################################################
svmLin<-svm(class~ ., data=traindata, kernel="linear", probability=TRUE)
summary(svmLin)

##test set predictions
predLin<-predict(svmLin,testdata, probability=TRUE)

## See mis/correct classifications
trueClass<-testdata$class
table(predLin, trueClass)
##        trueClass
##predLin   CHD Healthy
##  CHD      21      10
##  Healthy  39      88

## What proportion is correctly classified (overall accuracy)?
mean(predLin==trueClass) 
## 69% (approx.)

## What proportion is misclassified (overall error rate)?
mean(predLin!=trueClass) 
## 31% (approx.)

## Calculate by hand 
####################

## Accuracy:
(21+88)/158


## Error rate:
(10+39)/158

## Sensitivity:
21/60 #0.35

## Specificity:
88/98 # 0.898

## Prediction performance using the confusionMatrix() function from caret package 
confusionMatrix(predLin,trueClass)

## Train a non-linear (RBF kernel) SVM with default parameters
##############################################################

## SVM with RBF kernel (default parameters) 
svmRBFd<-svm(class ~ ., data=traindata, kernel="radial")
summary(svmRBFd)
print(svmRBFd$gamma)

##test set predictions
predRBFd <-predict(svmRBFd,testdata)

## See misclassification
table(trueClass, predRBFd)

##         predRBFd
##trueClass CHD Healthy
##  CHD      20      41
##  Healthy   8      76

## Prediction performance (accuracy, sensitivity, specificity etc.)
confusionMatrix(predRBFd,trueClass)

## Tune SVM (RBF) to find the best parameters 
set.seed(123)
tuneSVM<-tune.svm(class~ ., data=traindata, kernel="radial",cost=2^(-2:4), gamma=2^(-3:5))
print(tuneSVM)

## Now fit SVM with the optimised parameters
svmRBFopt<-svm(class ~ ., data=traindata, kernel="radial",cost=1,gamma=0.125)
summary(svmRBFopt)

##test set predictions
predRBFopt <-predict(svmRBFopt,testdata)

## See misclassification
table(trueClass, predRBFopt)

##         predRBFopt
##trueClass CHD Healthy
##  CHD      20      41
##  Healthy   7      77

## Prediction performance (accuracy, sensitivity, specificity etc.)
confusionMatrix(predRBFopt,trueClass)

## End of code ########################
