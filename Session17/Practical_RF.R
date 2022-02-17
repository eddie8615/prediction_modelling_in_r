
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&#
#                                                                                                                         #
#                                 Biostatistics and Health Informatics Department - IoPPN                                 #
#                                                                                                                         #
#              Decision Trees and Random Forests for Classification, Regression and Variable Selection                    #
#                                                                                                                         #
#                                        Prediction modelling Module - Practical 2021                                     #
#                                                                                                                         #
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&#

# Datasets ####

# This practical is based on applying trees and random forests for the analysis of two problems (and datasets):

# 1) CLASSIFICATION: The stage C prostate cancer dataset will be used as an example of classification. 
#    The main clinical endpoint of interest is whether the disease recurs after initial 
#    surgical removal of the prostate. The endpoint (outcome) in this example is "pgstat", which takes on the value 1 if 
#    the disease has progressed and 0 if not.    
#    There are 7 predictor variables available. A short description of each of the variables can be inspected typing ?stagec 

#install.packages("rpart")
library(rpart)
?stagec 
View(stagec)

# 2) REGRESSION: The Boston dataset will be used as an example of regression. 
#    It contains data on housing values in suburbs of Boston.
#    The continuous outcome variable of interest is "medv" - median value of owner-occupied homes in $1000s.
#    There are 13 predictors. A short description of each variable can be inspected typing:

#install.packages("MASS")
library(MASS)
?Boston
View(Boston)

# We will use the package {tree} to grow trees:
# install.packages("tree")

# We will use the package {randomForest} tp grow Random Forests:
# install.packages("randomForest")

# We will use the package {caret} to derive performance measures and variable selection:
# install.packages("caret")
# install.packages("e1071")
# install.packages("dplyr")

# We will use the package {pROC} to derive performance measures and variable selection:
# install.packages("pROC")

# Load the packages

library(tree)
library(randomForest)
library(caret)
library(pROC)

?tree
?randomForest
?train #from caret 
?pROC

# The exercises will follow next structure:

# STEP 0: Load packages and data management.
# STEP 1: Define training and test datasets for internal validation 
# STEP 2: Grow a model (either regression tree or random forest)
# STEP 3: Interpret the model (either regression tree or random forest)
# STEP 4: Test the predictive ability of the model on test data
# STEP 5 (optional): Apply a modification to the model and see if it performs better

# Update the Random number generator defaut parameter so that you can replicate the solutions given for this practical.
RNGkind(sample.kind = "Rounding")


# BLOCK I: TREES ####
# Regression trees ####

# STEP 0: Load packages and data management.
############################################

# To grow regression trees, load the Boston data and the {tree} package

library(MASS)
data(Boston)
library(tree)

# Q1: Identify the dimension of the data. Identify the outcome variable.

dim(Boston)
?Boston
View(Boston)

# Dimension: 506 14
# Outcome variable is 'medv' which is median value of owner-occupied house in $1000 in Boston


# Q2: Are there missing values? If so, exclude them from analysis.

nnas<-apply(Boston,1,FUN=function(x){ nas<-sum(is.na(x)); nas})
sum(nnas==0)

# No missing value so that we can do complete analysis

# STEP 1: Define training and test datasets for internal validation 
###################################################################

# training n=337 observations
# test n=169 observations  approximately 1/3 

set.seed(4567)
flag<-sample(1:506,169)   # select 169 observations at random

train<-Boston[-flag,]
dim(train) # 337 14 
test<-Boston[flag,]
dim(test)  # 169  14 

# STEP 2: Grow and plot a regression tree 
#########################################

# The model is fit by using the tree function from package {tree}. The first argument of the function is a model formula, 
# with the ∼ symbol standing for “is modeled as”. The left-hand-side (response) should be either a factor, 
# when a classification tree is produced, or a numerical vector when a regression tree is fitted.
# The right-hand-side should be a series of numeric or factor variables separated by +.  

set.seed(123)
fit.tree  <- tree(medv ~., data = train)

# Plot the tree. The plot and text command plot the tree and then label the plot

pdf("Boston_tree.pdf")

plot(fit.tree)
text(fit.tree,pretty=0)

dev.off()

# STEP 3: Interpret the tree
############################

# Print the tree. 
# Q3: What variable determine the first split at the root of the tree?

print(fit.tree)

# SQ3: 

# STEP 4: Test the predictive ability of the tree 
#################################################

# Predict test set outcomes using pruned and unpruned trees

pred.tree<-predict(fit.tree,test)

myTest<-function(pred,y){
  
  # pred - predicted outcomes
  # y    - observed outcomes
  
  mse<-mean((y-pred)^2) # calculated MSE as mean of squared errors
  r2<-1-(sum((y-pred)^2)/sum((y-mean(y))^2)) # calculate 'Pseudo R2'
  out<-c(mse,r2)
  names(out)<-c("mse","r2")
  return(out)
}

# Q4: What is the classification performance of the unpruned tree?

myTest(pred.tree,test$medv)

# MSE: 26.573
# 68.94% of variance explained


# STEP 5: Modify the tree by pruning it and check if it performs better on test data
######################################################################

# We will prune the tree and check performance 

# Use function cv.tree(name,FUN=prune.tree) to check the optimal number of terminal nodes, in terms of RSS

set.seed(4567)
cv.tree <- cv.tree(fit.tree, FUN = prune.tree)
plot(cv.tree) 

# Select the size (number of terminal nodes with the minimum deviance). Around 8 terminal nodes (leaves)

# Prune the tree to 8 terminal nodes

prune.tree <- prune.tree(fit.tree, best = 8)

# Plot the pruned tree

plot(prune.tree)
text(prune.tree, pretty = 0,cex=1.0,col="darkblue")

# Predict test set outcomes and check test performance of the pruned tree

pred.prunedtree<-predict(prune.tree,test)
myTest(pred.prunedtree,test$medv)

# Q5: What is the classification performance of the pruned tree?

# MSE: 23.941
# R2: 70.71% of variance explained


# Q6: Which tree did perform better?
# No big difference in performance

# Classification trees ####

# STEP 0: Load packages and data management.
############################################

# To grow classification trees, load the stagec data and the {tree} package

library(rpart)
data(stagec)
library(tree) 

# Q7: Identify the dimension of the data. Identify the outcome variable "pgstat".

dim(stagec)
?stagec
View(stagec)

# Dimension: 146 8


# Q8: Are there missing values? If so, exclude them from analysis.

nnas<-apply(stagec,1,FUN=function(x){ nas<-sum(is.na(x)); nas})
sum(nnas==0)

# 134 missing values

# Excluding missing data:

stagec<-stagec[nnas==0,]
table(stagec$pgstat) 

#  0  1 
# 85 49  

#define a factor for the outcome

progstat <- factor(stagec$pgstat, levels = 0:1, labels = c("No", "Prog"))
stagec<-data.frame(stagec,progstat)
rm(progstat)

levels(stagec$progstat) # "No" "Prog" 
table(stagec$progstat) 

# No Prog 
# 85  49 

# STEP 1: Define training and test datasets for internal validation 
###################################################################

# training n=89 observations
# test n=45 observations  approximately 1/3 (29 from class "0=No", and 16 from class "1=Prog")

set.seed(4567)
stagec<-stagec[order(stagec$pgstat),]
flag<-c(sample(1:85,29),sample(86:134,16))   # select 89 observations at random

train<-stagec[-flag,-c(2)]
dim(train) # 89  8
test<-stagec[flag,-c(2)]
dim(test)  # 45  8 

# STEP 2: Grow and plot a classification tree 
#############################################

# The model is fit by using the tree function from package {tree}. The first argument of the function is a model formula, 
# with the ∼ symbol standing for “is modeled as”. The left-hand-side (response) should be either a factor, 
# when a classification tree is produced, or a numerical vector when a regression tree will be fitted later on in this practical. 
# The right-hand-side should be a series of numeric or factor variables separated by +.  

set.seed(4567)
fit.tree  <- tree(progstat ~  age + eet + g2 + grade + gleason + ploidy, data = train)

# Plot the tree. The plot and text command plot the tree and then label the plot

pdf("stagec_tree.pdf")

plot(fit.tree)
text(fit.tree,pretty=0)

dev.off()

# STEP 3: Interpret the tree
############################

# Print the tree

fit.tree 

# Q9: What variable does determine the first split at the root of the tree?

# SQ9:


# STEP 4: Check the classification error in the test set
########################################################

# Generate test class predictions

pred.te<-predict(fit.tree,test,type="class")

# Check the test set confusion matrix: Confusion matrix from caret

library(caret)
library(e1071)
confusionMatrix(pred.te,test$progstat)

# Compute an alternative measure of performance: AUC from pRoc package

library(pROC)

p<-predict(fit.tree,test)
auc.val<-auc(test$progstat,p[,1])
auc.val

# Accuracy: 64.44%
# Sensitivty: 0.6552
# Specificity: 0.6250
# AUC: 0.6142

# the model seems to be underfitted


# BLOCK II: RANDOM FOREST ####
# Random Forest - Regression ####

# STEP 0: Load packages 
#######################

library(MASS)
data(Boston)

#STEP 1: Define training and test datasets for internal validation 
##################################################################

set.seed(4567)
flag<-sample(1:506,169)   # select 169 observations at random

train<-Boston[-flag,]
dim(train) # 337 14 
test<-Boston[flag,]
dim(test)  # 169  14 

# STEP 2: grow a random Forest to predict medv outcome using all the available predictors. 
##########################################################################################

# The function is randomForest, and works similary to the previously used functions.

library(randomForest)
set.seed(4567)
fit.rf<-randomForest(medv~.,train)
fit.rf

# Q11: Report in Mean Squared Residuals, and percentage variance explained.
# MSE: 12.07
# R2: 85.93%

# Q12: How many variables were used on each split?
# 4 variables are used on each split

# STEP 3: Interpret the model. This step will come in the next session on variables selection.
# STEP 4: Check the classification error in the test set. 
#########################################################

pred.rf<-predict(fit.rf,test)

myTest<-function(pred,y){
  
  # pred - predicted outcomes
  # y    - observed outcomes
  
  mse<-mean((y-pred)^2) # calculated MSE as mean of squared errors
  r2<-1-(sum((y-pred)^2)/sum((y-mean(y))^2)) # calculate 'Pseudo R2'
  out<-c(mse,r2)
  names(out)<-c("mse","r2")
  return(out)
}

myTest(pred.rf,test$medv)

# Q13: Report on MSE and R2 for the model.
# MSE: 11.429
# R2: 86.02%

# There is no huge difference between apparent validation and internal validation
# In terms of R2, only 0.1% difference between the validation scores as well as MSE

 
# Plot the performance measures MSE and "pseudo R2" by number of trees using model$mse and model$rsq

pdf("Boston_rf_perform.pdf")

# MSE
plot(fit.rf$mse,xlab="Number of trees",ylab="MSE")
# R2
plot(fit.rf$rsq,xlab="Number of trees",ylab="RSQ")

dev.off()

# After approximately 200 trees the prediction performance is stable, it reaches a plateau.

# Random Forest - Classification ####

# STEP 0: Load packages and data management
###########################################

library(rpart)
data(stagec)
library(randomForest)

# Excluding missing data:

nnas<-apply(stagec,1,FUN=function(x){ nas<-sum(is.na(x)); nas})
stagec<-stagec[nnas==0,]

#define a factor for the outcome

progstat <- factor(stagec$pgstat, levels = 0:1, labels = c("No", "Prog"))
stagec<-data.frame(stagec,progstat)
rm(progstat)

levels(stagec$progstat) # "No" "Prog" 
table(stagec$progstat) 

# No Prog 
# 85  49 

# STEP 1: Define training and test datasets for internal validation 
###################################################################

# training n=89 observations
# test n=45 observations  approximately 1/3 (29 from class "0=No", and 16 from class "1=Prog")

set.seed(4567)
stagec<-stagec[order(stagec$pgstat),]
flag<-c(sample(1:85,29),sample(86:134,16))   # select 89 observations at random

train<-stagec[-flag,]
dim(train) # 89  9
test<-stagec[flag,]
dim(test)  # 45  9 


# STEP 2: grow a random Forest. 
###############################

# The function is randomForest, and works similary to the previously used functions 

set.seed(4567)
rf.class<-randomForest(progstat~.,train[,-2],importance=T)

# Print the forest

rf.class

# How many trees were used? how many variables were used in each split? How can we justify these values?
# 500 trees in the model
# 2 variables were used in each split


# STEP 4: Check the predictive performance in test data 
#######################################################

pred.rf<-predict(rf.class,test)

# Check the test set confusion matrix: Confusion matrix from caret

library(caret)
library(e1071)
confusionMatrix(pred.rf,test$progstat)

# Accuracy: 0.7778
# Sensitivity: 0.931
# Specificity: 0.5

# BLOCK III: VARIABLE SELECTION ####

#STEP 0: load packages and data cleaning
########################################

set.seed(4567)
library(MASS)
data(Boston)

#STEP 1: Define training and test datasets for internal validation
##################################################################

set.seed(4567)
flag<-sample(1:506,169)   # select 169 observations at random

train<-Boston[-flag,]
dim(train) # 337 14 
test<-Boston[flag,]
dim(test)  # 169  14 


# STEP 2: Grow a forest with 500 trees and order the predictors by raw VIM (permutation variable importance and impurity decrease) 
#################################################################################################################################

library(randomForest)

set.seed(4567)
fit.rf<-randomForest(medv~.,train,importance=T,ntree=500)

# STEP 3: Interpret and plot variables importance
#################################################

# the object "fit.rf$importance" is a matrix where first column contains the decrease in MSE 
# and second column contains the decrease in node impurity. 

imp<-fit.rf$importance
varImpPlot(fit.rf, main='Plots of variable importance', scale=F) 

# STEP 4: Modify the random forests. Grow random forests with the top 5, 9 and 13 variables, and assess
# predictive ability. 

# Order the variables by decreasing VIM (decrease in MSE, first column in object "imp") 

top<-imp[order(imp[,1],decreasing=T),] 
top<-rownames(top) 
top
# top variables?
# lstat > rm > nox > crim > indus > dis > ptratio > tax > age > black > rad > zn > chas

# run the models

set.seed(4567)
fit.rf5<-randomForest(medv~.,data=train[,colnames(train)%in%c(top[1:5],"medv")],importance=T,ntree=500)

set.seed(4567)
fit.rf9<-randomForest(medv~.,data=train[,colnames(train)%in%c(top[1:9],"medv")],importance=T,ntree=500)

set.seed(4567)
fit.rf13<-randomForest(medv~.,data=train[,colnames(train)%in%c(top[1:13],"medv")],importance=T,ntree=500)

# Test set predictions for each model

myTest<-function(pred,y){
  
  # pred - predicted outcomes
  # y    - observed outcomes
  
  mse<-mean((y-pred)^2) # calculated MSE as mean of squared errors
  r2<-1-(sum((y-pred)^2)/sum((y-mean(y))^2)) # calculate 'Pseudo R2' (% Variation Explained)
  out<-c(mse,r2)
  names(out)<-c("mse","r2")
  return(out)
}

# Q14: Report performance measures for each model

myTest(predict(fit.rf5,test),test$medv)

# MSE: 15.103
# R2: 81.53%


myTest(predict(fit.rf9,test),test$medv)

# SQ14: For the model including top 9 variables:
# MSE: 10.914
# R2: 86.65%


myTest(predict(fit.rf13,test),test$medv)

# SQ14: For the model including top 13 variables:
# MSE: 11.011
# R2: 86.53%

# The model including 5 variables shows the worst performance among them
# Interestingly, the model including 9 variables outperforms the one including 13 variables
# However, R2 score for 5 var model is less of 5% than other two models.
# If this difference does not provide outstanding impact, and if the models are much more complex,
# the 5 var model would be preferred.
# The unimportant variables will be trimmed out through dimension reduction process such as PCA or random projection.


# REMEMBER! set again the default parameter in RStudio for your Random Number Generator, as below:

RNGkind(sample.kind = "Rejection")







