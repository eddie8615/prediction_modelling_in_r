###### Daniel Stahl: Introduction to Prediction modelling 2021 #########
###### Session 11: Regularized Regression III ##########
####   Logistic regularized regression - Practical 1b  ####

###################################################################################################
######################  NESTED CROSS VALIDATION  ##################################################
###################################################################################################
######################  LASSO LOGISTIC REGRESSION   ###############################################

#install.packages("Rcpp")
library(Rcpp)
#install.packages("caret")
library(caret)
#install.packages("MatrixModels")
library(MatrixModels)
#install.packages("doParallel")
#install.packages("glmnet")
library(glmnet)
#install.packages("rlang")
#install.packages("ggplot2")
#library(ggplot2)
library(doParallel) #for parallelizing
#install.packages("e1071")# for lasso dealing with named levels of outcome
library(e1071)
#install.packages("pROC")#to compute the AUC
library(pROC)
#library(dplyr)# for stratified bootstrap functions group_by and sample_n (oversampling)
#install.packages("gmodels")
library(gmodels)# for CrossTable


options("scipen"=100, "digits"=4)

######################################################################################################################
# The data needs to have categorical and binary variables as factors, and continuous and ordinal variables as numeric#
######################################################################################################################

# OUTCOME: is the last variable ##############################
# check levels of outcome: the lowest level should be "no event"
# levels(outcome) to check that the first vector component here is the lowest level
# x and y need to be in one object (unlike glmnet)
# y needs to be defined as a factor
# the data set needs to be a dataframe (unlike glmnet)


# Importing data file
autism <- read.csv("autism_summerschool.csv")
levels(autism$group) <- c("lowRisk", "highRisk")  # levels 0 and 1 aren't valid names in R and you need to label your two levels


# Group 1 = high risk group Group 0 = control group
summary(autism)
# Import the data
# Descriptive statistics for each variable by group
# Group 1 = high risk group Group 0 = control group
describeBy(autism,autism$group)

######################################################################################################################
# The data needs to have categorical and binary variables as factors, and continuous and ordinal variables as numeric#
######################################################################################################################
# Important: glmnet needs the predictor and outcome variables in seperate data files
# There is no need for dummy coding for predictors
# This is done automatically within caret and this script 
# The script needs y in the last column
# autism[ ,-1] means that all columns except the first one (group) will be  copied into x
# We need to use the combined data set with x and y: autism
# The outcome needs to be defined as a factor
# Outcome: The lowest level should be "no event", makes interpretation easier

data<-autism[-1]
data$group <-as.factor(autism$group)
levels(data$group) <- c("lowRisk", "highRisk")  # levels 0 and 1 aren't valid names in R and you need to label your two levels
# The function needs data to be in a dataframe
data<-as.data.frame(data)
# Check
levels(data$group)
#  [1] "low risk"  "high risk"  <- "low risk" is lowest level as required
summary(data)

# To check:
# OUTCOME: is the last variable
# Check levels of outcome: the lowest level should be "no event", makes interpretation easier
# Use levels(outcome) to check that the first vector component here is the lowest level
# x and y need to be in one object (unlike glmnet)
# y needs to be defined as a factor
# categorical variables need to be defined as factors (unless dummy-coded)
# the data set needs to be a dataframe (unlike glmnet)




#GRID FOR TUNING PARAMETERS
# Grid=expand.grid(alpha = 1 ,lambda= 10^seq(-0.5,-3,length=40))  # 40 lambdas  (faster but less accurate)
Grid=expand.grid(alpha = 1 ,lambda= 10^seq(0.5,-3,length=100))   # 100 lambdas
# This usually works well


#FUNCTIONS NEEDED:__________________________________________________________________________________________________________________________________

### A function to compute optimism ################################
testPerformance<- function (coef,data.train,data.test) {
  
  Outcome.train  <- unlist(data.train[,dim(data.train)[2]])
  levels(Outcome.train) <- c("noEvent","Event")
  Outcome.test  <- unlist(data.test[,dim(data.test)[2]])
  levels(Outcome.test) <- c("noEvent","Event")
  modelMatrix.train <- model.Matrix(as.formula(paste(colnames(data.train)[dim(data.train)[2]],"~.",sep="")),data.train)  
  X.train<-modelMatrix.train[,-1]   
  modelMatrix.test <- model.Matrix(as.formula(paste(colnames(data.test)[dim(data.test)[2]],"~.",sep="")),data.test)  
  X.test<-modelMatrix.test[,-1]   
  Predicted.train  <-  as.matrix(exp(coef[1] + X.train%*%coef[-1])/(exp(coef[1] + X.train%*%coef[-1])+1))
  Predicted.train[Predicted.train==1]  <- 0.999999999
  Predicted.train[Predicted.train==0]  <- 0.000000001
  Predicted.test  <-  as.matrix(exp(coef[1] + X.test%*%coef[-1])/(exp(coef[1] + X.test%*%coef[-1])+1))
  Predicted.test[Predicted.test==1]  <- 0.999999999
  Predicted.test[Predicted.test==0]  <- 0.000000001
  

  roc_obj.train <- roc(Outcome.train, as.vector(Predicted.train),quiet=T)
  bestThreshold.train<-pROC::coords(roc_obj.train, "best", ret = "threshold",transpose=T)
  PredictedClass.test<-factor(Predicted.test>0.5); levels(PredictedClass.test)<-c("noEvent","Event")
  confMatrix50threshold.test<-confusionMatrix(PredictedClass.test, Outcome.test, positive="Event")
  sensitivity.test<-confMatrix50threshold.test$byClass["Sensitivity"]
  specificity.test<-confMatrix50threshold.test$byClass["Specificity"]
  PPV.test<-confMatrix50threshold.test$byClass["Pos Pred Value"]
  NPV.test<-confMatrix50threshold.test$byClass["Neg Pred Value"]
  Accuracy.test<-confMatrix50threshold.test$overall["Accuracy"]
  Kappa.test<-confMatrix50threshold.test$overall["Kappa"]

  if(bestThreshold.train[1]==-Inf ){
    sensitivityBest.test<-NA
    specificityBest.test<-NA
    PPVBest.test<-NA
    NPVBest.test<-NA
    AccuracyBest.test<-NA
    KappaBest.test<-NA
    
  } else {
    PredictedClassBest.test<-factor(Predicted.test>as.numeric(bestThreshold.train[1])); levels(PredictedClassBest.test)<-c("noEvent","Event")
#    PredictedClassBest.test<-factor(Predicted.test>bestThreshold.train[1]); levels(PredictedClassBest.test)<-c("noEvent","Event")
    confMatrixBestthreshold.test<-confusionMatrix(PredictedClassBest.test, Outcome.test, positive="Event")
    sensitivityBest.test<-confMatrixBestthreshold.test$byClass["Sensitivity"]
    specificityBest.test<-confMatrixBestthreshold.test$byClass["Specificity"]
    PPVBest.test<-confMatrixBestthreshold.test$byClass["Pos Pred Value"]
    NPVBest.test<-confMatrixBestthreshold.test$byClass["Neg Pred Value"]
    AccuracyBest.test<-confMatrixBestthreshold.test$overall["Accuracy"]
    KappaBest.test<-confMatrixBestthreshold.test$overall["Kappa"]
  }
  roc_obj.test <- roc(Outcome.test, as.vector(Predicted.test),quiet = T)
  AUC.test   <-as.numeric(roc_obj.test$auc)
  logOdds.test<-log(Predicted.test/(1-Predicted.test))
  glm.coef.test.beta       <-  glm(Outcome.test ~ logOdds.test,family=binomial)$coef
  Beta.test	 <-  glm.coef.test.beta[2]
  glm.coef.test.alpha       <-  glm(Outcome.test ~ offset(logOdds.test),family=binomial)$coef
  Alpha.test  <-  glm.coef.test.alpha[1]
  
  out.best.tol  <- list(AUC.test,Alpha.test,Beta.test,
                        sensitivityBest.test,specificityBest.test,
                        PPVBest.test,NPVBest.test,AccuracyBest.test,
                        KappaBest.test,sensitivity.test,
                        specificity.test,PPV.test,NPV.test,
                        Accuracy.test,Kappa.test)
  out.best.tol
}


#________________________________________________________________________________________________________________________________________________________

# MAIN FUNCTION ###########################################################################################

NestedCVvalidation<- function (outerFolds,outerRepeats,innerFolds,innerRepeats,Data,Grid,seed){
  
  set.seed(seed)
  seeds<-sample(1:10000000,outerFolds*outerRepeats)
  set.seed(seed)
  Cv.row <- createMultiFolds(Data[,1],outerFolds,outerRepeats) #should create the 1000 folds for the 100 times repeated 10-cv
  # inspect Cv.row as each repetition doesn't necessarily have 10 folds because of stratified sampling ####
  if(outerRepeats<10){
    numberFolds<-unlist(lapply(sprintf(".Rep%01d", 1:outerRepeats),function(x) length(grep(x,names(Cv.row)))))
  } else if (outerRepeats<100){
    numberFolds<-unlist(lapply(sprintf(".Rep%02d", 1:outerRepeats),function(x) length(grep(x,names(Cv.row)))))
  } else {
    numberFolds<-unlist(lapply(sprintf(".Rep%03d", 1:outerRepeats),function(x) length(grep(x,names(Cv.row)))))
  }
  N<-sum(numberFolds)
  
  All.sensitivityBest.best   <-All.sensitivityBest.tol1SE <-rep(NA,N)
  All.specificityBest.best   <-All.specificityBest.tol1SE <-rep(NA,N)
  All.PPVBest.best   <-All.PPVBest.tol1SE <-rep(NA,N)
  All.NPVBest.best   <-All.NPVBest.tol1SE <-rep(NA,N)
  All.AccuracyBest.best   <-All.AccuracyBest.tol1SE <-rep(NA,N)
  All.KappaBest.best   <-All.KappaBest.tol1SE <-rep(NA,N)
  All.sensitivity.best   <-All.sensitivity.tol1SE <-rep(NA,N)
  All.specificity.best   <-All.specificity.tol1SE <-rep(NA,N)
  All.PPV.best   <-All.PPV.tol1SE <-rep(NA,N)
  All.NPV.best   <-All.NPV.tol1SE <-rep(NA,N)
  All.Accuracy.best   <-All.Accuracy.tol1SE <-rep(NA,N)
  All.Kappa.best   <-All.Kappa.tol1SE <-rep(NA,N)
  All.AUC.test.best   <- All.AUC.test.tol1SE   <-   rep(NA,N)
  All.Alpha.test.best  <- All.Alpha.test.tol1SE   <-    rep(NA,N)
  All.Beta.test.best  <- All.Beta.test.tol1SE   <-  rep(NA,N)
  
  #Matrix for best performances in tuning
  TuningROC<-matrix(NA,innerFolds*innerRepeats,N)
  
  #Matrix for tuning parameters
  Lambdas<-matrix(NA,2,N,dimnames=list(c("Best","1SE"),NULL))
  
  f=1 #indexing the fold in numberFolds
  while(f<=N){
    print(f)
    data.train<-Data[Cv.row[[f]],]
    data.test<-Data[-Cv.row[[f]],]
    
    # Defining outcome and predictors 
    Outcome.train<-data.train[,length(names(data.train))] #outcome is the last variable now
    modelMatrix.data.train <- model.Matrix(as.formula(paste(colnames(data.train)[length(names(data.train))],"~.",sep="")),data.train)  
    # X.data.train<-modelMatrix.data.train[,-1]  #predictors only (no intercept)
    Outcome.test<-data.test[,length(names(data.test))] #outcome is the last variable now
    modelMatrix.data.test <- model.Matrix(as.formula(paste(colnames(data.test)[length(names(data.test))],"~.",sep="")),data.test)  
    # X.data.test<-modelMatrix.data.test[,-1]  #predictors only (no intercept)
    
    levels(Outcome.train) <- c("noEvent","Event")
    levels(Outcome.test) <- c("noEvent","Event")
    
    options(warn=-1)
    #cl=makeCluster(cores_2_use);registerDoParallel(cl)
    set.seed(seeds[f])
    cvIndex <- createMultiFolds(Outcome.train, k=innerFolds, times=innerRepeats)# for repeated startified CV
    Fit.Caret <- train(x=modelMatrix.data.train[,-1] ,y=Outcome.train, method="glmnet",tuneGrid=Grid, family="binomial",            
                       trControl=trainControl(method="repeatedcv", number=innerFolds,  repeats=innerRepeats, 
                                              index = cvIndex,
                                              selectionFunction="best",
                                              summaryFunction = twoClassSummary, #optimizing with AUC instead of accuracy
                                              classProbs = TRUE )) 
    
    #stopCluster(cl)
    options(warn=0)
    
    #- Best performances in tuning -#
    TuningROC[,f]<-Fit.Caret$resample$ROC
    
    #- Tuning parameters
    Lambdas[,f]<-c(Fit.Caret$bestTune$lambda,max(Fit.Caret$results$lambda[max(Fit.Caret$results$ROC)-Fit.Caret$results$ROC <=
                                                                            (Fit.Caret$results[row.names(Fit.Caret$bestTune),]$ROCSD)/sqrt(innerFolds*innerRepeats)]))
    
    #- Model coefficients -#
    
    coef.best <- as.matrix(coef(Fit.Caret$finalModel,s=Fit.Caret$bestTune$lambda))
    coef.tol1SE <- as.matrix(coef(Fit.Caret$finalModel,s=max(Fit.Caret$results$lambda[max(Fit.Caret$results$ROC)-Fit.Caret$results$ROC <=
                                                                                        (Fit.Caret$results[row.names(Fit.Caret$bestTune),]$ROCSD)/sqrt(innerFolds*innerRepeats)])))
    
    
    #-- Calculate optimism and calibration slope (beta)-- #
    
    Model.best = testPerformance(coef=coef.best,data.train=data.train,data.test=data.test)
    Model.tol.1SE = testPerformance(coef=coef.tol1SE,data.train=data.train,data.test=data.test)
    
    
    All.AUC.test.best[f]    <- Model.best[[1]] 
    All.Alpha.test.best[f]       <- Model.best[[2]]
    All.Beta.test.best[f]        <- Model.best[[3]]
    All.AUC.test.tol1SE[f]       <- Model.tol.1SE[[1]] 
    All.Alpha.test.tol1SE[f]          <- Model.tol.1SE[[2]]
    All.Beta.test.tol1SE[f]   	       <- Model.tol.1SE[[3]]
    
    All.sensitivityBest.best[f]<- Model.best[[4]] 
    All.sensitivityBest.tol1SE[f]<- Model.tol.1SE[[4]]
    
    All.specificityBest.best[f]  <- Model.best[[5]] 
    All.specificityBest.tol1SE[f] <-Model.tol.1SE[[5]]
    
    All.PPVBest.best[f]   <- Model.best[[6]] 
    All.PPVBest.tol1SE[f]<-Model.tol.1SE[[6]]
    
    All.NPVBest.best[f]  <- Model.best[[7]] 
    All.NPVBest.tol1SE[f] <-Model.tol.1SE[[7]]
    
    All.AccuracyBest.best[f]   <- Model.best[[8]] 
    All.AccuracyBest.tol1SE[f]<-Model.tol.1SE[[8]]
    
    All.KappaBest.best[f]  <- Model.best[[9]] 
    All.KappaBest.tol1SE[f] <-Model.tol.1SE[[9]]
    
    All.sensitivity.best[f]  <- Model.best[[10]] 
    All.sensitivity.tol1SE[f] <-Model.tol.1SE[[10]]
    
    All.specificity.best[f] <- Model.best[[11]] 
    All.specificity.tol1SE[f] <-Model.tol.1SE[[11]]
    
    All.PPV.best[f]   <- Model.best[[12]] 
    All.PPV.tol1SE[f] <-Model.tol.1SE[[12]]
    
    All.NPV.best[f]  <- Model.best[[13]] 
    All.NPV.tol1SE[f] <-Model.tol.1SE[[13]]
    
    All.Accuracy.best[f]  <- Model.best[[14]] 
    All.Accuracy.tol1SE[f] <-Model.tol.1SE[[14]]
    
    All.Kappa.best[f]   <- Model.best[[15]] 
    All.Kappa.tol1SE[f] <-Model.tol.1SE[[15]]
    
    # #Saving performance
    # 
    # if(f==1){
    #   write(t(cbind(unlist(Model.best),unlist(Model.tol.1SE))),file = "NestedCV.txt",append = FALSE, ncolumns=2)
    # } else {
    #   write(t(cbind(unlist(Model.best),unlist(Model.tol.1SE))),file = "NestedCV.txt",append = TRUE, ncolumns=2)
    # }
    
    f<-f+1
  } 
  ## Average test performances
  testPerformances<-matrix(c(mean(All.AUC.test.best, na.rm = T),mean(All.AUC.test.tol1SE, na.rm = T),
                             mean(All.sensitivityBest.best,na.rm=T),mean(All.sensitivityBest.tol1SE,na.rm=T),
                             mean(All.specificityBest.best,na.rm=T),mean(All.specificityBest.tol1SE,na.rm=T) ,
                             mean(All.PPVBest.best,na.rm=T),mean(All.PPVBest.tol1SE,na.rm=T),
                             mean(All.NPVBest.best,na.rm=T),mean(All.NPVBest.tol1SE,na.rm=T),
                             mean(All.AccuracyBest.best,na.rm=T),mean(All.AccuracyBest.tol1SE,na.rm=T),
                             mean(All.KappaBest.best,na.rm=T),mean(All.KappaBest.tol1SE,na.rm=T),
                             mean(All.sensitivity.best,na.rm=T),mean(All.sensitivity.tol1SE,na.rm=T),
                             mean(All.specificity.best,na.rm=T),mean(All.specificity.tol1SE,na.rm=T),
                             mean(All.PPV.best,na.rm=T),mean(All.PPV.tol1SE,na.rm=T),
                             mean(All.NPV.best,na.rm=T),mean(All.NPV.tol1SE,na.rm=T) ,
                             mean(All.Accuracy.best,na.rm=T),mean(All.AccuracyBest.tol1SE,na.rm=T),
                             mean(All.Kappa.best,na.rm=T),mean(All.Kappa.tol1SE,na.rm=T),
                             mean(All.Beta.test.best,na.rm=T),mean(All.Beta.test.tol1SE,na.rm=T),
                             mean(All.Alpha.test.best,na.rm=T),mean(All.Alpha.test.tol1SE,na.rm=T)),15,2,byrow=TRUE)
  
  colnames(testPerformances)<-c("Av.best","Av.tol.1SE")
  
  row.names(testPerformances)<-c("Av.AUC","Av.sensBestThresh","Av.specBestThresh",
                                 "Av.PPVBestThresh","Av.NPVBestThresh",
                                 "Av.AccuracyBestThresh","Av.KappaBestThresh",
                                 "Av.sens50Thresh","Av.spec50Thresh",
                                 "Av.PPV50Thresh","Av.NPV50Thresh",
                                 "Av.Accuracy50Thresh","Av.Kappa50Thresh",
                                 "Av.Beta","Av.Alpha")
  
  out.temp <- list(testPerformances=testPerformances,TuningROC=TuningROC,Lambdas=Lambdas)
  
  out.temp
  
}
#___________________________________________________________________________________________________________________________________________
###########################################################################################################################################
### The analyses starts here
# Set up number of cores 


cl=makeCluster(4);registerDoParallel(cl)

(NestedCV_results<-NestedCVvalidation(outerFolds=5,outerRepeats=2,innerFolds=5,innerRepeats=5,Data=data,Grid=Grid,seed=779))

stopCluster(cl) # Stop the use of cores!

print(NestedCV_results)

### Obtain only summary performance measures
paste("The nested CV performance measures are: ")
print(NestedCV_results$testPerformance)


#> print(NestedCV_results$testPerformance)
#                       Av.best Av.tol.1SE
# Av.AUC                0.62292    0.68819
# Av.sensBestThresh     0.48667    0.34833
# Av.specBestThresh     0.53167    0.71000
# Av.PPVBestThresh      0.59167    0.59375
# Av.NPVBestThresh      0.45500    0.49167
# Av.AccuracyBestThresh 0.54762    0.51667
# Av.KappaBestThresh    0.01358    0.05753
# Av.sens50Thresh       0.56167    0.63333
# Av.spec50Thresh       0.57000    0.53167
# Av.PPV50Thresh        0.58667    0.65786
# Av.NPV50Thresh        0.52500    0.52976
# Av.Accuracy50Thresh   0.55119    0.51667
# Av.Kappa50Thresh      0.10244    0.13553
# Av.Beta               0.86907    7.86766
# Av.Alpha              0.21881    0.12416

# THe auc of the minimum lambda model is 0.623 (1Se=0.68). Sensitivity and Specifity and the 50%
# cut-off are 56% and 57%, resp and the accuracy is 55%.
# The calibration slobpe beta is 0.87 and calibration in the large = 0.22. 
