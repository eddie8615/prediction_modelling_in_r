###### Daniel Stahl: Introduction to Prediction modelling 2021 #########
###### Session 11: Regularized Regression III ############
###### Logistic regularized regression - Practical 2  ####


################################################################################
### Data set from Applied Predictive Modelling (2013) by Kuhn and Johnson.
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
 
##### Packages needed for regularized logistic regression 
library(glmnet)
library(psych)
library(caret)
library(pROC)
library(dplyr) 


# Load the library Applied Prediction modelling with data set
# install.packages("AppliedPredictiveModeling")
library(AppliedPredictiveModeling)
# There is one file "predictors" with 130 potential predictors of AD diagnosis
data(AlzheimerDisease)
predictors
summary(predictors)
ncol(predictors)
# and one data frame called diagnosis with the diagnosis of AD (Impaired=1) and not impaired "Control"=2
summary(diagnosis)

# The factor variable Genotype has got 6 alleles: E2E2, E2E3, E2E4,E3E2, E3E2, E3E3.
# They need to be dummy coded for glmnet
# Model matrix can be used to create dummy coded variables
  predictors2<-model.matrix( ~ .-1, data=predictors)  # this formula creates 6 dummy coded variables for genotype

  
# Create an x and y file
  x<-predictors
# The outcome needs to be defined as a factor

  # Create an x and y file
  x<-predictors2
  y<-as.numeric(diagnosis)
  y<-as.factor(y)
  y <- as.numeric(y) - 1 # to see how the categories are coded (1 and 2)
  #  1= Ad and 2 = control,
  # Comment: Glmnet does not like labels like AD and control for predictions

  # Check that all is correct:
  summary(x)
  summary(y)
  
  
# Logistic regularized regression 
# Note alpha =1 for lasso only, if you want ridge regression set alpha to 0
  glmmod<-glmnet(x,y=y,alpha=1, family='binomial')
  plot(glmmod, xvar="lambda")

# Select best lambda by cross validation using cv.glmnet
  set.seed(777)
  cv.glmmod<- cv.glmnet(x,y,alpha=1,family='binomial', type.measure = "dev" )
  plot(cv.glmmod)
  
# Store best lambdas (minimum and minimum + 1SE)
  lmin <- cv.glmmod$lambda.min
  l1se <- cv.glmmod$lambda.1se
  
  # Print coefficients for the best minimum lambda
  coef.min<-coef(cv.glmmod, s="lambda.min")
  print(coef.min)
  # How many variables were selected? (Number of variabkles not being set to 0)
  colSums(coef.min != 0)
  # 41 variables were selected
  
  # Minimum lambda + 1SE model
  coef.min.1se<-coef(cv.glmmod, s="lambda.1se")
  print(coef.min.1se)
  # How many variables were selected? (Number of variabkles not being set to 0)
  colSums(coef.min.1se != 0)
  # 19 variables were selected
  
  
# Apparent validity
## Get prediction accuracy measures based on observed data set (overoptimistic!)
y_prob<-predict(cv.glmmod, s="lambda.min", type="response", newx = x)
y_pred<-as.numeric(predict(cv.glmmod,type="class",newx=x))


# Get discrimination estimates
# Positive = 2 means that factor level=2 is clinical outcome
confusionMatrix(as.factor(y_pred), as.factor(y), positive = "1" )

# Get the AUC value
roc_obj.train <- roc(y, (y_prob))  # Important correction from y_pred to y_prob!  22/2/21. this 
roc_obj.train

# Calibration plot on  apparent data
# We need our y data as 0's (Control)  and 1's (cases) and not 1's (control) and 2's (cases)
 y_obs<-recode(y,'1'=0, '2'=1)
 y_obs<-as.factor(y_obs)# define it as factor
# We also could have used:  y_obs<-as.factor(recode(y,'1'=0, '2'=1))
cali<-calibration(y_obs ~ y_prob, cuts=5, class=1 )
xyplot(cali)
ggplot(cali)
# We predict too low for small predictions and slightly too high for large ones

# 41 biomarkers were selected using the minimum lambda method. 
# The AUC value is 0.83. Apparent accuracy is 91% and apparent specificity is excellent (98%)
# Sensitivity is lower (73%).The minimum lambda criteria is rather unstable, if we repeat 
# the analyses we get different results. We should run a loop with at least 
# 10 repeated cross-validations to obtain a more stable estimate (better: 50-100 repeats)


######### Repeated cross-validation ####### 

#  Repeating n-fold cross-validation ###### 
#  Selecting lambda based on a single run of 10-fold cross-validation is usually
#  not recommended. The procedure should be repeated 100 times (with different folds)
#  and the mean of each 100 minimum lambdas (or 100 minimum +1 SE lambdas) should be used 
#  This can be easily done with a loop or we use the caret package and its function "trainControl" and train ####
#  The package allows allows to use parallel computing: Large portions of code can run concurrently in different cores 
#  and reduces the total time for the computation. To use parallel computing we need to load the package "doParalell" 


library(caret)
library(glmnet)
library(doParallel)

# Create an x and y file
x<-predictors


# Comment: There is no need to dummy code categorical variables using caret!
# Merge x and y in a new dataframe object called dat
dat<-as.data.frame(cbind(y,x))
# The outcome needs to be defined as a factor
dat$y<-as.factor(y) 
levels(dat$y) <- c("AD", "Control")  # levels 1 and 2 aren't valid names in Caret and you need to label your two levels


# Set up number of cores 
cl=makeCluster(4);registerDoParallel(cl)

# Set up training settings object

trControl <- trainControl(method = "repeatedcv", # repeated CV 
                          repeats = 10,          # number of repeated CV
                          number = 10   ,         # Number of folds
                          summaryFunction = twoClassSummary,  #function to compute performance metrics across resamples.AUC for binary outcomes
                          classProbs = TRUE, 
                          savePredictions = "all",
                          allowParallel = TRUE,
                          selectionFunction = "best" ) # best - minimum lambda, oneSE for minimum lambda + 1 Se, Tolerancwe for minimum + 3%



# Set up grid of parameters to test
params = expand.grid(alpha=c(1),   # L1 & L2 mixing parameter
                     lambda=2^seq(1,-10, by=-0.1)) # regularization parameter


# Run training over tuneGrid and select best model
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
#str(glmnet.obj)


# Function which retrieves key output
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


# Model coefficients
# The fitted coefficients at the optimal penalties can be obtained by  using the  coef command 
# Best lambda is here: glmnet.obj$bestTune$lambda 
coef.min<-coef(glmnet.obj$finalModel, glmnet.obj$bestTune$lambda)  # best model 
coef.min
# How many variables were selected? (Number of variables not being set to 0)
colSums(coef.min != 0)
# 44 variables were selected

# Variable importance of Top 10 unstandardised coefficients
plot(varImp(glmnet.obj, scale = FALSE), top = 10, main = "glmnet")

# This is the final model we present!

#### Apparent validity ############
## Get prediction accuracy measures based on observed data set (overoptimistic!)
## Predict values for the training data set (apparent validity)


## Get prediction accuracy measures based on observed data set (overoptimisitc!)

predictions_prob<-(predict(glmnet.obj,s=best_lambda, alpha=best_alpha, type="prob",newx=x))
predictions_class<-(predict(glmnet.obj,s=best_lambda, alpha=best_alpha, type="raw",newx=x))

# Model prediction performance using caret functions (not Metrics)
# Get more information about your model prediction quality 
# positive defines the treatment (or positive) group, here high risk is coded as 1 
# and will be used as "positive" group 
confusionMatrix(as.factor(predictions_class), dat$y, positive="AD")

# AUC is a measure of discrimination (equivalent to the Concordance or C statistics)
# One way of interpreting AUC is as the probability that the model ranks a random positive example more highly than a random negative example.
# AD is coded as 2 and control as 1, we need to tell roc that AD is the "event"
# This is only important for the ROC curve. THe AUC doe snot change
roc_obj.train <- roc(dat$y, (predictions_prob$AD),  levels= c("Control", "AD"))
roc_obj.train

# ROC curve
plot(roc_obj.train)
##
# The AUC values of the repeated CV is 0.98!
# 44 biomarkers were selected using the minimum lambda method 
# Apparent accuracy is 94% and apparent specificity is excellent (98%)
# Sensitivity is sightly lower (85%).

# Calibration plot on  apparent data
cali<-calibration(y ~ predictions_prob$AD, cuts=5, class=1 )
xyplot(cali)
ggplot(cali)



### Calibration alpha and beta, see lecture calibration
predictions_prob[predictions_prob==1]  <- 0.999999999 # choose column with proibability to be a case = 1
predictions_prob[predictions_prob==0]  <- 0.000000001
logOdds<-log(predictions_prob/(1- predictions_prob))
glm.coef.beta   <-  glm(dat$y ~ logOdds[,2],family=binomial)$coef  
Beta	 <-  glm.coef.beta[2]
glm.coef.alpha   <-  glm(dat$y ~ offset(logOdds[,2]),family=binomial)$coef  
Alpha  <-  glm.coef.alpha[1]
paste("Calibration slope beta is ", round(Beta,3))
paste("Calibration in the large alpha is ", round(Alpha,5))
# Apparent Calibration slope beta is  1.8"
# Apparent Calibration in the large alpha is  -0.0002"


# In the next part we will estimate internal validity by repeatedly splitting the data 
# into training (80% of data), test (10% of data for lambda selection) and validation 
# data set (10% of data to estimate accuracy) = Nested cross-validation
# "Optional logistic regression Altzheimer internal validation using nested cross validation.R"

