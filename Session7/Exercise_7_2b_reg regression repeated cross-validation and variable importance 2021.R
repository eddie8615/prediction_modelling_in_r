###### Session 7: Regularized Regression I ##########
###### 7.2b: Repeated cross-validation ##############

# If you repeat the cv.glmnet function you will get always slightly different results, esp. if 
# the sample size is small (relative to th number of predictors). It is therefore recommended 
# to rerun the cross-validation analyses several times and to take the average MSE (or other accuracy measures)
# of the overall best lambda. 
# Repeated cross-validation provides a better estimate of the test-set error!
# see e.g. http://rstudio-pubs-static.s3.amazonaws.com/251240_12a8ecea8e144fada41120ddcf52b116.html
# We will use the package caret and the functions "trainControl" and "train"

library(caret)
library(glmnet)
library("doParallel")  # for parallel computing

#### Simulation ####
# Simulate a data set with 90 x and 1 y variable. 
# Y is only related with 1 x variables (X1)
# All other variables are simply random normal-distributed numbers
# with mean fo 0 and SD of 1


# First we need our simulated data set from exercise 7.2. 
set.seed(4)
x<-matrix(rnorm(9000,0,1),nrow=100, ncol=90)  # matrix with 90 x variables and 100 obs
y = rnorm(100)+x[,1]/2 # y = 0.5* x1 + random error (N(0,1)
y <- (y - mean(y)) / sd(y)  # z transform y so that mean is 0 and Sd=1
# Perform a linear regression
# a) with y and x1 only
r1<-lm(y~x[,1])
summary(r1)  # b should be about 0.5
# b) do a multiple regression with all 90 x variables
lr<-(lm(y~x))
summary(lr)

# Let's start

# train.control is used to choose 10 fold cross-validation with 10 repeats
set.seed(1234) 
# We used for the validation of our regression model the train.control and train function of the caret package.
# train.control <- trainControl(method = "repeatedcv", number = 10, ,repeats=10)
# We will use it again but need to modify it for glmnet
# Set up training settings object

trControl <- trainControl(method = "repeatedcv", 
                          number = 10   ,# Number of folds
                          repeats = 10 , # Number of iterations for repated CV. increase until results stabilize
                          selectionFunction = "best")     # set grid of params to test over, if not specified defualt gris is used (not always the best))  # best - minimum lambda, oneSE for minimum lambda + 1 Se, Tolerancwe for minimum + 3%

# Caret needs outcome and predictor in one object. The object needs to be a dataframe.
# Caret allows the use of factors and does dummy coding internally
# We need to combine our y and x variables using cbind
mydata <- as.data.frame(cbind(y,x))  # caret needs outcome and predictors in one object. 
summary(lm(y~x[,1], data=mydata))  # same results

### Set up grid of parameters (lambdas) to test.
# It is better to provide glmnet a user-defined grid of lambdas! 
# We have to set alpha to either 0 (Ridge) or 1 (Lasso)

params = expand.grid(alpha=c(1),   # L1 & L2 mixing parameter
                     lambda=2^seq(1,-10, by=-0.1)) # regularization parameter lambda
params  # lambda ranges from 0.001 to 2, this often works well. You can have smaller units by changing b= -0.3 to by = -0.1

# Train the model
cl=makeCluster(4);registerDoParallel(cl)   # using more than one core

model <- train(y ~ .,                 # model formula (. means all features)
               data = mydata,         # data.frame containing training set
               method = "glmnet",     # model to use
               trControl = trControl, # set training settings
               tuneGrid = params)     # set grid of parameters to test over, if not specified defualt gris is used (not always the best)



# Summarize the results
print(model)
# The final values used for the model were alpha = 1 and lambda = 0.07179365.
# R2: 20.32%

# This function automatically selects the best model (either minimum lambda or lamda + 1 Se)
get_best_result = function(caret_fit) {
  best = which(rownames(caret_fit$results) == rownames(caret_fit$bestTune))
  best_result = caret_fit$results[best, ]
  rownames(best_result) = NULL
  best_result
}
# Get alpha, lambda, MSE, r2, MAE (Mean absolute error) and the SD of each (RMSESD, RsquaredSD, MAESD) for
# of the best tmodel (=on average best predcition of unseen cases)

get_best_result(model)
# alpha    lambda      RMSE  Rsquared       MAE    RMSESD RsquaredSD     MAESD
#   1   0.2176376 0.9118986 0.2418601 0.7132472 0.2173866  0.1974693 0.1561621
# MSE = RMSE^2
print(paste("MSE of best model is ", get_best_result(model)[[3]]^2))
# "MSE of best model is  0.82"
# The r2 is 0.42
# Important: Esp if sample size is small this r2 cannot be used for internal validation
# See session: Nested cross-validation

# Plot performance for different paramameters
plot(model, xTrans=log, xlab="log(lambda)")
# Plot regularization paths for the best model
plot(model$finalModel, xvar="lambda", label=T)


# Save best alpha and lambda
best_alpha <-get_best_result(model)$alpha  # alpha is set to 1 (lasso)
best_lambda <- get_best_result(model)$lambda

# Model coefficients
# The fitted coefficients at the optimal penalties can be obtained by  using the  coef command 
coef(model$finalModel, s=best_lambda, alpha=1)  # s = best_lanbda 
coef(model$finalModel, s=model$bestTune$lambda)  # 

# Predict values for the training data set (apparent validity)
predictions<-predict(model$finalModel,newx=x,s=model$bestTune$lambda, alpha=best_alpha,type="response")
#Explained variance within the sample (apparent validation)
# Model prediction performance using caret functions
paste("The apparent RMSE of model  is: ", RMSE(predictions, y) )
paste("The apparent MSE of model  is: ", (RMSE(predictions, y)^2))
paste("The apparent R2 is: ",  R2(predictions, y))

# Apparent r2 is 0.62






# 2. Variable importance: glmnet presents the coefficents in a backtransformed format on the original scale
# To get the coefficient in a space that lets you directly compare their importance, you have to standardize them. 
# standardised Beta for variable xj = Bj *(SD(Xj)/SD(Y)) 
# dummy coded categorical variables can be standardised as well, intepretation of coefficent is difficult, use it only for variable importance 


sd_x <- as.matrix(apply(x, 2, sd))
sd_y<-sd(y)
coefs <- as.matrix(coef(cv10, s = "lambda.min"))
coefs<-coefs[-1,]
std_coefs <- coefs * as.matrix(sd_x)*sd_y
std_coefs

# Simple Dotplot
dotchart(std_coefs,labels=row.names(std_coefs),cex=.7,
         main="Variable importance plot",
         xlab="Standardised coefficents")

# Sort by size of regression coefficents
ordered <- std_coefs[order(std_coefs)] # sort by regression coefficents

dotchart(ordered,labels=(order(std_coefs)),cex=.7,
         main="Variable importance plot",
         xlab="Standardised coefficents")





# 3. Obtaining confidence intervals using bootstrapping using the R package HDCI

"Hanzhong Liu, Xin Xu and Jingyi Jessica Li (2017) A Bootstrap Lasso + Partial Ridge Method to Construct Confidence Intervals for
Parameters in High-dimensional Sparse Linear Models:  https://arxiv.org/pdf/1706.02150.pdf"

"Overall, the Bootstrap Lasso+OLS method has the shortest confidence interval lengths
with good coverage probabilities for large coefficients, while for small but nonzero coef33
ficients, the Bootstrap LPR method (rBLPR and pBLPR) has the shortest confidence
interval lengths with good coverage probabilities. Therefore for practitioners, if they care
only about the confidence intervals for larger coefficients, we recommend the Bootstrap
Lasso+OLS method; if they are also concerned with small coefficients, the Bootstrap
LPR is a better choice."


library("glmnet")
library("mvtnorm")
library("HDCI")


## residual bootstrap Lasso+OLS
obj <- bootLassoOLS(x = x, y = y, B = 100, ,cv.method="cv" )
## residual bootstrap bootLPR: residual bootstrap Lasso+Partial Ridge 
obj <- bootLPR(x = x, y = y, B = 100, ,cv.method="cv" )
## residual bootstrap bootLPR: residual bootstrap Lasso+OLS + Partial Ridge 
obj <- bootLOPR(x = x, y = y, B = 100, ,cv.method="cv" )


#Best Lambda: The optimal value of lambda selected by cv/cv1se/escv.
obj$lambda.opt
#Lasso+OLS estimate of the regression coefficients
obj$Beta.LassoOLS
obj$Beta.LPR
obj$Beta.LOPR
# confidence interval
obj$interval.LassoOLS
obj$interval.LPR
obj$interval.LOPR
#sum((obj$interval[1,]<=beta) & (obj$interval[2,]>=beta))
## using parallel in the bootstrap replication
library("doParallel")
registerDoParallel(4)
set.seed(0)
system.time(obj <- bootLassoOLS(x = x, y = y))
system.time(obj <- bootLassoOLS(x = x, y = y, parallel.boot = TRUE, ncores.boot = 4))



