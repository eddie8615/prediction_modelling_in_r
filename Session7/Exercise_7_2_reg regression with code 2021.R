###### Session 7: Regularized Regression I ##########
######   Exercise 7.2. Does lasso regression really work?  #######

# We can demonstrate it with a simple simulation. 
# We simulate data from a known model and then apply a lasso regression to see if it identifies the correct model 
# or at least performs better than a multiple linear regression

library(glmnet)
# library(caret)

#### Simulation ####
# Simulate a data set with 90 x and 1 y variable. 
# Y is only related with 1 x variables (X1)
# All other variables are simply random normal-distributed numbers
# with mean fo 0 and SD of 1

set.seed(4)

x<-matrix(rnorm(9000,0,1),nrow=100, ncol=90)  # matrix with 90 x variables and 100 obs
summary(x)

y = rnorm(100)+x[,1]/2 # y = 0.5* x1 + random error (N(0,1)

# PS If you have time you may want to another simulations with 25 predictors
# You could also compare lasso with ridge regression
# simulate a dat set with 25 predictors
# y1 = rnorm(100) + (x[,1]+x[,2]+x[,3]+x[,4]+x[,5]+x[,6]+x[,7]+x[,8]+x[,9]+x[,10]+x[,11]+x[,12]+x[,13]+x[,14]+x[,15]+x[,16]+x[,17]+x[,18]+x[,19]+x[,20]+x[,21]+x[,22]+x[,23]+x[,24]+x[,25])/2

# Z-transform y: If both y and X have a mean of 0 and variance of 1 we can interpete 
# the regression coefficents as standardised beta
# get the mean of y : mean(y) 
# get the sd of y: sd(y)
y <- (y - mean(y)) / sd(y)  # z transform y so that mean is 0 and Sd=1
summary(y)
sd(y)
## Optional: Comment for the purists: We probbaly should use as the theoretical SD 1.22 for y: 
## Var(y) + 0.5 Var(x) = Var(y) + 0.5**2(Var(X))
## = Var(y) + 1/4(Var(x)) = 1+1/4 = 1.25  --> SD= sqrt(1.25)= 1.22
## http://www.stat.ncsu.edu/people/dickey/courses/st711/notes/Notes_pdf/ThreeRules.pdf

# Perform a linear regression
# a) with y and x1 only
r1<-lm(y~x[,1])
summary(r1)  # b should be about 0.5
# 20% of variance is explained by the model

# b) do a multiple regression with all 90 x variables
lr<-(lm(y~x))
summary(lr)

# Questions:
# In this multiple regression, 98.41% of variance is explained and Adjusted R2 is 82.48%
# Also, relatively many variables are statistically significant with 5% of alpha.
# However, although the true predictor (x1) was selected, it is not the most significant variable among the variables
# Why is it overfitting on the data?

##### Model selection using lasso regression ####### 
# Fit a penalised regression mode
# Gaussian means we perform a linear regularized regression with continuous outcome
# alpha = 1 means that we perform lasso regression 
# for a large range of lambdas (100 is default) 
fit1<-glmnet(x, y, family = "gaussian", alpha = 1)

# Plot the penalised regression coefficients against the penalty values. 
plot(fit1, xvar="lambda", label=TRUE)    # plots the coefficents against (log)lamda

# Use 10-fold cross-validation to find the optimal penalty
# The cv.glmnet command can be used to find optimal penalty  by cross-validation

set.seed(101)           # allows us to use the same random vlaue to get the same results, alpha =1 is lasso
cv10<-cv.glmnet(x,y, alpha=1, nfold=10)  # nfold=10 --> 10 fold cv is used to find optimal lambda

# Identify the best lambda: Plot log lambda against MSE of unseen cases 
plot(cv10)  # Vertical lines are drawn at minimum MSE and (min + 1SE) MSE respectively  

# The  optimal value of lambda can be obtained by extracting the lambda.min component of the fitted cv.glmnet object
cv10$lambda.min

# The fitted coefficients at the optimal penalty can be obtained by  using the  coef command 
coef(cv10, s="lambda.min") 

# LASSO only left 3 variables in the model and made the rest of coefficients toward 0
# which means that the rest variables are not hugely considered and removed by regularisation.


# Additional information: Get the MSE for the best lambda model. This is the MSE of  
MSE <- min(cv10$cvm)  # $cvm includes the MSE of all 100 lambdas , min() selects the snmallest
# This is the MSE of the predicted test data 
# Alternative to get MSE for minimum lambda and to get MSE for minimum + 1SE lambda
MSE_min<- cv10$cvm[which(cv10$lambda == cv10$lambda.min)]
MSE_min
# You can also get the min and 1Se lambda and the MSE by using 
print(cv10)


# Explained variance of unseen cases
r2 <- cv10$glmnet.fit$dev.ratio[which(cv10$glmnet.fit$lambda == cv10$lambda.min)]

paste("R2 is: ",  r2)

# There are a lot of formulas for R^2 that can be used.
# Glmnet calculates the fraction of (null) deviance explained (for linear models this is the R-square). 
# See Kvalseth. Cautionary note about R^2. American Statistician (1985) vol. 39 (4) pp. 279-285.
# An alternative which provides often a different result with unstable models is: paste("R2 is: ",  1-(MSE_min/var(y)))


# Apparent validation
# Predict values for the training data set (apparent validity)
predictions<-predict(cv10,newx=x,s="lambda.min",type="response")
# Explained variance within the sample (apparent validation)
# Model prediction performance using caret functions
postResample(pred = predictions, obs = y)
# or: 
paste("The apparent RMSE of model  is: ", RMSE(predictions, y) )
paste("The apparent R2 is: ",  R2(predictions, y))
paste("The apparent MSE of model  is: ", (RMSE(predictions, y)^2))




# Alternative to get MSE for minimum lambda and to get MSE for minimum + 1 SE lambda
# Alternative best lambda (lambda.1SE)
# This lambda penalizes slightly more than the minimum lambda
# It selects a more parsimonious model with almost the same MSE
# It is often better if the changes of lambda around the minimum are very small
# How many variables are selected 
coef(cv10, s="lambda.1se")

# Addtional information: Get MSE for minimum + 1SE lambda
MSE_1SE<- cv10$cvm[which(cv10$lambda == cv10$lambda.1se)]
MSE_1SE
# Calculate the explained variance 
r2 <- cv10$glmnet.fit$dev.ratio[which(cv10$glmnet.fit$lambda == cv10$lambda.1se)]
paste("R2 is: ",  r2)

### End of practical ########

