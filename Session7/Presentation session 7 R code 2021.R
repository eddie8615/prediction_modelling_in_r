library(MASS)
library(glmnet)
## Set working directory
# Set working directiory to source files location
setwd("~/course_materials/2-Tuesday")

# Import the heart data set, description see lecture 
heart<-read.csv("heart.csv")
View(heart)
head(heart)
summary(heart)
str(heart)

#### Ridge Regression using glmnet function  #####
#### Dependent (y) and predictors 9x) need to be stored in seperate objects
#### Important x and y need to be a matrix/vector object!
x<-as.matrix(heart[,-1])
y<-as.matrix(heart[,1])

# Using the heart dataset, a linear RIDGE model can be fitted and plotted using following R commands:

#### We fit the model using the function glmnet. 
fit1<-glmnet(x, y, family = "gaussian", alpha = 0)
### We can visualize the coefficients by using the plot function
plot(fit1, xvar="lambda", label=TRUE)

### The cv.glmnet command can be used to find optimal penalty  by cross-validation
# Again we need to set alpha to 0
set.seed(101)  
cv10<-cv.glmnet(x,y, nfold=10, alpha=0)
plot(cv10) # plot of the average MSE for each lambda

# To find the coefficent of the best model, we need to set lambda to the minimum lambda 131.131
cv10$lambda.min
### The fitted coefficients at the optimal penalty can be obtained by
### using the coef command 

coef(cv10, s="lambda.min")
## Show results with 2 digits only
round(coef(cv10, s="lambda.min"), digits=2)


# Hastie et al recommend the use of a slighter stronger penalty. 
# Less variables are selected (sparse model) and the model performs often better in new data sets
# This penalty is the minumumlambda vlaue + 1SE
cv10$lambda.1se
# The fitted coefficients at the lambda + 1se can be obtained by  using the  coef command 
coef(cv10, s="lambda.1se")
round(coef(cv10, s="lambda.1se"), digits=2)
# Compare the coefficents with a simple linear regression model
ols_reg<- glm(sbp~., data=heart)
summary(ols_reg)
# Compare the cross-validated MSE of the RIDGE model with the cross-valdate MSE of a simple regression (OLS)
library(boot)
cv_ols_reg <- cv.glm(heart, ols_reg, K=10)

# Results are stroed in  $delta
# Delta is a vector of length two. The first component is the raw cross-validation estimate of prediction
# error. The second component is the adjusted cross-validation estimate. 
# The adjustment is designed to compensate for the bias introduced by not using leave-one-out cross-validation.
cv_ols_reg$delta
# The MSE of the first delta is slightly larger thanthe MSE of the best lambda:
mse.min <- cv10$cvm[cv10$lambda == cv10$lambda.min]
# The MSE of the first delta is smaller than the MSE of the best lambda + 1SE:
mse.min.1se <- cv10$cvm[cv10$lambda == cv10$lambda.1se]
print(paste("The cross-validated MSE of the OLs, the min lambda and min lmabda + 1Se models are: ", cv_ols_reg$delta[1],mse.min,mse.min.1se))

# Using the heart dataset, a linear LASSO model can be fitted and plotted using following R command

library(glmnet)
x<-as.matrix(heart[,-1])   # Important x and y need to be a matrix/vector object!
y<-as.matrix(heart[,1])
fit1<-glmnet(x, y, family = "gaussian", alpha = 1)
plot(fit1, xvar="lambda", label=TRUE)

# The cv.glmnet command can be used to find optimal penalty  by cross-validation
set.seed(101)
cv10<-cv.glmnet(x,y, nfold=10)
plot(cv10)

# The  optimal value of  can be obtained by extracting the lambda.min component of the fitted cv.glmnet object
cv10$lambda.min

# The fitted coefficients at the optimal penalty can be obtained by  using the  coef command 
coef(cv10, s="lambda.min")


# Hastie et al recommend the use of a slighter stronger penalty. 
# Less variables are selected (sparse model) and the model performs often better in new data sets
# This penalty is the minumumlambda vlaue + 1SE
cv10$lambda.1se
# The fitted coefficients at the lambda + 1se can be obtained by  using the  coef command 
coef(cv10, s="lambda.1se")

#Compare the crossvlaidated MSE of the lasso model with the OLS model (see above)
# The MSE of the first delta is slightly larger thanthe MSE of the best lambda:
mse.min <- cv10$cvm[cv10$lambda == cv10$lambda.min]
# The MSE of the first delta is smaller than the MSE of the best lambda + 1SE:
mse.min.1se <- cv10$cvm[cv10$lambda == cv10$lambda.1se]
print(paste("The cross-validated MSE of the OLs, the min lambda and min lmabda + 1Se models are: ", cv_ols_reg$delta[1],mse.min,mse.min.1se))
# The min lasso has got the smallest MSE
