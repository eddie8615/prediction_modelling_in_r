#############################################
#
# Introduction to Prediction Modelling.
#
# Exercise & Pratical Solutions: GLM-III.
#
#############################################
#############################################
#############################################
### Exo 1: Multicollinearity
# (Chapter 3 in ISLR.)

library(MASS)

############
### MODEL (1).
############
### Fit multiple regression with all variables, using a dot in the formula of lm,
# to represent all other variables.
mod.all <- lm(medv~.,data=Boston)
summary(mod.all)

### Compute Variance Inflation Factors (VIFs) of all variables:
require(car)
vif(mod.all)
# (Several of the variables have VIFs greater than 5.) 

### Identify variable with largest VIF.
which(vif(mod.all)==max(vif(mod.all)))

############
### MODEL (2).
############
### Remove variable with largest VIF from model.
mod.all.notax=lm(medv~.-tax,data=Boston)
summary(mod.all.notax)

### An identical way to update your model 
mod.all.notax=update(mod.all, ~.-tax)
summary(mod.all.notax)

### Compare R^2 of two models.
c(summary(mod.all)$r.squared,summary(mod.all.notax)$r.squared)
# R^2 for full model is greater than for model without tax. 
summary(mod.all)$r.squared > summary(mod.all.notax)$r.squared
# 0.74 for full model and 0.73 for the model excluding tax variable.
# The result does not show that the tax variable is not significantly important for modelling
# Due to the small amount of difference of R2 between the mdoels.


### F-test comparing the two models.
anova(mod.all,mod.all.notax)
### Consider also the VIFs of the variables in the model after removing tax. 
vif(mod.all.notax)

#############################################
#############################################
#############################################
### Exo 2: Finding the proverbial needle in the haystack.
# (One significant variable in the midst of many insignificant ones.)

### n=p+1.
n  <- 100
p  <- 99
mu <- 0; sig <- 1
y  <- rnorm(n,mu,sig)
X  <- matrix(0,n,p+1)

### The correlated variable:
X[,1] <- y + rnorm(n)
### The uncorrelated variables:
for(j in 2:(p+1)) X[,j] <- rnorm(n,mu,sig)

### Put it all together as a data frame for analysis. 
data <- data.frame(cbind(y,X));
names(data) <- c("y",paste0("x",1:100))
    
### Compare regressions for x1 and x2, independently.
summary(lm("y~x1",data))
summary(lm("y~x2",data))

### Coefficients for variables 1 to 98 included in the model.
formula <- paste0("y~x1",paste0("+x",2:98,collapse=""))
print(formula)
lm.fit98 <- lm(formula,data)
### Print summary of coefficients for the first 5 parameters. 
summary(lm.fit98)$coefficients[1:5,]

### Plot a histogram of the coefficients:
hist(coef(lm.fit98),col="red3",main="Coefficients of Saturated Model",xlab="Coefficients")
# (The coefficients are essentially normally distributed around zero.)
# Which means most of the coefficients are not significant in the model
# This variables will be excluded when we use regularisation such as LASSO

###################
### Saturated (singular) models.

### Coefficients for variables 1 to 99 included in the model.
formula <- paste0("y~x1",paste0("+x",2:99,collapse=""))
print(formula)
lm.fit99 <- lm(formula,data)
### Print summary of coefficients for the first 5 parameters. 
summary(lm.fit99)$coefficients[1:5,]

### Coefficients for variables 1 to 100 included in the model.
lm.fit100 <- lm(y~.,data)
summary(lm.fit100)$coefficients[1:5,]

####
# EoF
