###### Session 7: Regularized Regression I ##########
#### Dealing with overfitting via Penalised Regression - Practical  ####

# 1(a). 
# Fit a set of ridge regression models using a range of penalty values (=0, 0.1, 0.2, …, 4.8, 4.9, 5.0) and plot the ridge path (regression coefficients against  ).
library(MASS)
library(glmnet)
library(haven)

# Import the heart data set
import.data<-read.csv("import.csv")

# Exploring the data
View(import.data)
head(import.data)
summary(import.data)
str(import.data)
# Correlation matrix 
cor(import.data)  # all pairwise correlations

# The correlation shows that each variable is highly correlated with other.
# The range is from 0.97 - 0.99

##(1a)

#### Fit a set ridge regression models using a range of lambda penalty values using glmnet 
#### and plot the ridge path (regression coefficients against  lambda).
#### Ridge Regression using glmnet function 
#### Dependent (y) and predictors (x) need to be stored in separate objects
#### Important x and y need to be a matrix/vector object!
x<-as.matrix(import.data[,-1])
y<-as.matrix(import.data[,1])
#### We fit the model using the function glmnet. 
fit1<-glmnet(x, y, family = "gaussian", alpha = 0)
#### We can visualize the coefficients by using the plot function
plot(fit1, xvar="lambda", label=TRUE)


##(1b)
#### Plot the cross-validated mean squared error (MSE) of the fitted ridge object against lambda  
set.seed(101)  
cv10<-cv.glmnet(x,y, nfold=3, alpha=0)  # nfold needs to be reduced to 3 because of the small sample size
plot(cv10) # plot of the average MSE for each lambda


##1(c).
#### Find the optimal value of the penalty that minimises MSE. 
#### Obtain the fitted regression coefficients at optimal penalty.  

# Lambda of the best model, 
cv10$lambda.min
### The fitted coefficients at the optimal penalty can be obtained by
### using the coef command 

coef(cv10, s="lambda.1se")
## Show results with 2 digits only
round(coef(cv10, s="lambda.min"), digits=2)
# No coefficent becomes 0!
cor(import.data)

# Intercept) -14083173.04
# year             7130.10
# gnp                35.38
# cpi               368.91




# Question 2: 
#  
#  The MRI brain imaging data (brain.csv) contains disease status (disease, coded 0, 1, and 2 for 
# the control, mild cognitive impairment and Alzheimer’s disease respectively)
# for n=805 subjects from the Alzheimer's Disease Neuroimaging Initiative (ADNI) study (www.loni.ucla.edu/ADNI).  
# The dataset also contains information on 43 covariates (age, sex, APOE genotype and 
# 40 other variables measuring brain volumes of different segmented regions of the brain).  

#2(a)

#Fit a penalised regression model for predicting the disease progression outcome based on the available covariates
# using LASSO. Plot the penalised regression coefficients against the penalty values.

library(glmnet)
data<-read.csv("brain.csv")
colnames(data)
y<-data$disease
x<-as.matrix(data[,-1])
lasso<-glmnet(x,y, family="gaussian", alpha=1)
plot(lasso, xvar="lambda")


2(b)
#
# Use 10-fold cross-validation to find the optimal penalty. 
# Select (identify) the brain volume measures that are predictive of the progression of Alzheimer’s disease
# based on the optimal LASSO model.

set.seed(101)
cv10<-cv.glmnet(x,y, nfold=10)
plot(cv10)
cv10$lambda.min
coef(cv10, s="lambda.min")

# By using LASSO model, the weak predictors in the model has 0 coefficient and only the predictors effecting the outcome well remain
# LASSO model helps us to reduce the dimension of the data by penalising irrelevant predictors heavily.


