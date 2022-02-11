#############################################
#
# Introduction to Prediction Modelling.
#
# Exercise & Pratical Solutions: GLM-II.
#
#############################################
#############################################
#############################################
### Exo 1: Matrices.
# Calculating covariance matrix

library(MASS)
X <- Boston[,c("crim","rm","age","dis","lstat")]
n <- nrow(X)
p <- ncol(X)

# Matrix of column means:
M <- matrix(0,n,p)
for(j in 1:p) M[,j] <- rep(mean(X[,j]),n)

# Centered and normalized matrix with respect to n.
A <- as.matrix(X-M)/sqrt(n-1)
dim(A)

# Compute covariance matrix.
# Covariance matrix can be formed by multiplying transposed matrix and its matrix
Sig.hat <- t(A)%*%(A)
dim(Sig.hat)

### Check difference between computed version and default function in R:
round(Sig.hat-cov(X),digits=10)
# (These two matrices are identical up to floating-point approximation.)

### Matrix scatterplot:
pairs(X,pch=19,col="red3")

#############################################
### Supplementary Questions:

### Compute inverse of Sig.hat
solve(Sig.hat)

### Pre-multiply inverse of Sig.hat with Sig.hat.
solve(Sig.hat)%*%Sig.hat
### Post-multiply inverse of Sig.hat with Sig.hat.
Sig.hat%*%solve(Sig.hat)
# (These two products yield the same matrix.)

### Compare this matrix product with the identity matrix.
round(solve(Sig.hat)%*%Sig.hat - diag(1,5),digits=10)
# (The matrix product is identical to the identity matrix, up to floating-point approximation.)

### Creating a singular covariance matrix by including an incremented age variable. 
Y <- cbind(X,Boston[,"age"]+1)
cov(Y)
solve(cov(Y))
solve(cor(Y))
# (The resulting covariance matrix is non-invertible.)

#############################################
#############################################
#############################################
### Exo 2: Polynomial Regression:

### Data set
library(ISLR)
Auto

### Linear
mod1 <- lm(mpg ~ horsepower,data=Auto)
summary(mod1)

### Degree 2
mod2 <- lm(mpg ~ horsepower + I(horsepower^2),data=Auto)
summary(mod2)

### Degree 5
mod5 <- lm(mpg ~ horsepower + I(horsepower^2) + I(horsepower^3) + I(horsepower^4) + I(horsepower^5),data=Auto)
summary(mod5)
### Degree 5 (Using poly function with raw coefficients. More convenient, but same results.)
mod5 <- lm(mpg ~ poly(horsepower,5,raw=TRUE),data=Auto)
summary(mod5)
### Degree 5 (Using poly function to produce orthogonal polynomials terms. Different results.)
mod5 <- lm(mpg ~ poly(horsepower,5,raw=FALSE),data=Auto)
summary(mod5)
# Summary so far, the larger R2 values for higher order models are shown.
# Linear: 60.59%
# Second order: 68.76%
# Fifth order: 69.67%

### Plot different polynomial fits:
par(mfrow=c(1,1))
plot(Auto$horsepower,Auto$mpg,pch=19,xlab="Horsepower",ylab="MPG")
lines(sort(Auto$horsepower), fitted(mod1)[order(Auto$horsepower)], col='red3', type='l', lwd=2)
lines(sort(Auto$horsepower), fitted(mod2)[order(Auto$horsepower)], col='blue3', type='l', lwd=2)
lines(sort(Auto$horsepower), fitted(mod5)[order(Auto$horsepower)], col='green3', type='l', lwd=2)
legend(x=210,y=45,fill=c('red3','blue3','green3'),legend=c("Linear","Degree 2","Degree 5"))

### ANOVA for model of degree 5.
# ANOVA is analysis of variance
anova(mod1,mod2,mod5)
# Even if the most complex model among them shows the best performance for explaining total variance,
# it is relatively not significant comparing to second order model.

### Considering higher degrees. 
mod6 <- lm(mpg ~ poly(horsepower,6,raw=TRUE),data=Auto)
mod7 <- lm(mpg ~ poly(horsepower,7,raw=TRUE),data=Auto)
mod8 <- lm(mpg ~ poly(horsepower,8,raw=TRUE),data=Auto)
anova(mod5,mod6,mod7,mod8)

# As a result, ANOVA shows decreasing F-value when the model is getting to higher order (complex model)
# Making model complex may increase its performance of prediction or generalisation of the data.
# If the model is too complex, the model would be less efficient and, furthermore, it may lead to overfitting later
####
# EoF
