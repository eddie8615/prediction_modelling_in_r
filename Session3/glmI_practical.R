#############################################
#
# Introduction to Prediction Modelling.
#
# Exercise & Pratical Solutions: GLM-I.
#
############################################
### Exo 1: Simple Regression.

### Load package.
library(MASS)
names(Boston)

### Fit 
lm.fit <- lm(medv~lstat,data=Boston) 
summary(lm.fit)

### Extract confidence intervals.
names(lm.fit)
coef(lm.fit)
confint(lm.fit)

### Plot the model.
attach(Boston)
x11(pointsize=22)
plot(lstat,medv)
abline(lm.fit,lwd=3,col="red3")
detach(Boston)

#############################################
### Exo 2: Simulate data with unrelated variables. 

require(car)
set.seed(0)

### Simulated data.
n <- 1000
y.mu <- 2;
x.mu <- 4;
y.sig <- 1;
x.sig <- 10;
y <- rnorm(n, y.mu, y.sig)
x <- rnorm(n, x.mu, x.sig)

### Construct a data.frame:
data <- data.frame(y,x)
names(data)

### Plot data:
plot(y ~ x); 
scatterplot(y~x,data)

### Two different ways of computing the column means:
colMeans(data)
sapply(data,mean)

### Covariance of x and y, computed 'manually':
xy.sig2.hat <- 0
for(i in 1:n) xy.sig2.hat <- xy.sig2.hat + sum((y[i]-mean(y))*(x[i]-mean(x)))/(n-1)
print(xy.sig2.hat)
### Compare with base-R function for covariance matrix.
var(data)
var(data)[1,2]; xy.sig2.hat

### Linear model, and outputs:
mod <- lm(y ~ x, data)
mod
print(mod)
summary(mod)

### Linear model with z-scores:
y.z <- (y - mean(y))/sd(y);
x.z <- (x - mean(x))/sd(x);
mod.z <- lm(y.z ~ x.z); mod.z;
summary(mod.z)

### Compare correlation coefficient
# with regression coefficient of standardized model.
coef(mod.z)[2]; cor(x,y)

####
# EoF
