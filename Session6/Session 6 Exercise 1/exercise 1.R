# Session 6: Validation - Exercise 1
library("caret")
library(foreign)

# I saved the data as a data file called data.csv
## 1. ##
# Read in the data and look at the summary statisitcs
library(readr)
data <- read_csv("data.csv")
summary(data)

# Perform a linear regression with the only true predictor X1 only as independent variable
# Then, see how much the independent variable explains the total variance of the data

result<-lm(Y~X1, data=data)
summary(result)
confint(result, level=0.95)
# In the statistical inference perspective, the coefficient estimate of X1 is 0.43 with statistically significant
# From this, we can infer that the dependent variable Y is increase by 0.43 when X1 increases by unit
# Also, the 95% confidence interval of coefficient of X1 is 0.28 ~ 0.58.
# When we fit the model with only using X1, the value of R2 is 0.24 and Adjusted R2 is 0.23
# This indicates that the model explains 23-24% of the total variance of the data only using X1

## 2. ###
# Perform a linear regression with all 90 predictors
model <- lm(Y~., data = data)
summary(model)

# By including all predictors in the model, R2, which is explained variance, increases drastically comparing to the previous one
# R2: 0.97 and Adjusted R2: 0.71 
# The difference of R2 between the models is 0.97 - 0.24 = 0.73
# Technically, the reason for increasing R2 is due to the model becomes very complex and try to match all observed patterns
# Even the random noise pattern.
# This phenomenon should lead to overfitting.

## 3. ###
# Perform internal validation with 10 fold cross-validation
# train.control is used to choose 10 fold cross-validation 

set.seed(1234) 
train.control <- trainControl(method = "cv", number = 10)  # 
# Train the model with the cv method defined in train.Control (10-fold CV)
model <- train(Y~., data=data, method="lm", trControl=train.control)

# Summarize the results
print(model)

# The cross-validated model shows 0.11 of R2 which is extremely poor performance.
# The apparent validation (the outcome from training set) explained 97% of the total variance nearly everything.
# At the internal validation stage (validation step), the performance drops sharply to 11%.
# As mentioned before, the overfitting occurs in this model.

## 4. ###
# It is better to repeat the 10 fold cross-validation several times to get a more stable estiamte of r2

# Perform internal validation with repeated 10 fold cross-validation
# train.control is used to choose 10 fold cross-validation with 10 repeats
 train.control <- trainControl(method = "repeatedcv", number = 10, repeats=10)
# Train the model
model <- train(Y~., data=data, method="lm", trControl=train.control)

# Summarize the results
print(model)

# Despite repeated cross-validation, the performance did not improve.
# R2: 0.12

# Conclusion
# In this exercise, rather than using MSE(Mean Square Error), R2 is used due to R2 is relatively more easy to understand than MSE
# The benchmark model that only X1 variable was included in the model has the value of 0.24 for R2
# The candidate model that included all predictors in the model has 0.97 for R2, which shows drastic improvement.
# However, when we cross-validate the model, the R2 dropped to 0.11 and also even repeated cross-validation shows 0.12
# There are huge difference between apparent validation and internal validation about 80% of explained variance
# This situation is called "overfitting"
