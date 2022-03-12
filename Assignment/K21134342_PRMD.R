# library(glmnet)
library(caret)
library(foreign)

# set seed number as follwing the guide
set.seed(123)

# 1.1 Importing data and data manipulation
data = read.spss('tbi.sav', to.data.frame=TRUE)

# Removing unusing variables: d.gos, d.mort
data <- subset(data, select = -c(d.gos, d.mort))
# Also, removing redundant variables: hb, pupil.i, sodium, glucose
data <- subset(data, select = -c(hb, pupil.i, sodium, glucose))

# Factorise the categorical variables
data$trial <- factor(data$trial)
data$d.unfav <- factor(data$d.unfav)
data$cause <- factor(data$cause)

dim(data)
# Remove missing data
data <- na.omit(data)
dim(data)

# About 1000 observations were removed
summary(data)

# 1.2 Model 1: Logistic regression

# install.packages("pmsampsize")
library(pmsampsize)

# Prevalence
summary(data$d.unfav)
# 0   1
# 653 458 => 0.4122412
prevalence <- 458 / (653 + 458)

# Power analysis
pmsampsize(type="b", rsquared=0.2, parameters = 19, shrinkage = 0.9, prevalence = prevalence, seed=123)

nrow(data) # 1111
nrow(data[data$trial == "Tirilazad US",]) # 545

# Cite the package
citation("pmsampsize")

# The maximum number of params that used for analysis of both dataset was 13 suggested by pmsampsize
pmsampsize(type="b", rsquared=0.2, parameters = 13, shrinkage = 0.9, prevalence = prevalence, seed=123)

# split data into training and testing
train <- data[data$trial == "Tirilazad US",]
train <- train[,-1]
train$d.unfav <- factor(train$d.unfav)

test <- data[data$trial != "Tirilazad US",]
test <- test[,-1]
test$d.unfav <- factor(test$d.unfav)

set.seed(123)
model <- glm(d.unfav ~ ., data = train,  family="binomial")

library(pROC)
# summarise coefficients and their statistics
summary(model)
options(scipen=9)
# Calculate odds ratios and their confidence interval
exp(cbind(coef(model), confint(model)))

# predict the outcome probability by using global dataset
y_prob<-predict(model,newdata = test[,-1], type="response")
y_prob[y_prob >= 0.5] <- 1
y_prob[y_prob < 0.5] <- 0

confusionMatrix(as.factor(y_prob), as.factor(test[,1]), positive="1")

roc(test[,1], y_prob, ci=TRUE)

# 1.3 Model 2 and 3: Regularised Logistic regression

# Modelling LASSO with minimum lambda
min_trControl <- trainControl(method = "repeatedcv", 
                          repeats = 10,          
                          number = 10,
                          selectionFunction = "best") # best - minimum lambda, oneSE for minimum lambda + 1 Se, Tolerancwe for minimum + 3%
set.seed(123)
params = expand.grid(alpha=c(1),   # L1 & L2 mixing parameter
                     lambda=2^seq(1,-10, by=-0.1))

min_lambda_model<- train(d.unfav ~ .,             
                    data = train,         
                    method = "glmnet",    
                    family="binomial",    
                    trControl = min_trControl, 
                    tuneGrid = params)

y_prob<-predict(min_lambda_model, type="prob", test[,-1])
y_pred<-as.numeric(predict(min_lambda_model, type="raw",test[,-1]))-1

confusionMatrix(as.factor(y_pred), test[,1], positive="1")

roc(test[,1], y_prob$`1`, ci=TRUE)


# Modelling LASSO with lambda + 1SE
se_trControl <- trainControl(method="repeatedcv",
                             repeats = 10,
                             number=10,
                             selectionFunction = "oneSE")

lambda_1se_model <- train(d.unfav ~.,
                          data = train,
                          method = "glmnet",
                          family = "binomial",
                          trControl=se_trControl,
                          tuneGrid = params)

se_y_prob <-predict(lambda_1se_model, type="prob", test[,-1])
se_y_pred<-as.numeric(predict(lambda_1se_model, type="raw",test[,-1]))-1

confusionMatrix(as.factor(se_y_pred), test[,1], positive = "1")

roc(test[,1], se_y_prob$`1`, ci=TRUE)

# listing coefficients for both models
coef(min_lambda_model$finalModel, min_lambda_model$bestTune$lambda)
coef(lambda_1se_model$finalModel, lambda_1se_model$bestTune$lambda)
