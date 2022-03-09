library(glmnet)
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
pmsampsize(type="b", rsquared=0.2, parameters = 20, shrinkage = 0.9, prevalence = prevalence, seed=123)

nrow(data) # 1111
nrow(data[data$trial == "Tirilazad US",]) # 545

# Cite the package
citation("pmsampsize")

# The maximum number of params that used for analysis of both dataset was 13 suggested by pmsampsize
pmsampsize(type="b", rsquared=0.2, parameters = 13, shrinkage = 0.9, prevalence = prevalence, seed=123)

train <- data[data$trial == "Tirilazad US",]
train <- train[,-1]

test <- data[data$trial != "Tirilazad US",]
test <- test[,-1]

set.seed(123)
model <- glm(d.unfav ~ ., data = train,  family="binomial")
summary(model)
library(pROC)

confint(model)
# confusionMatrix(as.factor(y_pred), as.factor(y_test), positive="1")

