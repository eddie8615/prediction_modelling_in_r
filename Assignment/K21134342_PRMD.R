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
