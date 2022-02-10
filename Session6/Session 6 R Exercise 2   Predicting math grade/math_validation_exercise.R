###### Session: Validation I ##########
#
# Predicting final grade in math
#
# In this exercise we want to build a simple multiple regression model to predict final grade in math
# in two secondary Portuguese schools in 2005. 

# The data was collected by using school reports and questionnaires and include student grades, 
# demographic, social and school related features. Two datasets are provided regarding the 
# performance in two distinct subjects in secondary school: Mathematics (mat) and Portuguese language (por).
# We will use only the math data set with a total sample size of 395. Baseline data were collected
# at the beginning of the school year. Outcome is final grade at the end of the year. It ranges from 0 to 20
#  0-9: failed, 10-11: Sufficient, 12-13: satisfactory, 14-15: good, 16-20: very good/excellent)  
# We will use the numeric grades as outcomes. 
# The research question is: Can we reliable predict students performance in math?
  
# The original data set was published and analysed by Cortez and Silva (2008)
# P. Cortez and A. Silva. (2008) Using Data Mining to Predict Secondary School Student Performance.
# In A. Brito and J. Teixeira Eds., Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008) 
# pp. 5-12, Porto, Portugal, April, 2008, EUROSIS, ISBN 978-9077381-39-7.

# Data set
# Attributes 
# school - student's school (binary: 'GP' - Gabriel Pereira or 'MS' - Mousinho da Silveira)
# sex - student's sex (binary: 'F' - female or 'M' - male)
# age - student's age (numeric: from 15 to 22)
# address - student's home address type (binary: 'U' - urban or 'R' - rural)
# famsize - family size (binary: 'LE3' - less or equal to 3 or 'GT3' - greater than 3)
# parent_cohabitation_status - parent's cohabitation status (binary: 'T' - living together or 'A' - apart)
# mother_edu - mother's education (numeric: 0 - none, 1 - primary education (4th grade), 2 - 5th to 9th grade, 3 - secondary education or 4 - higher education)
# father_edu - father's education (numeric: 0 - none, 1 - primary education (4th grade), 2 - 5th to 9th grade, 3 - secondary education or 4 - higher education)
# Mjob - mother's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')
# Fjob - father's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')
# reason - reason to choose this school (nominal: close to 'home', school 'reputation', 'course' preference or 'other')
# guardian - student's guardian (nominal: 'mother', 'father' or 'other')
# traveltime - home to school travel time (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour)
# studytime - weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)
# failures - number of past class failures (numeric: n if 1<=n<3, else 4)
# schoolsup - extra educational support (binary: yes or no)
# famsup - family educational support (binary: yes or no)
# paid - extra paid classes within the course subject (Math or Portuguese) (binary: yes or no)
# activities - extra-curricular activities (binary: yes or no)
# nursery - attended nursery school (binary: yes or no)
# higher - wants to take higher education (binary: yes or no)
# internet - Internet access at home (binary: yes or no)
# romantic - with a romantic relationship (binary: yes or no)
# famrel - quality of family relationships (numeric: from 1 - very bad to 5 - excellent)
# freetime - free time after school (numeric: from 1 - very low to 5 - very high)
# goout - going out with friends (numeric: from 1 - very low to 5 - very high)
# Dalc - workday alcohol consumption (numeric: from 1 - very low to 5 - very high)
# Walc - weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)
# health - current health status (numeric: from 1 - very bad to 5 - very good)
# absences - number of school absences (numeric: from 0 to 93)
# 
# final_grade- final grade in maths (numeric: from 0 to 20, output target)

# Data can be obtained from: 
# math_student_data_url <- "https://raw.githubusercontent.com/arunk13/MSDA-Assignments/master/IS607Fall2015/Assignment3/student-mat.csv";
# math_student_data <- read.table(file = math_student_data_url, header = TRUE, sep = ";");
# por_student_data_url <- "https://raw.githubusercontent.com/arunk13/MSDA-Assignments/master/IS607Fall2015/Assignment3/student-por.csv";
# por_student_data <- read.table(file = por_student_data_url, header = TRUE, sep = ";");

# math<-as.data.frame(math)
# install.packages("fastDummies")
# library(fastDummies)
# maths_d<- dummy_cols(math,    remove_first_dummy = TRUE)
# write_sav(maths_d,"maths_d.sav")

###### Begin of data analyses ###
# Required packages
sapply(c("caret","arm", "ggplot2", "tidyverse", "readr"), require, character.only = TRUE)


# Load data
library(readr)
math_d <- read_csv("math_d.csv")
# View(math_d)
math<-as.data.frame(math_d)
head(math_d)

# Correlation plot of all variables
arm::corrplot((math_d), details=FALSE)
# Correlation of all variables with main outcome "final grade (=Column 40)
cor(math_d[-40], math_d[,40])
# round to 2 d.p.
sort(round(cor(math_d[-40], math_d[,40]), 2))

# Most of the variables were not strongly correlated with final grade.
# The most correlated variable is 'failures' with -0.36 of correlation coefficient.
# From this outcome, we can expect that a good prediction model cannot be modelled with these variables

# Perform a linear regression with all 39 predictors using lm 
# A rule of thumb says that you need at least 10 observations per variable 
# (or per parameter if we have categorical variables with more than two levels) 
# to get a stable model. (Learned in Intro to statistical modelling module)

dim(math_d)

# Dimension of the data did not follow the golden rule which should have more than 400 observations
# But, the actual observation is 395


# Nevertheless, fit the multiple regression using lm 
fit <- lm(math$final_grade ~ ., data=math)
summary(fit) # show results

# The strongest predictor is 'failures' and the next is the intercept of the model based on the p-value
# failures: 3.75e-07 and intercept: 0.001 (p-value)
# In terms of prediction modelling, R2 is 0.2756, which explains 27% of total variance of the data in apparent validation stage
# Despite including all predictors, the model shows poor performance in predicting final_grade.


# Apparent validity is not acceptable as a measure of prediction accuracy.
# In the next step we will perform internal validation and obtain an internally
# validated estimate of prediction accuracy of new, unseen cases from the 
# same population

# Perform internal validation with 10 fold cross-validation with 10 repeats
# train.control is used to choose 10 fold cross-validation with 10 repeats
set.seed(1234) 

train.control <- trainControl(method="repeatedcv", number=10, repeats = 10)

# Train the model
model <- train(final_grade~., data=math, trControl=train.control, method="lm")

# Summarize the results
print(model)

# Internal validation showed 0.14 of R2, which is huge decrease comparing to apparent validation.

# The data set consists of two schools. We can separate the two schools and use one as an 
# external validation data set
# We need to divide the data in a training and a test data set:                   
 train <- math_d[math_d$school_MS==0, -1]  # 349 students
 test <- math_d[math_d$school_MS==1 ,-1]   # 46 students
                             
dim(train)  #349 students and 39 variables
dim(test)   #46 students and 39 variables
# The sample sizes are now very small and we would expect more overfitting

# Please calculate apparent and internally validated r2:                             
# We fit our model to the training data set (school 1) using lm
fit <- lm(final_grade ~ ., data=train)
summary(fit)

# This is our new model
# R2 is 0.2962 (apparent validity)
# Adjusted R2: 0.2099


# We now do internal validation using 10 fold CV with 10 repeats 
## Define training control
set.seed(1234) 

train.control = trainControl(method="repeatedcv", number=10, repeats=10)

# Train the model
model <- train(data=train, final_grade~., method="lm", trControl=train.control)

# Summarize the results
print(model)

# R2: 0.1586 which is half of the apparent R2

# External validation: We choose our model derived form the training data (train) and predict the grades of
# the students in the other school (test)
# We saved our model in "fit" 
# The function "predict" needs the regression model and the name of the new data set (=test)
predicted<-predict(fit, newdata=test[,-39])  #-39 means we remove the grade, this is not necessary for this regression analyses
#postResample calculates the MSE and r2:
postResample(pred = predicted, obs = test$final_grade)
# Question: How much variance does our model explain in the external data set?
# What is your conclusion about the model quality?

# Only 2.6% of total variance was explained
# We can conclude that the model cannot explain the variance a lot in the different schools
# It means the model is very poor
