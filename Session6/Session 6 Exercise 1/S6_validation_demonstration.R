#### Session 6 Validation Demonstration ######
#### Internal validation using Caret #########

# We will use a data set about mental health, stress and socio-economic status:
# from: https://data.princeton.edu/wws509/sets/solset1 retrieved 3.2.22:
# Agresti and Finlay(1997) report data from a Florida study investigating 
# the relationship between mental health and several explanatory variables
# using a random sample of 40 subjects. 
# The outcome of interest is an index of mental impairment that 
# incorporates measures of anxiety and depression. 
# We will use two predictors: a life-events score that combines the number and severity
# of various stressful life events and SES, an index of socio-economic status.
# see https://topepo.github.io/caret/model-training-and-tuning.html#basic for more

library(foreign)
library(caret)

# Get data set af
af <- read.dta("http://data.princeton.edu/wws509/datasets/afMentalHealth.dta")
write.table(af, file = "af.csv", sep = ",", row.names = F) 
af <- read_csv("af.csv")
# Scatter plot
 pairs(af)

# Perform linear regression
m1 <- lm(mentalImpair ~ lifeEvents+ses, data = af)
summary(m1)
# Mental impairment shows a positive association with stressful life events (coefficient: 0.10)
# and is negatively associated with SES (coefficient: -0.10)
# Both predictors are statistically significant
# The model explains 33.92% of the variance of Mental impairment

# The function trainControl is used to specify the type of resampling:
# 10 folds are used and many repeats (100) to get a stable result 

  fitControl <- trainControl(## 10-fold CV
   method = "repeatedcv",
   number = 10,
   ## repeated ten times
   repeats = 100) 

# The function train is used to estimate model performance from our training set af
 
   set.seed(12345)
  lm_val  <- train(mentalImpair ~ lifeEvents+ses, data = af, 
                   method = "lm", 
                   trControl = fitControl)
                 
lm_val  
print(lm_val)
# 46.06% of the variance are explained   

# Optional: all 500 r2s are saved in"xxx$resample$Rsquared":
str(lm_val)  # you can see what's stored in the object
lm_val$resample$Rsquared  # all 500 r2
hist(lm_val$resample$Rsquared)    
# It shows that our individual r2 vary a lot from close to 0% up to 100%,
# which suggests that our model is not stable (because of the small sample size)

# Run a simple cross-validation
fitControl <- trainControl(## 10-fold CV
  method = "CV",
  number = 10)
set.seed(12345)

lm_val  <- train(mentalImpair ~ lifeEvents+ses, data = af, 
                 method = "lm", 
                 trControl = fitControl,
)

lm_val  



