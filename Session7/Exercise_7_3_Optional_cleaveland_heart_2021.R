library(caret)
library(glmnet)
# Predciting Coronary Artery Disease in a population referred for angiography,  
#
# Dataset:
# http://archive.ics.uci.edu/ml/datasets/Heart+Disease
# The heart disease data are available at UCI The description of the database can be found here.
# https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data
# The description of the data can be downloaded from
# https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/heart-disease.names
# The "goal" field refers to the presence or degree of a heart disease in the patient. 
# It is integer valued from 0 (no presence) to 4. 
# It is possible to simply attempting to distinguish presence (values 1,2,3,4) from absence (value 0).
# However, we will try to develop a model which predicts the degree of Coronary Artery Disease using the score as a continous outcome
# in  a population referred for angiography

# Question: Do you see any potential problems with the data set?

# Answer: We develop the model in a population referred for angiography. We expect a good performance in such a population.
# Clinicians select patients for angiography because their symptoms and test results merit this procedure and we will have
# a large prevelance in our data set and is not representative for the general population. This bias is present in many 
# diagnostic or other predictive studies and hard to avoid. Models will usually not perform well outside this seelctive population.
# Many papers and reports ignores this....

# The data consists of 143 potentail predcitors and one clinical outcome (num)
# Variable name	Short desciption	
# age      Age of patient	
# thalach	maximum heart rate achieved
# sex	Sex, 1 for male	
# exang	exercise induced angina (1 yes)
# cp	chest pain	
# oldpeak	ST depression induc. ex.
# trestbps	resting blood pressure
# slope	slope of peak exercise ST
# chol	serum cholesterol
# ca	number of major vessel
# fbs	fasting blood sugar larger 120mg/dl (1 true)
# thal thalassemia score (1-10: 3 normal; 6 fixed defect; 7 reversable defect)
# restecg	resting electroc. result (1 anomality)	
# num	clinical outcome: diagnosis of heart disease (angiographic disease status): degree ranging from 0 to 4 


# Load data set and add labels
heart.data <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",header=FALSE,sep=",",na.strings = '?')
names(heart.data) <- c( "age", "sex", "cp", "trestbps", "chol","fbs", "restecg","thalach","exang", "oldpeak","slope", "ca", "thal", "num")

head(heart.data,3)
dim(heart.data)
# We need to define some variables as factor and then dummy code them
heart.data$sex <- factor(heart.data$sex)
levels(heart.data$sex) <- c("female", "male")
heart.data$cp <- factor(heart.data$cp)
levels(heart.data$cp) <- c("typical","atypical","non-anginal","asymptomatic")
heart.data$fbs <- factor(heart.data$fbs)
levels(heart.data$fbs) <- c("false", "true")
heart.data$restecg <- factor(heart.data$restecg)
levels(heart.data$restecg) <- c("normal","stt","hypertrophy")
heart.data$exang <- factor(heart.data$exang)
levels(heart.data$exang) <- c("no","yes")
heart.data$slope <- factor(heart.data$slope)
levels(heart.data$slope) <- c("upsloping","flat","downsloping")
heart.data$ca <- factor(heart.data$ca) # not doing level conversion because its not necessary
heart.data$thal <- factor(heart.data$thal)
levels(heart.data$thal) <- c("normal","fixed","reversable")
# heart.data$num <- factor(heart.data$num) # not doing level conversion because its not necessary



# Check for missing values - only 6 so just remove them.
s = sum(is.na(heart.data))
heart.data <- na.omit(heart.data)
#str(heart.data)



#### Do a (simple) lasso regression analyses to identify th ebest set of predcitors ####
library(glmnet)
library(caret)

# dummicode all categorical variables (factor variables) to n-1 dummy variables
# using the function dummyVar fro the package caret
dummy <- dummyVars(" ~ .", data = heart.data, fullRank=TRUE)
data <- data.frame(predict(dummy, newdata = heart.data,))
dim(data)
x<-as.matrix(data[,-21]) 
y<-as.matrix(data[,21])
##### Model selection using lasso regression ####### 
# Fit a penalised regression mode
# Gaussian means we perform a linear regularized regression with continous outcome
# alpha = 1 means that we perform lasso regression 
# for a large range of lambdas (100 is default) 
fit1<-glmnet(x, y, family = "gaussian", alpha = 1)

# Plot the penalised regression coefficients against the penalty values. 
plot(fit1, xvar="lambda", label=TRUE)    # plots the coefficents against (log)lamda

# Use 10-fold cross-validation to find the optimal penalty
# The cv.glmnet command can be used to find optimal penalty  by cross-validation

set.seed(111)           # allows us to use the same random vlaue to get the same results, alpha =1 is lasso
cv10<-cv.glmnet(x,y, alpha=1, nfold=10)  # nfold=10 --> 10 fold cv is used to find optimal lambda

# Identify the best lambda: Plot log lambda against MSE of unseen cases 
plot(cv10)  # Vertical lines are drawn at minimum MSE and (min + 1SE) MSE respectively  

# The  optimal value of lambda can be obtained by extracting the lambda.min component of the fitted cv.glmnet object
cv10$lambda.min

# The fitted coefficients at the optimal penalty can be obtained by  using the  coef command 
coef(cv10, s="lambda.min") 

# How many variables remain in the model?


# Addtional information: Get the MSE for the best lambda model. This is the MSE of  
MSE <- min(cv10$cvm)  # $cvm inlcudes the MSE of all 100 lambdas , min() selects the snmallest
# This is the MSE of the predicted test data 
# Alternative to get MSE for minimum lambda and to get MSE for minimum + 1SE lambda
MSE_min<- cv10$cvm[which(cv10$lambda == cv10$lambda.min)]
MSE_min

# Explained variance of unseen cases
paste("R2 is: ",  1-(MSE_min/var(y)))


# Predcit values for the training data set (apparent validity)
predictions<-predict(cv10,newx=x,s="lambda.min",type="response")
#Explained variance within the sample (appatent validation)
# Model prediction performance using caret functions
paste("The apparent RMSE of model  is: ", RMSE(predictions, y) )
paste("The apparent R2 is: ",  R2(predictions, y))
paste("The apparent MSE of model  is: ", (RMSE(predictions, y)^2))

MSE<-(RMSE(predictions, y)^2)

#  Calibration
MSE.test    <- (RMSE(predictions, y)^2)
lm.coef      <-  lm(y ~ predictions)$coef  
Alpha  <-  lm.coef[1]   
Beta	 <-  lm.coef[2]
paste("THe apparent callibration slope b is ", round(Beta,3))
paste("THe apparent calibration-in-the-large alpha is ", round(Alpha,3))





# Alternative to get MSE for minimum lambda and to get MSE for minimum + 1 SE lambda
# Alternative best lambda (lambda.1SE)
# This lambda penalizes slightly more than the minimum lambda
# It selects a more parsimonious model with almost the same MSE
# It is often better if the changes of lambda around the minimum are very small
# How many variables are selected 
coef(cv10, s="lambda.1se")

# Addtional information: Get MSE for minimum + 1SE lambda
MSE_1SE<- cv10$cvm[which(cv10$lambda == cv10$lambda.1se)]
MSE_1SE
# explained variance 
paste("R2 is: ",  1-(MSE_1SE/var(y)))  # if negative round to 0


### End of practical ########

##### Optional topics #####
# 1 Repeated cross-validation 

# If you repeat the cv.glmnet function you will get always slightly different results, esp. if 
# the sample size is small (relative to th enumbe rof predictors). It is therefore recommended 
# to rerun the cross-valdiation analyses several times and to take the average MSE (or other accuracy measures)
# of the overall best lambda. 
# Repeated cross-validation provides a better estimate of the test-set error!
# see e.g. http://rstudio-pubs-static.s3.amazonaws.com/251240_12a8ecea8e144fada41120ddcf52b116.html
# We will use the package caret and the functions "trainControl" and "train"

library(caret)
library(glmnet)
library("doParallel")  #for paralell computing

### Simulation of data

# use the ssame data as above:

data <- as.data.frame(heart.data)  # caret needs outcome and predcitor in one object. THe object needs to be a dataframe
##Important dependent variable is called num here not y as in other examples!)

### Set up training settings object
trControl <- trainControl(method = "repeatedcv", 
                          number = 10   ,# Number of folds
                          repeats =  5 , # Number of iterations for repated CV. increase until results stabilize
                          savePredictions = "final",
                          selectionFunction = "best")  # best - minimum lambda, oneSE for minimum lambda + 1 Se, Tolerancwe for minimum + 3%


### Set up grid of parameters (lambdas) to test.
# It is better to provide glmnet a user-defined grid of lambdas! 
# We have to set alpha to either 0 (Ridge) or 1 (Lasso)
# Later we will see that we can also have alphas ranging between 0 and 1 (Elastic net)

params = expand.grid(alpha=c(1),   # L1 & L2 mixing parameter
                     lambda=2^seq(1,-10, by=-0.1)) # regularization parameter lambda
params  # lambda ranges from 0.001 to 2, this often works well. You can have smaller units by changing b= -0.3 to by = -0.1

### Run training over tuneGrid and select best model
cl=makeCluster(4);registerDoParallel(cl)   # using more than one core

glmnet.obj <- train(num ~ .,                 # model formula (. means all features)   <---
                    data = data,            # data.frame containing training set
                    method = "glmnet",     # model to use
                    trControl = trControl, # set training settings
                    tuneGrid = params)     # set grid of params to test over, if not specified defualt gris is used (not always the best)

stopCluster(cl)  # stop using cores

# Plot performance for different params
plot(glmnet.obj, xTrans=log, xlab="log(lambda)")

# Plot regularization paths for the best model
plot(glmnet.obj$finalModel, xvar="lambda", label=T)

print(glmnet.obj)
#str(glmnet.obj)

glmnet.obj$results

# This function automatically selects the best model (either minimum lambda or lamda + 1 Se)
get_best_result = function(caret_fit) {
  best = which(rownames(caret_fit$results) == rownames(caret_fit$bestTune))
  best_result = caret_fit$results[best, ]
  rownames(best_result) = NULL
  best_result
}

# Get alpha, lambda, MSE, r2, MAE (Mean absolute error) and the SD of each (RMSESD, RsquaredSD, MAESD) for
# of the best tmodel (=on average best predcition of unseen cases)

get_best_result(glmnet.obj)
# MSE = RMSE^2
print(paste("MSE of best model is ", get_best_result(glmnet.obj)[[3]]^2))

# Save best alpha and lambda
best_alpha <-get_best_result(glmnet.obj)$alpha
best_lambda <- get_best_result(glmnet.obj)$lambda

# Model coefficients
# The fitted coefficients at the optimal penalties can be obtained by  using the  coef command 
coefs<-coef(glmnet.obj$finalModel, glmnet.obj$bestTune$lambda)  # best model = best alpha
coefs

# Predcit values for the training data set (apparent validity)
predictions<-predict(glmnet.obj$finalModel,newx=x,s=glmnet.obj$bestTune$lambda, alpha=best_alpha,type="response")
#Explained variance within the sample (appatent validation)
# Model prediction performance using caret functions
paste("The apparent RMSE of model  is: ", RMSE(predictions, y) )
paste("The apparent MSE of model  is: ", (RMSE(predictions, y)^2))
paste("The apparent R2 is: ",  R2(predictions, y))

#  Calibration
MSE.test    <- (RMSE(predictions, y)^2)
lm.coef      <-  lm(y ~ predictions)$coef  
Alpha  <-  lm.coef[1]   
Beta	 <-  lm.coef[2]
paste("THe apparent callibration slope b is ", round(Beta,3))
paste("THe apparent calibration-in-the-large alpha is ", round(Alpha,3))




# 2. Variable importance: glmnet presents the coefficents in a backtransformed format on the original scale
# To get the coefficient in a space that lets you directly compare their importance, you have to standardize them. 
# standardised Beta for variable xj = Bj *(SD(Xj)/SD(Y)) 
# dummy coded categorical variables can be standardised as well, intepretation of coefficent is difficult, use it only for variable importance 


sd_x <- as.matrix(apply(x, 2, sd))
sd_y<-sd(y)
coefs <- as.matrix(coef(cv10, s = "lambda.min"))
coefs<-coefs[-1,]
std_coefs <- coefs * as.matrix(sd_x)*sd_y
std_coefs

# Simple Dotplot
dotchart(std_coefs,labels=row.names(std_coefs),cex=.7,
         main="Variable importance plot",
         xlab="Standardised coefficents")

# Sort by size of regression coefficents
ordered <- std_coefs[order(std_coefs)] # sort by regression coefficents

dotchart(ordered,labels=(order(std_coefs)),cex=.7,
         main="Variable importance plot",
         xlab="Standardised coefficents")

# Conclusion: Except cholesterin all variables remain in the model.Chest pain,thalach	maximum heart rate achieved, number of major vessels
# and thalassemia score are the most important predictors.   
