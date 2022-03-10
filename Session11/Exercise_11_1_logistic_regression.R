###### Session 11: Regularized Regression III ##########
###### Logistic regularized regression - Practical 1  ####

# Libraries needed
# install.packages("popbio")
library(popbio)
library(glmnet)
library(psych)
library(caret)
library(pROC)

## Lecture examples ######

## First, we'll create a fake dataset of 30 individuals of different ages:
#set seed to ensure reproducible results
set.seed(123)       # remove seed to get different simulations  
age=rnorm(30,70,7)  # generates 30 values, with mean of 70 (years) and s.d.=7
age=sort(age)       # sorts these values in ascending order. 
ad=c(0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,1,1,1,1,1,1,0,1,0,1,1,1,1,1,1) # assign 'developed althzheimer' to these 30 in a non-random way.

# saves dataframe with two columns: age & developed Altzheimer
dat=as.data.frame(cbind(age, ad))
dat # shows you what your dataset looks like.

# Plot the data
# Plot with age on x-axis and developed Altzheimer (0 or 1) on y-axis
plot(age,ad,xlab="Age",ylab="Developed Altzheimer", ) 

# Logistic regression model (in this case, generalized linear model 
# with logit link). see ?glm
 mylogit=glm(ad~age,family=binomial,data = dat) 
 summary(mylogit)
# CIs using standard errors
 confint.default(mylogit)
# Odds ratios and 95% CI
# exponentiate the coefficient means the odds ratio in logistic regression
 exp(coef(mylogit)) 
 exp(confint(mylogit))
# Nicer output
 exp(cbind(OR = coef(mylogit), confint(mylogit)))
 
# Plot the predicted probability of developing Altzhemier 
# Plot with age on x-axis and developed Altzheimer (0 or 1) on y-axis
 plot(age,ad,xlab="Age",ylab="Probability of developing Altzheimer") 
# Add a prediction curve based on your logistic regression model to your plot
# "curve" draws a curve based on prediction from logistic regression model
 curve(predict(mylogit,data.frame(age=x),type="resp"),add=TRUE)
# optional: this draws an the predicted probabilities for each case 
# based on a fit "mylogit" to glm model. pch= changes type of dots.
 points(age,fitted(mylogit),pch=20)
 
 ## If you have many data a histogram shows you the distribution of events better
 library(popbio)
 logi.hist.plot(age,ad,boxp=FALSE,type="hist",col="gray") # each box of the two histogramms represents the number of cases within a 5 year period
 
##### Regularized logistic regression #### 
##### Predcitng at risk of autism ########
 library(glmnet)
 library(psych)
 library(caret)
 
# Import the data
# Please note that this is not the original data file. 
 autism <- read.csv("/home/changhyun/King's College London/prediction_modelling/prediction_modelling_in_r/Session11/autism_summerschool.csv")
 View(autism)
# Group 1 = high risk group Group 0 = control group
 summary(autism)

# Descriptive statistics for each variable by group
# Group 1 = high risk group Group 0 = control group
 describeBy(autism,autism$group)

# Important: glmnet needs the predictor and outcome variables in separate data files
# x needs to be in matrix format and y needs to be defined as a factor (=categorical) variable!
# autism[ ,-1] means that all columns except the first one (group) will be  copied into x
 x<-as.matrix(autism[,-1])
# autism[ ,1] means that only that first column will be copied into y
# as.factor()  change the group variable into a factor variable. This will be saved in the vecotr y
 y<-as.factor(autism[,1])  # 1 = high risk 0 = control

# Regularized logistic regression (family = binomial calls logistic regression)
# Note alpha = 1 for lasso only (penalty down to alpha=0 ridge only)
# We fit the model and store the results in the object "glmmod"
# glmmod contains all information of the fitted model 
 glmmod<-glmnet(x,y,alpha=1,family='binomial')

# Visualize the coefficients by executing the plot function:
# Each curve corresponds to a variable: 
# The curve shows the path of its coefficient against log of lambda of the whole coefficient vector at as
# The axis on the top shows the number of nonzero coefficients at the current
# The y axis shows the regression coefficients of a variable at current 
 plot(glmmod, "lambda" )
# If we use the print function we will get a summary of glmnet at each step of ??
# including the number of included variables (Df) and the % explained DEviance (%Dev) 
 print(glmmod) 

# Select best lambda using 10-fold cross-validation based on sum of deviance residuals ####
# If we want  AUC as selection criteria we need to include <<"type.measure = "auc">>
# AUC needs larger smaple sizes that 36 
 set.seed(123) # remove if you want to get a different random sample of CV
 cv.glmmod<- cv.glmnet(x,y,family='binomial')
# Plot the object. It shows the cross-validation curve (and upper and lower standard devitaion) along the ?? sequence 
# On the y-axis is a measure of the model goodness (here: Binomial deviance: the samller the deviance the better the model)
# The two horizontal lines shows the lambda of minimum deviance and the lambda 1 SE larger than the minimal lambda (more regularized than minimum lambda)
  plot(cv.glmmod, xvar="lambda")
# To view the two lambdas type:
  cv.glmmod$lambda.min  # 0.03808529
  cv.glmmod$lambda.1se  # 0.1400922
# To see the regression coefficents at minimum or minimum + 1se lambda
 coef(cv.glmmod, s = "lambda.min")  # 12 coefficients left after regularisation
 coef(cv.glmmod, s = "lambda.1se")  # min lambda + 1SE penalise coefficients more so that only 4 coefficients are left
# Save the minimum and 1se below minimum lambda 
 best_lambda <- cv.glmmod$lambda.min
 best_lambda1se <- cv.glmmod$lambda.1se 
 best_lambda 


# We now predict the probability for a case to belong to the high risk autism group 
# using minimum lambda. 
  y_prob<-predict(cv.glmmod,type="response",newx=x, s = "lambda.min")
# using minimum lambda + 1SE:
#    y_prob<-predict(cv.glmmod,type="response",newx=x, s = "lambda.1se")
  
  
  # Or we can predict to which class a case belongs (high risk or control) 
# The default threshold for interpreting probabilities to class labels is 0.5,
  y_pred<-as.numeric(predict(cv.glmmod,type="class",newx=x, s = "lambda.min"))  # comment roc command needs predction to be numberic
# Plot a cross table of observed and predicted
  # Group 1 = high risk group Group 0 = control group
  table(y,y_pred)
  
  # Showed that
  # TP: 16
  # FP: 1
  # FN: 1
  # TN: 18

# Get more information about your model prediction quality 
# positive defines the treatment (or positive) group, here high risk is coded as 1 
# and will be used as "positive" group 

# Get sensitivity (true negatives) and specifity (true positives),
# Positive Predictive Value and Negative Predictive Value  
   confusionMatrix(as.factor(y_pred), y, positive="1")
# For more information, see help 
  # ?confusionMatrix

# Sensitivity: 0.9474
# Specificity: 0.9412

# AUC is a measure of discrimination (equivalent to the Concordance or C statistics)
# One way of interpreting AUC is as the probability that the model ranks a random positive example
# more highly than a random negative example.
# High risk is coded as 1 and low risk as 0
 roc_obj.train <- roc(y, (y_prob))
 roc_obj.train
# Almost perfect separation 0.997
# ROC curve
 plot(roc_obj.train)
     
      
# Important : These measures of prediction accuracy are over-optimistic because they are 
# they are estimated from the same data set we used for model selection. 
# These validity measures are called apparent validity!  
# To obtain internal validity, we need to hold out a sample before we select (using CV) and fit the model 
# see lecture Validity 
 

 ################# Performance on the unseen data ####################### 
 
 # Lets split the sample in training (2/3) and test data 1/3) sets
 library(caret)
 set.seed(690)
 
 ind<- createDataPartition(1:dim(autism)[1], p = 2/3, list = FALSE)
 train <- autism[ ind,]
 test  <- autism[-ind,]
 
 y_train = train$group
 #y_train = ifelse(y_train=="LowRisk",0,1)
 x_train = as.matrix(train[, 2:37])
 
 y_test = test$group
 #y_test = ifelse(y_test=="LowRisk",0,1)
 x_test = as.matrix(test[, 2:37])
 
 cv.autism = cv.glmnet(x_train,y_train,family='binomial', nfold = 3)
 plot(cv.autism, xvar="lambda")
 
 # To view the two lambdas type:
 cv.autism$lambda.min
 cv.autism$lambda.1se
 
 # To see the regression coefficents at minimum or minimum + 1se lambda
 coef(cv.autism, s = "lambda.min")
 coef(cv.autism, s = "lambda.1se")
 # Save the minimum and 1se below minimum lambda 
 best_lambda <- cv.autism$lambda.min
 best_lambda1se <- cv.autism$lambda.1se 
 best_lambda 
 
 #Apparent performance on the train set: 
 # Probabilities and predicted class 
 y_prob_train<-predict(cv.autism,type="response",newx=x_train, s = "lambda.min")
 y_pred_train<- as.numeric(predict(cv.autism,type="class",newx=x_train, s = "lambda.min"))  # comment roc command needs predction to be numberic
 
 
 # Plot a cross table of observed and predicted
 table(y_train,y_pred_train)
 confusionMatrix(as.factor(y_pred_train), as.factor(y_train), positive="1")
 
 roc_obj.train <- roc(y_train, y_prob_train)
 roc_obj.train # AUC-ROC 0.972
 # ROC curve
 plot(roc_obj.train)
 
 ###### Validating performance on the test set: 
 y_prob_test <-predict(cv.autism,type="response",newx=x_test, s = "lambda.min")
 y_pred_test <-as.numeric(predict(cv.autism,type="class",newx=x_test, s = "lambda.min"))  # comment roc command needs predction to be numberic
 table(y_test,y_pred_test)
 
 confusionMatrix(as.factor(y_pred_test), as.factor(y_test), positive="1")
 #accuracy is 0.417 on the test set, down from 0.917 on the train set 
 
 roc_obj.test <- roc(y_test, y_prob_test)
 roc_obj.test # AUC-ROC 0.639
 # ROC curve
 plot(roc_obj.test)
 
 
 ### Calibration alpha and beta, for test and train sets: 
 # train
 y_prob_test[y_prob_test==1]  <- 0.999999999 # choose column with probability to be a case = 1
 y_prob_test[y_prob_test==0]  <- 0.000000001
 logOdds_test<-log(y_prob_test/(1- y_prob_test))
 glm.coef.beta       <-  glm(y_test ~ logOdds_test,family=binomial)$coef  
 Beta_test <-  glm.coef.beta[2]
 glm.coef.alpha       <-  glm(y_test ~ offset(logOdds_test),family=binomial)$coef 
 Alpha_test  <-  glm.coef.alpha[1]
 paste("Calibration slope beta is ", round(Beta_test,3))
 paste("Calibration in the large alpha is ", round(Alpha_test,3))
 
 #test
 y_prob_train[y_prob_train==1]  <- 0.999999999 # choose column with probability to be a case = 1
 y_prob_train[y_prob_train==0]  <- 0.000000001
 logOdds_train<-log(y_prob_train/(1- y_prob_train))
 glm.coef.beta  <-  glm(y_train ~ logOdds_train,family=binomial)$coef  
 Beta_train <-  glm.coef.beta[2]
 glm.coef.alpha       <-  glm(y_train ~ offset(logOdds_train),family=binomial)$coef 
 Alpha_train  <-  glm.coef.alpha[1]
 paste("Calibration slope beta is ", round(Beta_train,3))
 paste("Calibration in the large alpha is ", round(Alpha_train,3))
 cbind(y_prob_train, y_train)
 
 
 
 
 
### # Repeated cross-validation ###### 
 
#  Optional: Repeating n-fold cross-validatoin usig loop ###### 
#  Selecting lambda based on a single run of 10-fold cross-validation is usually
#  not recommended. The procedure should be repeated 100 times (with different folds)
#  and the mean of each 100 minimum lambdas (or 100 minimum +1 SE lambdas) should be used 
#  This can be easily done with a loop or we use the caret package and its function "trainControl" and train ####
#  The package allows allows to use parallel computing: Large portions of code can run concurrently in different cores 
# and reduces the total time for the computation. To use parallel computong we need to load the package "doParalell" 

 
 library(caret)
 library(glmnet)
 library(doParallel)

 
# We need to use the combined data set with x and y: autism
# The outcome needs to be defined as a factor
autism$group<-as.factor(autism$group) 
levels(autism$group) <- c("LowRisk", "HighRisk")  # levels 0 and 1 aren't valid names in R and you need to label your two levels

# Set up number of cores for parallel computing
cl=makeCluster(4);registerDoParallel(cl)

# Set up training settings object

set.seed(123)
 trControl <- trainControl(method = "repeatedcv", # repeated CV 
                           repeats = 10,          # number of repeated CV
                           number = 6   ,         # Number of folds
                           summaryFunction = twoClassSummary,  #function to compute performance metrics across resamples.AUC for binary outcomes
                           classProbs = TRUE, 
                           savePredictions = "all",
                           allowParallel = TRUE,
                           selectionFunction = "best" ) # best - minimum lambda, oneSE for minimum lambda + 1 Se, Tolerancwe for minimum + 3%

 
 # Set up grid of parameters to test
 params = expand.grid(alpha=c(1),   # L1 & L2 mixing parameter
                      lambda=2^seq(1,-10, by=-0.1)) # regularization parameter
 
 

 # Run training over tuneGrid and select best model
 glmnet.obj <- train(group ~ .,             # model formula (. means all features)
                     data = autism,         # data.frame containing training set
                     method = "glmnet",     # model to use
                     metric ="ROC",         # Optimizes AUC, best with deviance for unbalanced outcomes 
                     family="binomial",     # logistic regression
                     trControl = trControl, # set training settings
                     tuneGrid = params)     # set grid of paramameters to test over, if not specified defualt gris is used (not always the best)
 
 
 stopCluster(cl) # Stop the use of cores!
 
 # Plot performance for different params
 plot(glmnet.obj, xTrans=log, xlab="log(lambda)")
 
 # Plot regularization paths for the best model
 plot(glmnet.obj$finalModel, xvar="lambda", label=T)

  # Summary of main results 
 print(glmnet.obj)

 # See the content of the object 
 summary(glmnet.obj)

 
 #glmnet.obj$results
 
 
 get_best_result = function(caret_fit) {
   best = which(rownames(caret_fit$results) == rownames(caret_fit$bestTune))
   best_result = caret_fit$results[best, ]
   rownames(best_result) = NULL
   best_result
 }
 
 get_best_result(glmnet.obj)
 
 best_alpha <-get_best_result(glmnet.obj)$alpha
 best_lambda <- get_best_result(glmnet.obj)$lambda
 
 # Model coefficients
 # The fitted coefficients at the optimal penalties can be obtained by  using the  coef command 
 coef(glmnet.obj$finalModel, glmnet.obj$bestTune$lambda)  # best model 
 
 #Variable importance of unstandardised coefficents of best model
 plot(varImp(glmnet.obj, scale = FALSE), top = 10, main = "glmnet- unstandardised coefficents")
 
 
 # Predict values for the training data set (apparent validity)

 predictions_prob<-predict(glmnet.obj,as.matrix(autism[,-1],best_lambda), type="prob")
 head(predictions_prob)
 # --> fist colum is probability to be Low risk, second group:P(Highrisk)
 predictions_class<-predict(glmnet.obj,as.matrix(autism[,-1],best_lambda), type="raw")
 head(predictions_class)
 
 # Model prediction performance using caret functions (not Metrics)
 # Get more information about your model prediction quality 
 # positive defines the treatment (or positive) group, here high risk is coded as 1 
 # and will be used as "positive" group 
 confusionMatrix(as.factor( predictions_class), autism[,1], positive="HighRisk")
 
 # AUC is a measure of discrimination (equivalent to the Concordance or C statistics)
 # One way of interpreting AUC is as the probability that the model ranks a random positive example more highly than a random negative example.
 roc_obj.train <- roc(autism[,1], as.numeric(predictions_prob$HighRisk))
 roc_obj.train
 # ROC curve
 plot(roc_obj.train)

 
 ### Optional Calibration alpha and beta, see lecture calibration
 ### High risk is in second column
 predictions_prob[predictions_prob==1]  <- 0.999999999 # choose column with proibability to be a case = 1
 predictions_prob[predictions_prob==0]  <- 0.000000001
 logOdds<-log(predictions_prob/(1- predictions_prob))
 glm.coef.beta       <-  glm(autism$group ~ logOdds[,2],family=binomial)$coef  
 Beta	 <-  glm.coef.beta[2]
 glm.coef.alpha       <-  glm(autism$group ~ offset(logOdds[,2]),family=binomial)$coef 
 Alpha  <-  glm.coef.alpha[1]
 paste("Calibration slope beta is ", round(Beta,3))
 paste("Calibration in the large alpha is ", round(Alpha,3))
 
 #