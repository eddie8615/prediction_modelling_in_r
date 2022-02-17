#############################################
#
# Introduction to Prediction Modelling
#
# Exercise & Practical: Cox Regression and SUrvival Analysis
#
############################################
# Practical 2 SURVIVAL ANALYSIS ###########################################################################################
## Code by Dr DEBORA Agbedjiro and D Shamsutdinova #################################################################################################

#install.packages("pastecs")# for stat.desc()
#install.packages("numbers")# for calibration plot function
#install.packages("varhandle")# for function unfactor()
#install.packages("survminer")# for Kaplan-Meier plot ggSurvplot()
#install.packages("timeROC")# for time-dependent AUC
#install.packages("ggplot2")# for package 'survminer' to work
#install.packages("hdnom") # for survcurve and calibration plots to work

library(glmnet)
library(caret)
library(survival)
library(survminer)
library(timeROC)
library(parallel)# for mclapply(), parallelized version of lapply()
library(c060)   #for function peperr() for prediction error for survival models and predictProb() #to extract predicted survival probabilities from "coxnet" and "glmnet" objects
library(devtools)
#library(foreign)# for nomogram
library(rms)#for nomogram
library(ggplot2)
#library(pROC)
library(readstata13)# to load .dta datasets
library(varhandle) # to unfactor a categorical variable 
library(survminer)
library(psych)
library(pastecs)
library(Hmisc)
#library(numbers)# for calibration plot function, to find divisors
require(DMwR)

options("scipen"=100, "digits"=4) #this is to print values as 0.00012 rather than 1.2e-4
#setwd("D:/PredictionModellingCourse2020")  #set to your path here

# Upload the data with missing values
data<-read.dta13("DementiaData.dta")
dim(data)
head(data)

#Summarise the data
summary(data)
describe(data)
#Percentage of missingness and complete cases
sum(is.na(data))/(dim(data)[1]*dim(data)[2]) 
sum(complete.cases(data))/dim(data)[1] 

# #Describe the time to event by outcome
describeBy(data$survtimed, data$dem_all)
sum(is.na(data$dem_all)) #no missing data in outcome variable

#Cases in consented sample
table(data$dem_all)
prop.table(table(data$dem_all))

#Review the survival time
hist(data$survtimed)
sum(is.na(data$survtimed)) #no missing data in time-to-event variable

factors = names(data) #see the list of available risk factors
factors 

#####---------------------- Kaplan-Meier Plot -------------------------------#######
#turning the outcome into a numeric variable:
levels(data$dem_all)<-c(0,1)
data$dem_all<-unfactor(data$dem_all)
KM <- survfit(Surv(survtimed, dem_all)~1, data = data)
summary(KM)
#plotting:
ggsurvplot(survfit(Surv(survtimed, dem_all)~1, data = data), 
           data=data, risk.table = TRUE, break.time.by = 2, ylim=c(0.85,1))# stretch the plot to see it properly
?ggsurvplot
describeBy(data$dem_all, data$PA_levels)

#Check which factors have discriminating power by plotting groups K-M curve 
#1) by gender (doesn't seem to discriminate well)
S <- Surv(data$survtimed, data$dem_all)
ggsurvplot(survfit(S~male_gender, data = data),data=data, risk.table = TRUE, break.time.by = 2, ylim=c(0.85,1))

# logrank test to check if the two groups are statistically different 
logrank_gender <- survdiff(S ~ data$male_gender)
logrank_gender

#2) by physical activity PA_levels (seems to work very well)
ggsurvplot(survfit(S~PA_levels, data = data),data=data, risk.table = TRUE, break.time.by = 2, ylim=c(0.85,1))

logrank_pa <- survdiff(S ~ data$PA_levels)
logrank_pa

###################################### Exercise 1  Plot K-M estimator ######################################
# 1a) plot K-M estimator for education ('college') and wealth ('wealth_grps') groups

ggsurvplot(survfit(S~college, data=data), data=data, risk.table = TRUE, break.time.by=2, ylim=c(0.70,1))

ggsurvplot(survfit(S~wealth_grps, data=data), data=data, risk.table = TRUE, break.time.by=2, ylim=c(0.7,1))



# 1b) perform logrank test to check if they  define statistically different groups in terms of survival #

survdiff(S~data$college) # p-value = 0.0005 which means each stratification has statistically different

survdiff(S~data$wealth_grps) # p-value = 0.00000001 more different than college.



##################################### end of exercise 1 ###############################

###########---------------------- Fitting a simple Cox Model --------------------######
#Let's fit Cox model with regression on PA_levels and age (which is usually a good predictor too)
data$gender = ifelse(data$male_gender =="1. yes", 1,0)

# bring age to be centered around 0, so baseline survival function 
# corresponds to a person of mean age by construction
data$age_ = data$age-mean(data$age)

#fit simple Cox regression with coxph function from survival library:
cox_basic <- coxph(S ~ PA_levels + age_ + male_gender, data = data)
summary(cox_basic)  #see coefficients, significance and other statistics 
cox.zph(cox_basic) #check if PH assumption holds

#Let's calculate Survival function estimated by the Cox model
#1) First we need to calculate baseline cumulative hazards,
#2) then we use the fact that survival function = exp(-H0(t)), H0 - cumulative hazard
#3) survival function for person with low level of PA is 
#   exp(-H0(t)*exp(beta for PA_levels_low))= exp(-H0(t))^exp(beta)
#1)
bh_0=basehaz(cox_basic)
#2)
plot(bh_0[,2],exp(-bh_0[,1]),main="Baseline Survival function",xlab="Time",ylab="S0(t)", col = 'blue', ylim = c(0.8,1))
#3) individual survival function is exp(-H(t))^(exp(beta*x) = exp(-H(t))^(lp))
b_base = 0.
b_medium = cox_basic$coefficients[1]*1.
b_low = cox_basic$coefficients[2]*1.
b_age = cox_basic$coefficients[3]*1.
b_gender = cox_basic$coefficients[4]*1.
#create dummies for PA levels, check the names using  names(table(data$PA_levels))[3]
data$PA_m = ifelse(data$PA_levels =="1. moderate", 1,0)
data$PA_l = ifelse(data$PA_levels =="2. sederary/low", 1,0)
# Survival function for sub-groups by physical activity
plot(bh[,2],exp(-bh[,1])^(exp(b_base)),main="Survival Function by physical activity",xlab="Time",ylab="S0(t)", col = 'blue', ylim = c(0.8,1))
lines(bh[,2],exp(-bh[,1])^(exp(b_medium)), col = 'orange', ylim = c(0.8,1))
lines(bh[,2],exp(-bh[,1])^(exp(b_low)), col = 'red', ylim = c(0.8,1))
lines(survfit(S ~ data$PA_levels), col = c('black', 'black', 'black'), ylim = c(0.8,1))
legend("bottomleft", legend = c('Cox estimate (blue/orange/red)', 'K-M estimate (black)'), col = c('red', 'black'), pch='_')
#end

#individual prognosis, or linear predictor is beta*x:
lps = data$age_ * b_age + data$PA_m * b_medium + data$PA_l * b_low + b_gender * data$gender  # + ifelse(male_gender = '1. male', 1, 0)

# calculating c-index with survCondordance index, should give the same as in Cox model summary
concordance_cox_basic <-survConcordance(S ~ lps)$concordance #0.812 - same as we saw in Cox summary above
concordance_cox_basic

# for reference: this is how one can calculate probability of event at t=10:
#probability to survive is 1-exp(-H(t))^exp(beta*x), 
#time=10 is in 322 row, bh[322,] so this is what we use for H0(10) 
head(bh)
t_10 = 10 
i_10 = match(1, round(bh$time,1) == 10, nomatch = -100)
bh[i_10,]
event_prob_10 = 1-exp(-bh[i_10,1])^exp(lps)

#best_cut_off is a function to calculate Youden point -  threshold that separates cases and non-cases such 
# that sensitivity + (1-sensitivity) is at maximum
best_cut_off = function(survtime, outcome, marker, times) {
  max_sum =0 
  best_cut_off = min(marker)
  for (c in seq(min(marker), max(marker), length.out = 50)) {
    SeSpPPVNPV_c<-SeSpPPVNPV(cutpoint= c, T= survtime, delta=outcome, marker=marker, cause=1,weighting = "cox", times=times, iid = FALSE)
    current_sum = sum(SeSpPPVNPV_c$TP+(1-SeSpPPVNPV_c$FP))
    #print (current_sum)  
    if (current_sum > max_sum) {
      best_cut_off=c
      max_sum = current_sum
    }
  }
  return(best_cut_off)
}
confusion_matrix_from_SeSp <- function(SeSpPPVNPV_2){
  cases = SeSpPPVNPV_2$Stats[2]
  sens = SeSpPPVNPV_2$TP[2]
  ppv = SeSpPPVNPV_2$PPV[2]
  spec = 1-SeSpPPVNPV_2$FP[2]
  tp = cases*sens
  fp = cases*(1-sens)
  controls = (1/ppv-1)*cases*sens/(1-spec)
  tn = controls*spec
  fn= controls *(1-spec)
  cat(c('       ','predicted 1','predicted 0'))
  cat(c('\nactual 1 (tp,fp):',round(tp,0),'',round(fp,0)))
  cat(c('\nactual 0 (fn,tn):',round(fn,0),'',round(tn,0)))
}

# Calculating discrimination statistics at the best cut_off point based on the linear predictors
times = c(5, 10)
cut_off = best_cut_off(data$survtimed,data$dem_all, lps, times)
SeSpPPVNPV_0<-SeSpPPVNPV(cutpoint = cut_off, T=data$survtimed, delta=data$dem_all, marker=lps, cause=1,weighting = "cox", times= times, iid = FALSE)
SeSpPPVNPV_0
confusion_matrix_from_SeSp(SeSpPPVNPV_0)

#--end of building an exploratory Cox model -------------------

################################ Exercise 2 #########################################
# a) build a Cox model using age_ only, what c-index does this model have? 

model <- coxph(S~age_, data=data)
summary(model)

# b) Now use PA_levels, male_gender, wealth_grps and college. 
# What variables are statistically significant? What is the resulting c-index? What is added-value of the additional variables?

model_1 <- coxph(S~age_+PA_levels+male_gender+wealth_grps+college, data=data)
summary(model_1)

# Significant variables are PA_levels.moderate and sederary/low, wealth_grops.low, age and gender
# C-index: 0.815

# Analysis
# Low PA_levels (Physical Activity) shows 2.5 times higher risk ratio to have dementia than people whose PA_level is high.
# Also, Male is a bit more chance to get dementia compared to female about 1.3times
# The most significant predictor 'age' shows that people have more chance to get dementia when they are getting old.

### end of exercise 2 ####

################################# Cox Lasso using glmnet #############################
X_glm<-model.matrix(S~.,data[, c("age_", "PA_m", "PA_l", "gender")]) 
cox_lasso_cv<- cv.glmnet(X_glm, S,alpha=1, family='cox',nfolds = 5,type.measure = "C")
plot(cox_lasso_cv)  # we see that adding more parameters does not add much to the c-index
summary(cox_lasso_cv)

#Remember that h(t|X)=h_baseline(t) * exp(b*X), this function calculates relative risks, exp(bx)
linpredictions<-predict(cox_lasso_cv,newx=X_glm,s="lambda.min",type="link") #this is beta * x, linear predictor
predictions_cv<-predict(cox_lasso_cv,newx=X_glm,s="lambda.min",type="response") #exp (linear predictions)
exp(linpredictions[1]) == predictions_cv[1]

cox_lasso_best<-glmnet(X_glm, S, family="cox", alpha=1, lambda=cox_lasso_cv$lambda.min) #re-fit with best lambda
cox_lasso_best$beta #this is how one can get coefficients
cox_basic$coefficients # compare to what we had in non-lasso version (very similar)

#use SeSpPPVNPV function to calculate performance at 5 and 10y times 
times = c(5,10)
SeSpPPVNPV_2<-SeSpPPVNPV(cutpoint= best_cut_off(data$survtimed, data$dem_all, linpredictions, times), T=data$survtimed, delta=data$dem_all, marker=linpredictions, cause=1,weighting = "cox", times=times, iid = FALSE)
SeSpPPVNPV_2
confusion_matrix_from_SeSp(SeSpPPVNPV_2)

concordance_cox_lasso <-survConcordance(S ~ linpredictions)$concordance
concordance_cox_lasso

########################## Validation of the simple COX Lasso model #####################
#splitting data into training/testing data using the trainIndex 
set.seed(1234) #fix the way the split works
trainIndex <- createDataPartition(data$dem_all,p=0.75,list=FALSE)

data_train <- data[trainIndex,] #training data (75% of data)
data_test <- data[-trainIndex,] #testing data (25% of data)

# define outcome and regression parameters
S_train <- Surv(data_train$survtimed, data_train$dem_all)
S_test <- Surv(data_test$survtimed, data_test$dem_all)
X_glm_train<-model.matrix(S_train~.,data_train[, c("age_", "PA_m", "PA_l")])
X_glm_test<-model.matrix(S_test~.,data_test[, c("age_", "PA_m", "PA_l")])
# fitting Cox Lasso to the data in the train set
cox_train <- cv.glmnet(X_glm_train, S_train,alpha=1, family='cox', nfolds = 3, type.measure = "C")
plot(cox_train)
summary(cox_train)
cox_train$lambda.min
coef(cox_train, s = "lambda.min") #check coefficients
coef(cox_train, s = "lambda.1se") #! only age is here

#re-train final model on train data
cox_train_final <-glmnet(X_glm_train, S_train , family="cox", alpha=1, lambda=cox_train$lambda.min)

# calculate linear predictor
linpredictions_train<-predict(cox_train_final,newx=X_glm_train,s="lambda.min",type="link")
linpredictions_test<-predict(cox_train_final,newx=X_glm_test,s="lambda.min",type="link")
times = c(5,10)

#calculate apparent performance statistics
cut_off_train = best_cut_off(data_train$survtimed,data_train$dem_all, linpredictions_train, times)
SeSpPPVNPV_train<-SeSpPPVNPV(cutpoint= cut_off_train, T=data_train$survtimed, delta=data_train$dem_all, marker=linpredictions_train, cause=1,weighting = "cox", times=times, iid = FALSE)
concordance_cox_train <-survConcordance(S_train ~ linpredictions_train)$concordance #0.8102 - same as we saw in Cox summary above
SeSpPPVNPV_train
concordance_cox_train

#validation results
######################################## Exercise 3 ########################################
#calculate performance statistics on test set

SeSpPPVNPV_test<-SeSpPPVNPV(cutpoint= cut_off_train, T=data_test$survtimed, delta=data_test$dem_all, marker=linpredictions_test, cause=1,weighting = "cox", times=times, iid = FALSE)
concordance_cox_test <-survConcordance(S_test ~ linpredictions_test)$concordance #0.8102 - same as we saw in Cox summary above
concordance_cox_test
SeSpPPVNPV_test

################################### end of exercise 3 ######################################
-----------------------
  
########### CALIBRATION PLOT ### CODE by Dr DEBORA Agbedjiro################################

calibration.plot.survival<-function(Predicted.prob,External.predicted.prob=NULL,y,y_ext,time,time_ext,time.point,unit,las=1, xlab = "Predicted survival probability", ylab = "Observed Fraction Surviving" ,
                                    minProb,cex=0.7,d0lab="0", d1lab="1",dist.label ,
                                    dist.label2){
  
  plot(0.5, 0.5, xlim = c(minProb,1), ylim = c(minProb,1), type = "n", xlab = xlab,main = paste("Calibration plot at", time.point,unit),
       ylab = ylab, las=las)
  abline(0, 1, lty = 2)
  lt <- 2
  leg <- "Ideal"
  marks <- -1
  
  q <- cut2(Predicted.prob, g=ifelse(ceiling(length(Predicted.prob)/50)<5,
                                     ceiling(length(Predicted.prob)/30),10),
            levels.mean = T, digits = 7)
  
  means <- as.single(levels(q))
  # KM Estimate  of survival probability by quantile of risk (fraction surviving):
  #(Number of subjects without the event at the start of the period-
  #Number of subjects with the event at 10 years (end of the period))/
  #Number of subjects without the event at the start of the period
  y.time.point<-y
  y.time.point[time>time.point]<-0
  prop <- tapply(y.time.point, q, function(x) ((length(x)-sum(x))/length(x)))
  
  points(means, prop, pch = 2)#, cex=0.75
  lt <- c(lt, 0)
  leg <- c(leg, "Grouped observations")
  marks <- c(marks, 2)
  
  # #18.11.02: CI triangles			
  # ng	<-tapply(y, q, length)
  # og	<-tapply(y, q, sum)
  # ob	<-og/ng
  # se.ob	<-sqrt(ob*(1-ob)/ng)
  # g		<- length(as.single(levels(q)))
  # 
  # for (i in 1:g) lines(c(means[i], means[i]), c(prop[i],min(1,prop[i]+1.96*se.ob[i])), type="l")
  # for (i in 1:g) lines(c(means[i], means[i]), c(prop[i],max(0,prop[i]-1.96*se.ob[i])), type="l")
  # 
  
  #aproximate uncalibrated line
  #lines(lowess(Predicted.prob,as.numeric(as.character(q)),iter=0),lty=3)
  #connecting points
  lines(means, prop,lty=3)
  lt <- c(lt, 3)
  marks <- c(marks, -1)
  leg <- c(leg, "Uncalibrated")
  
  # External validation curve ####
  if(!is.null(External.predicted.prob)){
    q_ext <- cut2(External.predicted.prob, g=ifelse(ceiling(length(External.predicted.prob)/50)<5,
                                                    ceiling(length(External.predicted.prob)/30),10),
                  levels.mean = T, digits = 7)
    
    means_ext <- as.single(levels(q_ext))
    y_ext.time.point<-y_ext
    y_ext.time.point[time_ext>time.point]<-0
    prop_ext <- tapply(y_ext.time.point, q_ext, function(x) ((length(x)-sum(x))/length(x))) #not needed, the order is the same
    lines(means_ext, prop_ext,lty=1,lwd=1.5)
    lt <- c(lt, 1)
    marks <- c(marks, -1)
    leg <- c(leg, "External validation")
    
    # External validation triangles
    points(means_ext, prop_ext, pch = 2)#, cex=0.75
  }
  
  #legend
  lp <-c(minProb,1)
  lp <- list(x = lp[1], y = lp[2])
  legend(lp, leg, lty = lt, pch = marks, bty = "n",cex=cex)
  
  x <- Predicted.prob
  bins <- seq(max(0,minProb), 1, length = 101) #it was max(min(ylim),0) and min(1,max(xlim))
  x <- x[x >= 0 & x <= 1]
  #08.04.01,yvon: distribution of predicted survival prob according to outcome at time.point
  f0	<-table(cut(x[y.time.point==0],bins)) #frequencies of survival probability intervals in the sample for people without the event
  f1	<-table(cut(x[y.time.point==1],bins)) #frequencies of survival probability intervals in the sample for people with the event
  j0	<-f0 > 0
  j1	<-f1 > 0
  bins0 <-(bins[-101])[j0]
  bins1 <-(bins[-101])[j1]
  f0	<-f0[j0]
  f1	<-f1[j1]
  maxf <-max(f0,f1)
  f0	<-(0.1*f0)/maxf
  f1	<-(0.1*f1)/maxf
  line.bins<-minProb+0.03
  length.seg<-1-minProb
  segments(bins1,line.bins,bins1,length.seg*f1+line.bins)
  segments(bins0,line.bins,bins0,length.seg*-f0+line.bins)
  lines(c(min(bins0,bins1)-0.01,max(bins0,bins1)+0.01),c(line.bins,line.bins))
  text(max(bins0,bins1)+dist.label,line.bins+dist.label2,d1lab,cex=cex)
  text(max(bins0,bins1)+dist.label,line.bins-dist.label2,d0lab,cex=cex)
}

###### calibration plot for fitted model #### 

#install.packages("hdnom")
#library(help = "hdnom")

#Calculating survival function at time 5 and 10 ('times') for all participants in train and test
bh_train = hdnom:::glmnet_survcurve(cox_train_final, S_train[,1], S_train[,2], X_glm_train, survtime = times)
head(bh_train$p[,2])

bh_test = hdnom:::glmnet_survcurve(cox_train_final, S_test[,1], S_test[,2], X_glm_test, survtime = times)
head(bh_test$p[,2])
length(bh_test$p[,2])

#train set
calibration.plot.survival(Predicted.prob= bh_train$p[,2],External.predicted.prob=NULL,
                          y= S_train[,2],y_ext=NULL,time=S_train[,1],
                          time_ext=NULL,time.point=10,unit="years",minProb=min(bh_train$p[,2]),dist.label=0.02,
                          dist.label2=0.02)
#test set
calibration.plot.survival(Predicted.prob= bh_test$p[,2],External.predicted.prob=NULL,
                          y= S_test[,2],y_ext=NULL,time=S_test[,1],
                          time_ext=NULL,time.point=10,unit="years",minProb=min(bh_test$p[,2]),dist.label=0.02,
                          dist.label2=0.02)


