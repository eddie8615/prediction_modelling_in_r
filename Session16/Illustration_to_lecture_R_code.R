
#############################################
#
# Introduction to Prediction Modelling.
#
# Exercise & Practical: Cox Regression and SUrvival Analysis.
#
############################################
# Lecture illustration code
############################################

library(survival)
library(ggplot2)
library(survminer) #for ggsurvplot

############# EXAMPLE1 ################

######## Building Kepler-Meier estimates of Survival Function ####### 
#S(t) = P(event time > t),  K-M estimate S(t) = S(t-1) x (1-d/n), 
#d - events at t, n - number at risk at t

#Generic example
t<- c(1,2,3,3,3,6,12,12,12,12)
status<- c(1,1,1,1,1,1,0,0,0,0)
data1 = data.frame(t = t, status = status)
data1
KM_fit <- survfit(Surv(t,status) ~ 1, data = data1)

ggsurvplot( KM_fit, data = data1, risk.table = TRUE)
# OR use this one
plot(KM_fit, xlab = "Time to Event (months)", ylab = "Survival Probability", 
     main = "Kaplan-Meier Estimate of S(t)")
# one more way to plot KM curve is using rms package - survplot(npsurv(Surv(t,status) ~ 1), xlab="t")

#lets see survival function with correspondent n and d at each event and censored time 
summary(KM_fit)

# here an observation that had event at t=3 dropped out at t=2 
t2<- c(1,2,2,3,3,6,12,12,12,12)
status2<- c(1,1,0,1,1,1,0,0,0,0)
data2 = data.frame(t = t2, status = status2)
KM_fit_2 <- survfit(Surv(t2,status2) ~ 1)
ggsurvplot( KM_fit_2, data = data2, risk.table = TRUE)
summary(KM_fit_2)

#plot both survival curves on one on one graph for comparison
plot(KM_fit_2, xlab = "Time to Event (months)", ylab = "Survival Probability", 
     main = "Kaplan-Meier Estimate of S1(red) and S2(black)")
lines(KM_fit, col='red')
# Slight gap between two plots

#One can also use Breslow estimator for Survival function, 
# Breslow estimate S(t) = S(t-1) * exp(-d/n), d - events at t, n - number at risk at t
# by construction Breslow estimate is > K-M estimate 
Br_fit <- survfit(Surv(t2, status2) ~ 1, type = "fleming-harrington")
summary(Br_fit)
plot(Br_fit,xlab = "Time to Event (months)", ylab = "Survival Probability", main='Breslow estimators for S1 and S2' )
lines(KM_fit_2, col="red")

############# EXAMPLE 2 ################
### Using KM curves for exploratory group analysis  #### 
tt <- c(1,2,2,2,3,3,3,5,7,8,10, 11,11,11,12,12,12,12,10,9,10,12,8,10,12,11,5,7,13,12,10)
stat <-    c(1,1,1,0,0,1,1,1,0,0,0,0,0,1,1,1,0,1,0,1,0,0,0,1,0,1,0,1,1,1,1)
gender1 <- c(0,0,1,1,1,0,1,1,1,1,1,0,0,1,1,1,0,1,0,1,0,0,0,1,0,1,0,1,1,1,1)
gender2 <- c(1,1,1,0,0,0,1,1,0,0,0,0,0,0,1,0,1,1,0,0,1,0,0,1,0,1,0,1,1,1,0)

S <- Surv(tt, stat)
plot(survfit(S ~ 1), xlab="t", ylab = "Survival Probability", main = "Kaplan-Meier Estimate for all population", col="blue") 
plot( survfit(S ~ gender1), col=c('red', 'blue'),  xlab="t",  ylab = "Survival Probability", main = "K-M Survival function by gender, example 1") 
legend("bottomleft", legend = c( 'gender=0', 'gender=1'), col = c('red', 'blue'), pch='_')
plot( survfit(S ~ gender2), col=c('red', 'blue'), xlab="t", ylab = "Survival Probability", main = 'K-M Survival function by gender, example 2') 
legend("bottomleft", legend = c( 'gender=0', 'gender=1'), col = c('red', 'blue'), pch='_')

#alternative way to plot KM is from rms package - survplot(npsurv(S ~ gender1), xlab="t")
#and another way to plot KM curve is with ggsurvplot:
data_g1 = data.frame(t = tt, status = stat, gender = gender1)
ggsurvplot(survfit(S ~ gender1, data = data_g1),
           conf.int = TRUE,
           risk.table = TRUE, # Add risk table
           risk.table.col = "strata", # Change risk table color by groups
           linetype = "strata", # Change line type by groups
           surv.median.line = "hv", # Specify median survival
           ggtheme = theme_bw(), # Change ggplot2 theme
           palette = c("#E7B800", "#2E9FDF"))
#------#

#Plotting cumulative hazard functions
km_fit_gender1 = survfit(S~gender1)
km_fit_gender2 = survfit(S~gender2)
plot(km_fit_gender1, fun = "cumhaz", main = 'Cumulative hazards by gender, example 1', col = c("red", "blue"))
plot(km_fit_gender2, fun = "cumhaz", main= 'Cumulative hazards by gender, example 2', col = c("red", "blue"))

#Performing log-rank test
logrank1 <- survdiff(Surv(tt, stat) ~ gender1)
logrank2 <- survdiff(Surv(tt, stat) ~ gender2)
logrank1
logrank2
#p-value of 0.08 and 0.09 are rather low so the curves do differ, however not at 0.05 cut off


# --------------------FIT COX REGRESSION--------------------
fit1 <- coxph(Surv(tt, stat) ~ gender1)
#Checking proportional hazards assumption
check_PH_1 <- cox.zph(fit1, transform = "km")
check_PH_1  #p-value 0.0067 is low and that means that proportionality is failing here

#fit regression for the second vairation of the data
fit2 <- coxph(Surv(tt, stat) ~ gender2)
check_PH_2 <- cox.zph(fit2, transform = "km")
check_PH_2 #p-value is high and this is good, i.e. we can not reject hypothesis that the hazards are propotionate

# plotting Schoenfeld residuals - 
#we want to see that the dots are around the flat line both up and down at all times 
#this indicates constant hazard ratio in time
# this is the case for gender2, not for gender1
plot(check_PH_1, var = 1)
abline(h = coef(fit1)[1], col = "red", lwd = 2)
plot(check_PH_2, var = 1)
abline(h = coef(fit2)[1], col = "red", lwd = 2)



