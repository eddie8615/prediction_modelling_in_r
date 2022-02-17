
#############################################
#
# Introduction to Prediction Modelling.
# D. Shamsutdinova (2021).
#
# Exercise & Practical: Cox Regression and SUrvival Analysis.
#
############################################
# Exercise on Survival Analysis and  Cox regression session 
############################################

library(survival)
library(help=survival) #this is how you can see what's inside the package and available datasets
library(ggplot2)
library(lattice)

### loading and exploring the data ####
# data description
# https://rdrr.io/cran/survival/man/stanford2.html

data(stanford2, package = 'survival')
names(stanford2)
length(stanford2)
sum(is.na(stanford2$t5))
sum(is.na(stanford2$age))

# id:	ID number
# time:	survival or censoring time
# status:	censoring status
# age:	in years
# t5:	T5 mismatch score

describe(stanford2)
dim(stanford2)
summary(stanford2)
hist(stanford2$time)
hist(stanford2$age)
hist(stanford2$t5)

### K-M curve and log-rank tests  #### 
S = Surv(stanford2$time, stanford2$status)
KM <- survfit(S ~ 1, data = stanford2)
KM
plot(KM, xlab="t", ylab = "Survival Probability", main = "Kaplan-Meier Estimate for all population", col="blue", ylim = c(0.1,1)) 
ggsurvplot( KM, data = stanford2, risk.table = TRUE)

# survival probabilities at 500 and 1000
summary(KM, times = c(500,1000))

#we can introduce age categories and see if age discriminates survival 
stanford2$age_cat = ifelse(stanford2$age < 35, 0, ifelse(stanford2$age > 50,2,1))
length(stanford2$age_cat)
hist(stanford2$age_cat)
logrank_2 <- survdiff(S ~ age_cat, data = stanford2)
logrank_2 
#corresponding K-M curves:
plot(survfit(S ~ stanford2$age_cat), xlab="t", ylab = "Survival Probability", main = "Kaplan-Meier Estimate for all population", col=c("blue","black","red"), ylim = c(0.1,1)) 

########################## exercise 1. Fitting Cox regressions ###################################
# 1a) fit Cox model on categorical age and t5




# 1b) fit Cox model on continuous age and t5



# 1c) what hazard ratio do the results imply for a 55 year old vs a 30 year old person for each of these models? 
#from the first model: 

#from the second model:


# what can you draw from the results? 

### exercise 3
#check if proportional hazard assumptions are valid for cox2 model and plot Schoenfeld residuals#
#your code here:




### end of exercise 3 ###

plot(checkph, var = 1, main = paste('Schoenfeld residual for age, Cox2 model', names(coef(cox2))[1] ))
abline(h = coef(cox2)[1], col = "red", lwd = 2)

plot(checkph, var = 2, main = paste('Schoenfeld residual for age, Cox2 model', names(coef(cox2))[2] ))
abline(h = coef(cox2)[2], col = "red", lwd = 2)

### Effect plots #### 
# checking and plotting impact of t5 within each age group 

ND <- expand.grid( age = seq(18, 75, length.out = 4), t5 = seq(0, 3.0, length.out = 4))
head(ND, 20)
prs <- predict(cox2, ND, se.fit = TRUE, type = "lp")
ND$pred <- prs[[1]]
ND$se <- prs[[2]]
ND$lo <- ND$pred - 1.96 * ND$se
ND$up <- ND$pred + 1.96 * ND$se
ND
# plot effect of t5 within each age group:
plot(ND$age, ND$pred)
plot(ND$t5, ND$pred)
xyplot(pred + lo + up ~ age |t5, data = ND, type = "l", 
       lty = c(1,2,2), col = "black", lwd = 2, xlab = "age", 
       ylab = "Linear predictor")

