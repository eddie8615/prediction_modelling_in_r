# Prediction_Modelling_in_R
All contents in this repository is from Prediction Modelling module of ASH&HI at King's College London\
This repository is for self-studying R codes from practical sessions. Each session has different topics regarding prediction modelling.


## Contents

### Session 1
Session 1 consists of basics of R coding to remind the syntax.
- Sample generating function `rnorm`
- for loop

### Session 2
Session 2 file is in the Session 1 folder. As a same purpose of the exercise with Session 1, this file also contains fundamental functions, manipulation techniques and plotting the data
- import data (.csv, .dta, .sav)
- Checking dimensions and summary statistics
- Plotting the data on scatter plot and draw a best fitted line using `abline`
- Plotting the data on `ggplot` in various types of figures such as scatter plot (`geom_point()`), histogram (`geom_histogram()` and `facet_wrap()` to divide the data by specified variable and plot them respectively) and density diagram(`geom_density()`)

### Session 3
Session 3 covers Generalised Linear Model (GLM) which is a root model of the models including from simple and multiple regression, fundamental model, to support vector machine (SVM), regularised model (Ridge, LASSO, Elastic net) and many other fancy models.
- Using Boston housing price data, fit simple linear regression model (Outcome variable: Median value of owner-occupied homes in $1000's, Explanatory variable: % lower status of the population)
- By looking the coefficients, we can conduct statistical inference and explain what the model is showing
- Performing simple regression on unrelated data
- Correlation matrix between outcome and explanatory variables

### Session 4
Session 4 covers calculating covariance matrix from scratch 
- Matrix multiplication
- Covariance matrix
- Inverse matrix
- Comparing simple linear regression with polynomial regression using ANOVA (Analysis of Variance)

### Session 5
Session 5 covers the final part of generalised linear models. 
- Exploring multi-collinearity of variables using variance inflation factors in BOSTON dataset
- Modelling GLM with a simple dataset consisting of 100 variables with 100 observation which violates the statistical analysis golden rule: each variable have at least 10 entries
- Plotting a histogram of the coefficients of 100 variables in a GLM which shows most of them are around zero.

### Session 6
This session is to practice cross validation for fitting statistical models, such as simple and multiple regression models using various types of data

- K-fold cross-validation
- Repeated cross-validation
- The whole process of model assessment for its prediction (apparent validation, internal validation, external validation)
- Non-random split

### Session 7
This session is about regularising regression models such as Ridge and LASSO model using `glmnet` package.
- Ridge
- LASSO

### Session 11
Session 11 covers regularised logistic regression and its fitting mechanism such as cross-validation for tuning lambda. Used data was predicting autism using stimuli of particular brain regions and Altzheimer dataset from AppliedPredictiveModeling package\
This session also covers two popular R packages for modelling regularised logistic regression such as `caret` and `glmnet`.
- LASSO logistic regression - Parameter selection
- Small portion of data analysis using logistic regression to look odds ratios
- To evaluate the model performance, `confusionMatrix` was used that shows 'Sensitvity', 'Specificity' and 'Accuracy'
- Used `roc` to measure discrimination
- `glmnet` vs `caret`

### Session 16
Session 16 covers Cox regression that enable us to measure "time-till-event' with adding risk factors. The event can be anything not just negative event such as:
- Recovery period
- Survival from cancer or any disease since diagnosis etc.
Addtion to Cox model, Kaplan-Meier model, hazard function and survival function are also included to understand the concept of Cox model. DementiaData was used and various methods of assessing the model performance were also performed
- Cox regression
- Survival function
- Hazard function
- Concordance index (c-statistics), Calibration
- log rank -> to test the difference of statistical properties
- LASSO Cox

### Session 17
Session 17 covers Random Forest and Decision tree in classification and regression problems. The used dataset for regression is Boston housing data and for classification is 'Stage C Prostate Cancer' dataset. [Check Here](https://vincentarelbundock.github.io/Rdatasets/doc/rpart/stagec.html)

- Decision tree
- Regression forest
- Random forest
- Pruning to improve prediction performance

### Session 19
Session 19 covers Support Vector Machine (SVM). The aim of this session is to model a SVM to predict whether individuals experienced a coronary heat disease (CHD) or not using South African heart Disease dataset. The dataset contains 8 predictors. This session also compared the predictive performance between linear kernel and RBF kernel with fine tuning.

- Support Vector Machine
- Linear kernel
- RBF kernel
