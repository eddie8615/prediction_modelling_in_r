# Prediction_Modelling_in_R
All contents in this repository is from Prediction Modelling module of ASH&HI at King's College London

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
