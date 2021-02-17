# Title: C3T2 Pipeline Script

# Last update: 2/13/2021

# File: C3T2_Mattis.R
# Project name: Course 3, Task2


###############
# Project Notes
###############

# Summarize project: 
# Goal: Determine which of two computer brands our customers prefer
        # dv = 'brand'
# 1. build an optimized model with data from CompleteResponses
# 2. Apply model to SurveyIncomplete
# 3. Export results from each of the classifiers tried


# Assignment "<-" short-cut:
#   OSX [Alt]+[-] (next to "+" sign)
#   Win [Alt]+[-]

# Clear console: CTRL + L

###############
# Housekeeping
###############

# Clear objects if necessary
# rm(list = ls())

# get working directory
# getwd()

# set working directory
# setwd("directory_that_your_project_is_located_in")
# dir()

#show loaded packages
# search()

################
# Load packages
################

install.packages("caret")
install.packages("corrplot")
install.packages("readr")
install.packages("mlbench")
#install.packages("doMC")        # for OSX parallel processing (see below) 
install.packages("doParallel")  # for Win parallel processing (see below) 
library(caret)
library(corrplot)
library(readr)
library(gbm)
library(mlbench)
#library(doMC)                   # for OSX
library(doParallel)             # for Win


#####################
# Parallel processing
#####################

# NOTE: Be sure to use the correct package for your operating system. 

#--- for OSX ---#

#detectCores()            # detect number of cores
#registerDoMC(cores = 2)  # set number of cores (don't use all available)

#--- for Win ---#

detectCores()          # detect number of cores
cl <- makeCluster(2)   # select number of cores
registerDoParallel(cl) # register cluster
getDoParWorkers()      # confirm number of cores being used by RStudio
# Stop Cluster. After performing your tasks, make sure to stop your cluster. 
stopCluster(cl)


##############
# Import data 
##############

##-- Load Train/Existing data (Complete Responses) --##

crOOB <- read.csv("CompleteResponses.csv", stringsAsFactors = FALSE)
str(crOOB)
#verify a data type of a column:
print(class(crOOB$credit))

##--- Load Predict/New data (Survey Incomplete) [dv = NA, 0, or Blank] ---##

siOOB <- read.csv("SurveyIncomplete.csv", stringsAsFactors = FALSE)
str(siOOB)


##--- Load preprocessed datasets that have been saved ---##

#read.csv("dataset_name1.csv", stringsAsFactors = FALSE) 
#read.csv("dataset_name2.csv", stringsAsFactors = FALSE) 


###############
# Save datasets
###############

# can save all datasets to csv format, or 
# can save the ds with the best performance (after all modeling)

#write.csv(ds_object, "ds_name.csv", row.names = F)


################
# Evaluate data
################

##--- Complete Responses Dataset ---##

str(crOOB)  # 35136 obs. of  8 variables 
# 'data.frame':	9898 obs. of  7 variables:
# $ salary : num  119807 106880 78021 63690 50874 ...
# $ age    : int  45 63 23 51 20 56 24 62 29 41 ...
# $ elevel : int  0 1 0 3 3 3 4 3 4 1 ...
# $ car    : int  14 11 15 6 14 14 8 3 17 5 ...
# $ zipcode: int  4 6 2 5 4 3 5 0 0 4 ...
# $ credit : num  442038 45007 48795 40889 352951 ...
# $ brand  : int  0 1 0 1 0 1 1 1 0 1 ...

# view first/last obs/rows
head(crOOB)
tail(crOOB)

# check for missing values 
anyNA(crOOB)
# [1] FALSE

# check for duplicates
anyDuplicated((crOOB))
# [1] 0
# Good news!  This OOB data is clean!

##--- Survey Incomplete ---##

# NOTE: often don't have a second (dataset with dv = 0, NA, or Blank)

str(siOOB)
# 'data.frame':	5000 obs. of  7 variables:
# $ salary : num  150000 82524 115647 141443 149211 ...
# $ age    : int  76 51 34 22 56 26 64 50 26 46 ...
# $ elevel : int  1 1 0 3 0 4 3 3 2 3 ...
# $ car    : int  3 8 10 18 5 12 1 9 3 18 ...
# $ zipcode: int  3 3 2 2 3 1 2 0 4 6 ...
# $ credit : num  377980 141658 360980 282736 215667 ...
# $ brand  : int  1 0 1 1 1 1 1 1 1 0 ...

# view first/last obs/rows
head(siOOB)
tail(siOOB)
#it is a little strange that brand has 0 and 1s filled in, mostly 0s.
  # will do some preprocessing to fix

# check for missing values 
anyNA(siOOB)
# [1] FALSE

# check for duplicates
anyDuplicated((siOOB))
# [1] 0

# Incomplete Survey is also a clean OOB data set

dir()

#############
# Preprocess
#############

##--- Complete Responses ---##

#data set is clean - no need at this point to remove/update columns or rows
# this will be a classification model (want an answer of 0 or 1, not a continuous number), 
# so will make our dv a factor instead of int
crOOB$brand <- as.factor(crOOB$brand)


##--- Incomplete Survey ---##

# will set the 'brand' column to all be zeros to avoid future confusion
# since this column is supposedly incomplete data and should be unknown
siOOB$brand <- 0
siOOB$brand <- as.factor(siOOB$brand)
str(siOOB)



#####################
# EDA/Visualizations
#####################

# statistics
summary(crOOB)

# plots
hist(crOOB$brand)
hist(crOOB$salary)
#salary very evenly distributed
hist(crOOB$credit)
#credit very evenly distributed
hist(crOOB$age)
#age very evenly distributed
plot(crOOB$salary, crOOB$age)
#not helpful at all.... 

qqnorm(crOOB$salary) # Be familiar with this plot, but don't spend a lot of time on it
#shape means it is "light tailed" - likely due to the abruptness of the hist plot
qqnorm(crOOB$age)

################
# Sampling
################

# create 10% sample 
set.seed(1) # set random seed
crOOB10p <- crOOB[sample(1:nrow(crOOB), round(nrow(crOOB)*.1),replace=FALSE),]
nrow(crOOB10p)
head(crOOB10p) # ensure randomness

# 1k sample
set.seed(1) # set random seed
crOOB1k <- crOOB[sample(1:nrow(crOOB), 1000, replace=FALSE),]
nrow(crOOB1k) # ensure number of obs
head(crOOB1k) # ensure randomness


#######################
# Feature selection
#######################

#######################
# Correlation analysis
#######################

# good for num/int data 
# can't do correlation for factor data, so will make a special set
  # with a conversion in case any factor correlates strongly with brand

corrSet <- crOOB1k
corrSet$brand <- as.integer(corrSet$brand)
# calculate correlation matrix for all vars
corrAll <- cor(corrSet[,1:7])
# view the correlation matrix
corrAll
# plot correlation matrix
corrplot(corrAll, method = "circle")
corrplot(corrAll, order = "hclust") # sorts based on level of collinearity
# find IVs that are highly corrected (ideally >0.90)
corrIV <- cor(crOOB1k[,1:6])
corrIV
corrplot(corrIV, method = "circle")
# create object with indexes of highly corr features
corrIVhigh <- findCorrelation(corrIV, cutoff=0.8)   
# print indexes of highly correlated attributes
corrIVhigh
# integer(0)

#none of the features are highly correlated enough to where we would need 
  # to remove any of them



###########################################
# caret RFE (Recursive Feature Elimination)
###########################################

# lmFuncs - linear model
# rfFuncs - random forests
# nbFuncs - naive Bayes
# treebagFuncs - bagged trees

## ---- rf ---- ##

# define the control using a random forest selection function (regression or classification)
RFcontrol <- rfeControl(functions=rfFuncs, method="cv", number=5, repeats=1)

# run the RFE algorithm
set.seed(7)
# rfe (independent v, dv, etc...)
rfeRF <- rfe(crOOB1k[,1:6], crOOB1k[,7], sizes=c(1:7), rfeControl=RFcontrol)
rfeRF 
# Variables Accuracy  Kappa     AccuracySD KappaSD    Selected
# 2         0.909     0.8081    0.01835    0.03816        *

# plot the results
plot(rfeRF, type=c("g", "o"))
# show predictors used
predictors(rfeRF)
# [1] "salary" "age"  
varImp(rfeRF)
# Overall
# salary 72.51890
# age    45.26829


##############################
# Feature engineering
# https://www.datavedas.com/feature-engineering-in-r/
##############################

#Feature Transformation
  # histograms for skewness issues
    # histograms look good - no skewness issues
#Feature Scaling
  # normalize for extreme outliers
    # data columns are all pretty linear with no extreme outliers
#Feature Construction
  # Binning - potentially investigate later

#Feature Reduction
# remove unimpactful variables

# create ds with predictors from varImp on top model
rfeRF_cr <- crOOB1k[,predictors(rfeRF)]
str(rfeRF_cr)
# add dv
rfeRF_cr$brand <- crOOB1k$brand
# confirm new ds
str(rfeRF_cr)

# define the control using a random forest selection function (regression or classification)
RFcontrol <- rfeControl(functions=rfFuncs, method="cv", number=5, repeats=1)

# run the RFE algorithm
set.seed(7)
# rfe (independent v, dv, etc...)
rfeRF_FE <- rfe(rfeRF_cr[,1:2], rfeRF_cr[,3], sizes=c(1:3), rfeControl=RFcontrol)
rfeRF_FE 
# Variables Accuracy  Kappa     AccuracySD KappaSD    Selected
# 2         0.912     0.8146    0.01945    0.0399        *

# plot the results
plot(rfeRF_FE, type=c("g", "o"))

# so, this reduced data set(rfeRF_cr) is just as accurate as the full variable set!


##################
# Train/test sets
##################

# crOOB
set.seed(123) 
inTraining <- createDataPartition(crOOB1k$brand, p=0.75, list=FALSE)
oobTrain <- crOOB1k[inTraining,]   
oobTest <- crOOB1k[-inTraining,]   
# verify number of obs 
nrow(oobTrain) # 751
nrow(oobTest)  # 249


# rfeRF_cr - reduced feature set
set.seed(123) 
inTraining <- createDataPartition(rfeRF_cr$brand, p=0.75, list=FALSE)
RFcrTrain <- rfeRF_cr[inTraining,]   
RFcrTest <- rfeRF_cr[-inTraining,]   
# verify number of obs 
nrow(RFcrTrain) # 751
nrow(RFcrTest)  # 249



################
# Train control
################

# set cross validation - 10 folds desired
fitControl <- trainControl(method="repeatedcv", number=10, repeats=1) 


###############
# Train models
###############

?modelLookup()
modelLookup('rf')
modelLookup('gbm')

## ------- RF ------- ##


# default
set.seed(123)
oobRFfit <- train(brand~.,data=oobTrain,method="rf",
                  importance=T,
                  trControl=fitControl)
oobRFfit
# mtry  Accuracy   Kappa    
# 2     0.8574822  0.6941370
# 4     0.8947828  0.7782181
# 6     0.8987477  0.7868037
plot(oobRFfit)
varImp(oobRFfit)
# Importance
# salary     100.000
# age         68.366
# zipcode      2.715
# credit       2.544
# elevel       2.399
# car          0.000

# manual grid to attempt a better fit
rfGrid <- expand.grid(mtry=c(1,2,3,4,5))  
set.seed(123)
# fit
oobRFfitM <- train(brand~.,data=oobTrain,method="rf",
                  importance=T,
                  trControl=fitControl,
                  tuneGrid=rfGrid)
oobRFfitM
# mtry  Accuracy   Kappa    
# 1     0.7602461  0.4665565
# 2     0.8589042  0.6962412
# 3     0.8920806  0.7726803
# 4     0.8961162  0.7815356
# 5     0.9041347  0.7974016
#a smidge better with 5... pretty minor gains though
plot(oobRFfitM)

# - Random Forest with reduced feature set ---#
set.seed(123)
RFRFfit <- train(brand~.,data=RFcrTrain,method="rf",
                 importance=T,
                 trControl=fitControl)
RFRFfit
# Accuracy   Kappa    
# 0.8987828  0.7858545
## ------- GBM ------- ##

set.seed(123)
oobGBMfit <- train(brand~., data=oobTrain, method="gbm", trControl=fitControl)
oobGBMfit
plot(oobGBMfit)
# interaction.depth  n.trees  Accuracy   Kappa   
# 3                  150      0.8708165  0.7255822
varImp(oobGBMfit)
# Overall
# salary  100.000
# age      55.997
# credit   18.822
# car       6.173
# zipcode   2.497
# elevel    0.000

# - GBM with reduced feature set ---#
RFGBMfit <- train(brand~., data=RFcrTrain, method="gbm", trControl=fitControl)
RFGBMfit
# interaction.depth  n.trees  Accuracy   Kappa 
# 3                  150      0.8948530  0.7779362

##################
# Model selection
##################

#-- OOB data sets --# 

oobFitComp1k <- resamples(list(rf=oobRFfit, rfManual=oobRFfitM, gbm=oobGBMfit))
# output summary
summary(oobFitComp1k)

# Accuracy 
#           Min.     1st Qu.   Median    Mean      3rd Qu.   Max.      NA's
# rf       0.8133333 0.8933333 0.9054054 0.8987477 0.9166667 0.9342105    0
# rfManual 0.8266667 0.8963514 0.9127928 0.9040991 0.9200000 0.9342105    0
# gbm      0.7600000 0.8562162 0.8800000 0.8708165 0.9063514 0.9078947    0

# Kappa 
#               Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# rf       0.6064468 0.7750056 0.7997298 0.7868037 0.8258530 0.8615160    0
# rfManual 0.6322143 0.7880355 0.8141193 0.7980755 0.8329577 0.8615160    0
# gbm      0.4874715 0.6996001 0.7436492 0.7255822 0.8028833 0.8092989    0

#-- Reduced Factor Sets --#

RFFitComp1k <- resamples(list(RFrf=RFRFfit, RFgbm=RFGBMfit))
summary(RFFitComp1k)
# Accuracy 
#       Min.      1st Qu.    Median   Mean       3rd Qu   Max.      NA's
# RFrf  0.8666667 0.8933333 0.8993694 0.8987828 0.9063514 0.9342105    0
# RFgbm 0.84      0.8709104 0.8873684 0.8948530  0.926298 0.96        0

# Kappa 
#            Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# RFrf  0.7188906 0.7714086 0.7890005 0.7858545 0.8007508 0.8577844    0
# RFgbm 0.6538462 0.7302640 0.7592728 0.7779362 0.8430474 0.9151264    0

##--- Save/load top performing model ---##

# save top performing model after validation
saveRDS(oobRFfitM, "oobRFfitM.rds")  

# load and name model
RFfitManual <- readRDS("oobRFfitM.rds")



############################
# Predict testSet/validation
############################

# predict with RF
rfPred1 <- predict(RFfitManual, oobTest)
# performace measurment
postResample(rfPred1, oobTest$brand)
# Accuracy     Kappa 
# 0.9236948    0.8392839 
# accuracy went up by a %.

# plot predicted verses actual
plot(rfPred1, oobTest$brand)



##########################################
# Predict new data (Incomplte Survey Data)
#########################################


rfPred2 <- predict(RFfitManual, siOOB)
head(rfPred2)

#Use the summary() function and your prediction object to learn how many 
  #individuals are predicted to prefer Sony and Acer.
summary(rfPred2)
# 0    1 
# 1937 3063

#You should also create a chart that shows the preferences for Sony and Acer for 
# the entire 15,000 existing customer survey.  
  # do this as a stacked bar in excel with real and predicted
summary(crOOB$brand)
# 0    1 
# 3744 6154 

################################################################
# Extra - would the model get better if full dataset was used? #
################################################################
#will try this out on RF for reduced variable set
# it performed the the best of the reduced variable models

#make the reduced factor set
rfe_cr_all <- crOOB[,predictors(rfeRF)]
str(rfe_cr_all)
# add dv
rfe_cr_all$brand <- crOOB$brand
# confirm new ds
str(rfe_cr_all)

set.seed(123) 
inTraining <- createDataPartition(rfe_cr_all$brand, p=0.75, list=FALSE)
crAllTrain <- rfe_cr_all[inTraining,]   
crAllTest <- rfe_cr_all[-inTraining,]   
# verify number of obs 
nrow(crAllTrain) # 7424
nrow(crAllTest)  # 2474
# - RF ALL with reduced feature set ---#
RFfitAll <- train(brand~., data=crAllTrain, method="rf",
                  importance=T,
                  trControl=fitControl)
RFfitAll
# Accuracy   Kappa    
# 0.9053067  0.7986832

#Very marginal gains.  1K data set gave:
# Accuracy   Kappa    
# 0.8987828  0.7858545