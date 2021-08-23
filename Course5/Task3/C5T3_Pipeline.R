# Title: C5T3 - Predictive models for sentiment analysis

# Last update: 8/23/21
# Author: Brian Mattis
# File: C5T3_Pipeline.R


###################\
# Project Notes ####
###################/

# Summarize project: Build predictive models for sentiment analysis for 
# both the Samsung Galaxy and Apple iPhone.  Apply these models to 
# the Big Data pulled from common-crawl on prior task to determine the 
# phone with the higher sentiment.

# training data uses sentiment from 0-5. 
#0: very negative
#1: negative
#2: somewhat negative
#3: somewhat positive
#4: positive
#5: very positive

# use C5.0, RF, KKNN, and SVM for model generation

# Outputs:
# 1) Written report with recommendations (high-level)
# 2) Lessons learned report


##################\
# Housekeeping ####
##################/

# Clear objects if necessary
rm(list = ls())

# Clear console: CTRL + L
# reference items in df: df[row_num, col_num]

###################\
# Load packages ####
###################/

#install.packages("caret")
#install.packages("klaR")
#install.packages("corrplot")
#install.packages("readr")
#install.packages("mlbench")
#install.packages("doParallel")  # for Win parallel processing (see below) 
#install.packages("data.table")
#install.packages("dplyr")
#install.packages("C50")
#install.packages("kknn")
library(caret)
library(corrplot)
library(readr)
library(plotly)
library(mlbench)
library(doParallel)             # for Win
library(data.table)
library(dplyr)
library(DataExplorer)
library(reshape2) # for melt
#########################\
# Parallel processing ####
#########################/

# NOTE: Be sure to use the correct package for your operating system. 

#--- for Win ---#
#detectCores()          # detect number of cores
#cl <- makeCluster(2)   # select number of cores
#registerDoParallel(cl) # register cluster
#getDoParWorkers()      # confirm number of cores being used by RStudio
# Stop Cluster. After performing your tasks, make sure to stop your cluster. 
#stopCluster(cl)

#################\
# Import data ####
#################/

dfGOOB <- read.csv("data/galaxy_smallmatrix_labeled_9d.csv", stringsAsFactors = FALSE)
str(dfGOOB)
summary(dfGOOB)
introduce(dfGOOB)

dfIOOB <- read.csv("data/iphone_smallmatrix_labeled_8d.csv", stringsAsFactors = FALSE)
str(dfIOOB)
summary(dfIOOB)
introduce(dfIOOB)

dfLMOOB <- read.csv("data/LargeMatrix.csv", stringsAsFactors = FALSE)
str(dfLMOOB)
introduce(dfLMOOB)


####################\
# Evaluate data #####
####################/
plot_ly(dfIOOB, x= ~dfIOOB$iphonesentiment, type='histogram')
plot_ly(dfGOOB, x= ~dfGOOB$galaxysentiment, type='histogram')
#both have high level of 5-level sentiments.

# Data Cleaning #
plot_missing(dfGOOB)

anyNA(dfGOOB) # FALSE
anyNA(dfIOOB) # FALSE
anyNA(dfLMOOB) # FALSE

anyDuplicated((dfGOOB)) # 8
sum(duplicated(dfGOOB)) # 10271

anyDuplicated((dfIOOB)) # 3
sum(duplicated(dfIOOB)) # 10391

# this is more than 80% of the rows.
# for the smallMatrix data sets, as there are no webpage identifiers, it seems reasonable
# that multiple articles could have the same number of mentions and pos/neg/neutral ratings
# Will not remove any duplicates.

anyDuplicated((dfLMOOB)) # 0 # but this has the ID column, so that individualizes each

#################\
# Preprocess #####
#################/
#Todo: 
#Make two copies of the LM data (one each for use with dfG or dfI)
# remove ID column
dfLMG <- subset(dfLMOOB, select= -c(id))
dfLMI <- subset(dfLMOOB, select= -c(id))
#add in the DV column with dummy value of 0
dfLMG$galaxysentiment <- 0
dfLMI$iphonesentiment <- 0

# turn the DVs into factors for LM (do it for SM later)
dfLMG$galaxysentiment<- as.factor(dfLMG$galaxysentiment)
dfLMI$iphonesentiment <- as.factor(dfLMI$iphonesentiment)

#######################\
# Feature selection ####
#######################/


# **************** #
# Correlation      #
# **************** #
# Corr rules: If any IV corr > 0.95 with DV, then remove. 
#             If any pair of IVs corr > 0.90, then remove IV with lowest corr to DV.

GCorr <- cor(dfGOOB)
#correlations to DV
GCorr[,"galaxysentiment"]

#highest corr to DV is -0.34, so no need to remove any

corrplot(GCorr)
#too large of a matrix to view well.  Let's display as a list:
GCorrTable <- GCorr
GCorrTable[lower.tri(GCorrTable,diag=TRUE)]=NA  #Prepare to drop duplicates and meaningless information
GCorrTable=as.data.frame(as.table(GCorrTable))  #Turn into a 3-column table
GCorrTable=na.omit(GCorrTable)  #Get rid of the junk we flagged above
GCorrTable=GCorrTable[order(-abs(GCorrTable$Freq)),]    #Sort by highest correlation (whether +ve or -ve)
GCorrTable
# 21 I-V's correlating above 0.9 to eachother
dfGCor <- dfGOOB
dfGCor$htcphone <- NULL
dfGCor$nokiadispos <- NULL
dfGCor$nokiacamunc <- NULL
dfGCor$googleperpos <- NULL
dfGCor$samsungperunc <- NULL
dfGCor$nokiacamunc <- NULL
dfGCor$samsungdisneg <- NULL
dfGCor$iosperneg <- NULL
dfGCor$iphone <- NULL
dfGCor$nokiadisunc <- NULL
dfGCor$nokiaperunc <- NULL
dfGCor$nokiacampos <- NULL
dfGCor$nokiadisunc <- NULL
dfGCor$samsungdisunc <- NULL
dfGCor$nokiadisneg <- NULL
dfGCor$nokiaperunc <- NULL
dfGCor$iosperpos <- NULL
dfGCor$sonydisneg <- NULL
dfGCor$nokiadisneg <- NULL

ICorr <-  cor(dfIOOB)
#correlations to DV
ICorr[,"iphonesentiment"]
#highest is 0.359 - no need to remove any
corrplot(ICorr)
#too large of a matrix to view well.  Let's display as a list:
ICorrTable <- ICorr
ICorrTable[lower.tri(ICorrTable,diag=TRUE)]=NA  #Prepare to drop duplicates and meaningless information
ICorrTable=as.data.frame(as.table(ICorrTable))  #Turn into a 3-column table
ICorrTable=na.omit(ICorrTable)  #Get rid of the junk we flagged above
ICorrTable=ICorrTable[order(-abs(ICorrTable$Freq)),]    #Sort by highest correlation (whether +ve or -ve)
ICorrTable
# 20 I-V's correlating above 0.9 to eachother (some overlap)
dfICor <- dfIOOB
dfICor$htcphone <- NULL
dfICor$nokiadispos <- NULL
dfICor$nokiacamunc <- NULL
dfICor$googleperpos <- NULL
dfICor$nokiaperpos <- NULL
dfICor$samsungperunc <- NULL
dfICor$nokiacamunc <- NULL
dfICor$samsungperneg <- NULL
dfICor$iosperneg <- NULL
dfICor$nokiadisunc <- NULL
dfICor$ios <- NULL
dfICor$nokiaperunc <- NULL
dfICor$nokiacamneg <- NULL
dfICor$samsungdisunc <- NULL
dfICor$nokiadisneg <- NULL
dfICor$nokiaperunc <- NULL
dfICor$iosperunc <- NULL

# **************** #
# Feature Variance #
# **************** #
# http://topepo.github.io/caret/pre-processing.html#nzv
#galaxy
nzvGMetrics <- nearZeroVar(dfGOOB, saveMetrics = TRUE)
nzvGMetrics
nzvG <- nearZeroVar(dfGOOB, saveMetrics = FALSE)
nzvG
dfGnzv <- dfGOOB[,-nzvG]
#iphone
nzvIMetrics <- nearZeroVar(dfIOOB, saveMetrics = TRUE)
nzvIMetrics
nzvI <- nearZeroVar(dfIOOB, saveMetrics = FALSE)
nzvI
dfInzv <- dfIOOB[,-nzvI]

# **************** #
#       RFE        #
# **************** #

# Let's sample the data before using RFE
set.seed(123)
iphoneSample <- dfIOOB[sample(1:nrow(dfIOOB), 1000, replace=FALSE),]
galaxySample <- dfGOOB[sample(1:nrow(dfGOOB), 1000, replace=FALSE),]

# Set up rfeControl with randomforest, repeated cross validation and no updates
ctrl <- rfeControl(functions = rfFuncs, 
                   method = "repeatedcv",
                   number = 5,
                   repeats = 2,
                   verbose = TRUE)

# Use rfe and omit the response variable (attribute 59 iphonesentiment / galaxysentiment) 
rfeResultsI <- rfe(iphoneSample[,1:58], 
                   iphoneSample$iphonesentiment, 
                   sizes=(1:58), 
                   rfeControl=ctrl)

rfeResultsG <- rfe(galaxySample[,1:58], 
                   galaxySample$galaxysentiment, 
                   sizes=(1:58), 
                   rfeControl=ctrl)
# Get results
rfeResultsI
rfeResultsG
predictors(rfeResultsI)
predictors(rfeResultsG)
# Plot results
plot(rfeResultsI, type=c("g", "o"))
plot(rfeResultsG, type=c("g", "o"))

# create new data set with rfe recommended features
dfIrfe <- dfIOOB[,predictors(rfeResultsI)]
dfGrfe <- dfGOOB[,predictors(rfeResultsG)]

# add the dependent variable to iphoneRFE
dfIrfe$iphonesentiment <- dfIOOB$iphonesentiment
dfGrfe$galaxysentiment <- dfGOOB$galaxysentiment
# review outcome
str(dfIrfe)
str(dfGrfe)

# **************** #
# Subsets (bonus)  #
# **************** #
### Subsets where only columns specific to each phone are maintained

# dfG - keep only galaxy/android-related columns
dfGsub <- dfGOOB[c(2,7,9,14,19,24,29,34,39,44,49,54,56,58,59)]
# dfI - keep only iphone/ios-related columns
dfIsub <- dfIOOB[c(1,6,8,13,18,23,28,33,38,43,48,53,55,57,59)]


# **************** #
# DVs as factors   #
# **************** #
dfGOOB$galaxysentiment<- as.factor(dfGOOB$galaxysentiment)
dfIOOB$iphonesentiment <- as.factor(dfIOOB$iphonesentiment)
dfGCor$galaxysentiment<- as.factor(dfGCor$galaxysentiment)
dfICor$iphonesentiment <- as.factor(dfICor$iphonesentiment)
dfGnzv$galaxysentiment<- as.factor(dfGnzv$galaxysentiment)
dfInzv$iphonesentiment <- as.factor(dfInzv$iphonesentiment)
dfGrfe$galaxysentiment<- as.factor(dfGrfe$galaxysentiment)
dfIrfe$iphonesentiment <- as.factor(dfIrfe$iphonesentiment)
dfGsub$galaxysentiment<- as.factor(dfGsub$galaxysentiment)
dfIsub$iphonesentiment <- as.factor(dfIsub$iphonesentiment)


#####################\
# Train/test sets ####
#####################/

#dfGOOB
set.seed(123) 
inTraining <- createDataPartition(dfGOOB$galaxysentiment, p=0.70, list=FALSE)
dfGOOBTrain <- dfGOOB[inTraining,]   
dfGOOBTest <- dfGOOB[-inTraining,] 
# verify number of obs 
nrow(dfGOOBTrain) # 
nrow(dfGOOBTest)  # 
#dfIOOB
inTraining <- createDataPartition(dfIOOB$iphonesentiment, p=0.70, list=FALSE)
dfIOOBTrain <- dfIOOB[inTraining,]   
dfIOOBTest <- dfIOOB[-inTraining,] 
# verify number of obs 
nrow(dfIOOBTrain) # 
nrow(dfIOOBTest)  # 

#dfGCor
inTraining <- createDataPartition(dfGCor$galaxysentiment, p=0.70, list=FALSE)
dfGCorTrain <- dfGCor[inTraining,]   
dfGCorTest <- dfGCor[-inTraining,] 
#dfICor
inTraining <- createDataPartition(dfICor$iphonesentiment, p=0.70, list=FALSE)
dfICorTrain <- dfICor[inTraining,]   
dfICorTest <- dfICor[-inTraining,] 

#dfGnzv
inTraining <- createDataPartition(dfGnzv$galaxysentiment, p=0.70, list=FALSE)
dfGnzvTrain <- dfGnzv[inTraining,]   
dfGnzvTest <- dfGnzv[-inTraining,] 
#dfInzv
inTraining <- createDataPartition(dfInzv$iphonesentiment, p=0.70, list=FALSE)
dfInzvTrain <- dfInzv[inTraining,]   
dfInzvTest <- dfInzv[-inTraining,] 

#dfGrfe
inTraining <- createDataPartition(dfGrfe$galaxysentiment, p=0.70, list=FALSE)
dfGrfeTrain <- dfGrfe[inTraining,]   
dfGrfeTest <- dfGrfe[-inTraining,] 
#dfIrfe
inTraining <- createDataPartition(dfIrfe$iphonesentiment, p=0.70, list=FALSE)
dfIrfeTrain <- dfIrfe[inTraining,]   
dfIrfeTest <- dfIrfe[-inTraining,] 

#dfGsub
inTraining <- createDataPartition(dfGsub$galaxysentiment, p=0.70, list=FALSE)
dfGsubTrain <- dfGsub[inTraining,]   
dfGsubTest <- dfGsub[-inTraining,] 
#dfIsub
inTraining <- createDataPartition(dfIsub$iphonesentiment, p=0.70, list=FALSE)
dfIsubTrain <- dfIsub[inTraining,]   
dfIsubTest <- dfIsub[-inTraining,] 

######################\
# Train models OOB ####
######################/
## Train OOB data with C5.0, RF, SVM, kknn
## decide which is best and apply to the rest

## ------- C50 ------- ##
fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 2, 
                           verboseIter = TRUE)

system.time(c50GOOB <- train(galaxysentiment ~.,
                             data = dfGOOBTrain,
                             method = 'C5.0',
                             tuneLength = 5,
                             trControl = fitControl))
c50GOOB #211sec

system.time(c50IOOB <- train(iphonesentiment ~.,
                             data = dfIOOBTrain,
                             method = 'C5.0',
                             tuneLength = 5,
                             trControl = fitControl))
c50IOOB #187sec

## ------- Random Forest ------- ##
## training model with Random Forest, with 5 fold CV,tuneLength = 5, on sub1_train
fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 2, 
                           verboseIter = TRUE)

system.time(rfGOOB <- train(galaxysentiment ~.,
                            data = dfGOOBTrain,
                            method = 'rf',
                            tuneLength = 3,
                            trControl = fitControl))
rfGOOB #2507sec

system.time(rfIOOB <- train(iphonesentiment ~.,
                            data = dfIOOBTrain,
                            method = 'rf',
                            tuneLength = 3,
                            trControl = fitControl))
rfIOOB #2473sec

## ------- SVM (svmLinear2 e1071) ------- ##

fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 2, 
                           verboseIter = TRUE)

system.time(svmGOOB <- train(galaxysentiment ~.,
                             data = dfGOOBTrain,
                             method = 'svmLinear2',
                             tuneLength = 5,
                             trControl = fitControl))
svmGOOB #3417sec

system.time(svmIOOB <- train(iphonesentiment ~.,
                             data = dfIOOBTrain,
                             method = 'svmLinear2',
                             tuneLength = 5,
                             trControl = fitControl))
svmIOOB #720sec

## ------- kknn ------- ##
fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 2, 
                           verboseIter = TRUE)

system.time(kknnGOOB <- train(galaxysentiment ~.,
                             data = dfGOOBTrain,
                             method = 'kknn',
                             tuneLength = 5,
                             trControl = fitControl))
kknnGOOB #785sec

system.time(kknnIOOB <- train(iphonesentiment ~.,
                             data = dfIOOBTrain,
                             method = 'kknn',
                             tuneLength = 5,
                             trControl = fitControl))
kknnIOOB #787sec

##############################\
# Model Type selection OOB ####
##############################/
GOOBresample <- resamples(list(C50_GOOB=c50GOOB,RF_GOOB=rfGOOB, SVM_GOOB=svmGOOB, KKNN_GOOB=kknnGOOB))
summary(GOOBresample)
bwplot(GOOBresample)

IOOBresample <- resamples(list(C50_IOOB=c50IOOB,RF_IOOB=rfIOOB, SVM_IOOB=svmIOOB, KKNN_IOOB=kknnIOOB))
summary(IOOBresample)
bwplot(IOOBresample)


####################################\
# Predict testSet/validation OOB ####
####################################/

C50GOOB_Pred <- predict(c50GOOB, newdata = dfGOOBTest)
postResample(C50GOOB_Pred, dfGOOBTest$galaxysentiment)
rfGOOB_Pred <- predict(rfGOOB, newdata = dfGOOBTest)
postResample(rfGOOB_Pred, dfGOOBTest$galaxysentiment)
svmGOOB_Pred <- predict(svmGOOB, newdata = dfGOOBTest)
postResample(svmGOOB_Pred, dfGOOBTest$galaxysentiment)
kknnGOOB_Pred <- predict(kknnGOOB, newdata = dfGOOBTest)
postResample(kknnGOOB_Pred, dfGOOBTest$galaxysentiment)

C50IOOB_Pred <- predict(c50IOOB, newdata = dfIOOBTest)
postResample(C50IOOB_Pred, dfIOOBTest$iphonesentiment)
rfIOOB_Pred <- predict(rfIOOB, newdata = dfIOOBTest)
postResample(rfIOOB_Pred, dfIOOBTest$iphonesentiment)
svmIOOB_Pred <- predict(svmIOOB, newdata = dfIOOBTest)
postResample(svmIOOB_Pred, dfIOOBTest$iphonesentiment)
kknnIOOB_Pred <- predict(kknnIOOB, newdata = dfIOOBTest)
postResample(kknnIOOB_Pred, dfIOOBTest$iphonesentiment)

#---------Evaluation--------------#
confusionMatrix(C50GOOB_Pred, dfGOOBTest$galaxysentiment)
confusionMatrix(rfGOOB_Pred, dfGOOBTest$galaxysentiment)
confusionMatrix(kknnGOOB_Pred, dfGOOBTest$galaxysentiment)

confusionMatrix(C50IOOB_Pred,dfIOOBTest$iphonesentiment)
confusionMatrix(rfIOOB_Pred,dfIOOBTest$iphonesentiment)

####################################\
# Train models Feature Selection ####
####################################/
# Evaluate Cor, nzv, rfe, sub dataframes with best model from OOB

## ------- C50 ------- ##
#-------Galaxy -------------------#
fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 2, 
                           verboseIter = TRUE)

system.time(c50GCor <- train(galaxysentiment ~.,
                             data = dfGCorTrain,
                             method = 'C5.0',
                             tuneLength = 5,
                             trControl = fitControl))
c50GCor 

system.time(c50Gnzv <- train(galaxysentiment ~.,
                             data = dfGnzvTrain,
                             method = 'C5.0',
                             tuneLength = 5,
                             trControl = fitControl))
c50Gnzv

system.time(c50Grfe <- train(galaxysentiment ~.,
                             data = dfGrfeTrain,
                             method = 'C5.0',
                             tuneLength = 5,
                             trControl = fitControl))
c50Grfe 

system.time(c50Gsub <- train(galaxysentiment ~.,
                             data = dfGsubTrain,
                             method = 'C5.0',
                             tuneLength = 5,
                             trControl = fitControl))
c50Gsub 

#-------iPhone -------------------#
system.time(c50ICor <- train(iphonesentiment ~.,
                             data = dfICorTrain,
                             method = 'C5.0',
                             tuneLength = 5,
                             trControl = fitControl))
c50ICor 

system.time(c50Inzv <- train(iphonesentiment ~.,
                             data = dfInzvTrain,
                             method = 'C5.0',
                             tuneLength = 5,
                             trControl = fitControl))
c50Inzv

system.time(c50Irfe <- train(iphonesentiment ~.,
                             data = dfIrfeTrain,
                             method = 'C5.0',
                             tuneLength = 5,
                             trControl = fitControl))
c50Irfe 

system.time(c50Isub <- train(iphonesentiment ~.,
                             data = dfIsubTrain,
                             method = 'C5.0',
                             tuneLength = 5,
                             trControl = fitControl))
c50Isub 

############################################\
# Model Type selection Feature Selection ####
############################################/

Gc50FSresample <- resamples(list(c50GCor=c50GCor,c50Gnzv=c50Gnzv, c50Grfe=c50Grfe , c50Gsub=c50Gsub))
summary(Gc50FSresample)
bwplot(Gc50FSresample)

Ic50FSresample <- resamples(list(c50ICor=c50ICor,c50Inzv=c50Inzv, c50Irfe=c50Irfe , c50Isub=c50Isub))
summary(Ic50FSresample)
bwplot(Ic50FSresample)

##################################################\
# Predict testSet/validation Feature Selection ####
##################################################/

c50GCor_Pred <- predict(c50GCor, newdata = dfGCorTest)
postResample(c50GCor_Pred, dfGCorTest$galaxysentiment)
c50Gnzv_Pred <- predict(c50Gnzv, newdata = dfGnzvTest)
postResample(c50Gnzv_Pred, dfGnzvTest$galaxysentiment)
c50Grfe_Pred <- predict(c50Grfe, newdata = dfGrfeTest)
postResample(c50Grfe_Pred, dfGrfeTest$galaxysentiment)
c50Gsub_Pred <- predict(c50Gsub, newdata = dfGsubTest)
postResample(c50Gsub_Pred, dfGsubTest$galaxysentiment)

c50ICor_Pred <- predict(c50ICor, newdata = dfICorTest)
postResample(c50ICor_Pred, dfICorTest$iphonesentiment)
c50Inzv_Pred <- predict(c50Inzv, newdata = dfInzvTest)
postResample(c50Inzv_Pred, dfInzvTest$iphonesentiment)
c50Irfe_Pred <- predict(c50Irfe, newdata = dfIrfeTest)
postResample(c50Irfe_Pred, dfIrfeTest$iphonesentiment)
c50Isub_Pred <- predict(c50Isub, newdata = dfIsubTest)
postResample(c50Isub_Pred, dfIsubTest$iphonesentiment)

summary(c50ICor_Pred)
#---------Evaluation--------------#
#compare ones close to OOB results
confusionMatrix(c50Gnzv_Pred, dfGnzvTest$galaxysentiment)
confusionMatrix(c50Grfe_Pred, dfGrfeTest$galaxysentiment)

confusionMatrix(c50ICor_Pred,dfICorTest$iphonesentiment)
confusionMatrix(c50Irfe_Pred,dfIrfeTest$iphonesentiment)

##############################\
# Feature engineering #########
##############################/
summary(dfIOOB)
summary(dfGOOB)
# **************** #
# Recode DV        #
# **************** #
# create a new dataset that will be used for recoding sentiment
dfIRC <- dfIOOB
# recode sentiment to combine factor levels 0 & 1 and 4 & 5
dfIRC$iphonesentiment <- recode(dfIRC$iphonesentiment, '0' = 1, '1' = 1, '2' = 2, '3' = 3, '4' = 4, '5' = 4) 
# inspect results
summary(dfIRC)
str(dfIRC)
dfIRC$iphonesentiment <- as.factor(dfIRC$iphonesentiment)

dfGRC <- dfGOOB
# recode sentiment to combine factor levels 0 & 1 and 4 & 5
dfGRC$galaxysentiment <- recode(dfGRC$galaxysentiment, '0' = 1, '1' = 1, '2' = 2, '3' = 3, '4' = 4, '5' = 4) 
# inspect results
summary(dfGRC)
str(dfGRC)
dfGRC$galaxysentiment <- as.factor(dfGRC$galaxysentiment)

#dfGRC
inTraining <- createDataPartition(dfGRC$galaxysentiment, p=0.70, list=FALSE)
dfGRCTrain <- dfGRC[inTraining,]   
dfGRCTest <- dfGRC[-inTraining,] 
#dfIRC
inTraining <- createDataPartition(dfIRC$iphonesentiment, p=0.70, list=FALSE)
dfIRCTrain <- dfIRC[inTraining,]   
dfIRCTest <- dfIRC[-inTraining,] 

# **************************** #
# Principal Component Analysis #
# **************************** #
# data = training and testing from iphoneDF (no feature selection) 
# create object containing centered, scaled PCA components from training set
# excluded the dependent variable and set threshold to .95
preprocessParamsI <- preProcess(dfIOOBTrain[,-59], method=c("center", "scale", "pca"), thresh = 0.95)
print(preprocessParamsI) #26 needed to capture 95% of the variance
# use predict to apply pca parameters, create training, exclude dependant
dfIpcaTrain <- predict(preprocessParamsI, dfIOOBTrain[,-59])
# add the dependent to training
dfIpcaTrain$iphonesentiment <- dfIOOBTrain$iphonesentiment
# use predict to apply pca parameters, create testing, exclude dependant
dfIpcaTest <- predict(preprocessParamsI, dfIOOBTest[,-59])
# add the dependent to training
dfIpcaTest$iphonesentiment <- dfIOOBTest$iphonesentiment

# inspect results
str(dfIpcaTrain)
str(dfIpcaTest)

preprocessParamsG <- preProcess(dfGOOBTrain[,-59], method=c("center", "scale", "pca"), thresh = 0.95)
print(preprocessParamsG) #24 needed to capture 95% of the variance

dfGpcaTrain <- predict(preprocessParamsG, dfGOOBTrain[,-59])
dfGpcaTrain$galaxysentiment <- dfGOOBTrain$galaxysentiment
dfGpcaTest <- predict(preprocessParamsG, dfGOOBTest[,-59])
dfGpcaTest$galaxysentiment <- dfGOOBTest$galaxysentiment

# inspect results
str(dfGpcaTrain)
str(dfGpcaTest)

######################################\
# Train models Feature Engineering ####
######################################/
# Evaluate Recode (RC) and pca data sets with C5.0

## ------- C50 ------- ##
#-------Galaxy -------------------#
fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 2, 
                           verboseIter = TRUE)

system.time(c50GRC <- train(galaxysentiment ~.,
                            data = dfGRCTrain,
                            method = 'C5.0',
                            tuneLength = 5,
                            trControl = fitControl))
c50GRC 

system.time(c50Gpca <- train(galaxysentiment ~.,
                             data = dfGpcaTrain,
                             method = 'C5.0',
                             tuneLength = 5,
                             trControl = fitControl))
c50Gpca 
#-------iPhone -------------------#

system.time(c50IRC <- train(iphonesentiment ~.,
                             data = dfIRCTrain,
                             method = 'C5.0',
                             tuneLength = 5,
                             trControl = fitControl))
c50IRC 

system.time(c50Ipca <- train(iphonesentiment ~.,
                             data = dfIpcaTrain,
                             method = 'C5.0',
                             tuneLength = 5,
                             trControl = fitControl))
c50Ipca

##############################################\
# Model Type selection Feature Engineering ####
##############################################/

Gc50FEresample <- resamples(list(c50GRC=c50GRC,c50Gpca=c50Gpca))
summary(Gc50FEresample)
bwplot(Gc50FEresample)

Ic50FEresample <- resamples(list(c50IRC=c50IRC,c50Ipca=c50Ipca))
summary(Ic50FEresample)
bwplot(Ic50FEresample)

####################################################\
# Predict testSet/validation Feature Engineering ####
####################################################/

c50GRC_Pred <- predict(c50GRC, newdata = dfGRCTest)
postResample(c50GRC_Pred, dfGRCTest$galaxysentiment)
c50Gpca_Pred <- predict(c50Gpca, newdata = dfGpcaTest)
postResample(c50Gpca_Pred, dfGpcaTest$galaxysentiment)

c50IRC_Pred <- predict(c50IRC, newdata = dfIRCTest)
postResample(c50IRC_Pred, dfIRCTest$iphonesentiment)
c50Ipca_Pred <- predict(c50Ipca, newdata = dfIpcaTest)
postResample(c50Ipca_Pred, dfIpcaTest$iphonesentiment)

#---------Evaluation--------------#
#compare ones close to prior best results
confusionMatrix(c50GRC_Pred, dfGRCTest$galaxysentiment)
confusionMatrix(c50Gpca_Pred, dfGpcaTest$galaxysentiment)
confusionMatrix(c50IRC_Pred,dfIRCTest$iphonesentiment)
confusionMatrix(c50Ipca_Pred,dfIpcaTest$iphonesentiment)

saveRDS(c50GRC, "c50GRC_Galaxy_Model.rds")
saveRDS(c50IRC, "c50IRC_iPhone_Model.rds")

####################################################\
# Apply best model to Large Matrix               ####
####################################################/

# Preprocess
   # set up with best known feature selection / engineering

dfLMG$galaxysentiment<- as.factor(dfLMG$galaxysentiment)
dfLMI$iphonesentiment <- as.factor(dfLMI$iphonesentiment)

# Predict
c50GRCLM_Pred <- predict(c50GRC, newdata = dfLMG)
c50IRCLM_Pred <- predict(c50IRC, newdata = dfLMI)
summary(c50GRCLM_Pred)
summary(c50GRC_Pred)
#Gtotal_pred <- summary(c50GRCLM_Pred) +summary(c50GRC_Pred)
summary(c50IRCLM_Pred)
summary(c50IRC_Pred)
temp <- summary(c50GRCLM_Pred)

sum(c50GRCLM_Pred=="3")
# Plot the results 
# https://plotly.com/r/bar-charts/
# sum(mydata$sCode == "CA") - counts of each type.  also available from summary(..._Pred)
sum(dfGOOB$galaxysentiment == "5")
SentimentGroups <- c("Negative","Somewhat Negative","Somewhat Positive","Positive")
GalaxyLMSentiment <- c(sum(c50GRCLM_Pred == "1"),
                       sum(c50GRCLM_Pred == "2"),
                       sum(c50GRCLM_Pred == "3"),
                       sum(c50GRCLM_Pred == "4"))
iPhoneLMSentiment <- c(sum(c50IRCLM_Pred == "1"),
                       sum(c50IRCLM_Pred == "2"),
                       sum(c50IRCLM_Pred == "3"),
                       sum(c50IRCLM_Pred == "4"))
sentiment_df <- data.frame(SentimentGroups, GalaxyLMSentiment, iPhoneLMSentiment)
sentiment_df
fig <- plot_ly(sentiment_df, x = ~SentimentGroups, y = ~GalaxyLMSentiment, type = 'bar', name = 'Galaxy')
fig <- fig %>% add_trace(y = ~iPhoneLMSentiment, name = 'iPhone')
fig <- fig %>% layout(yaxis = list(title = 'Count'), barmode = 'group', 
                      xaxis = list(categoryarray = ~SentimentGroups, categoryorder = "array"))
fig

pieData <- data.frame(COM = c("negative", "somewhat negative", "somewhat positive","positive"), 
                      values = sentiment_df$GalaxyLMSentiment)

# create pie chart
plot_ly(pieData, labels = ~COM, values = ~ values, type = "pie",
              textposition = 'inside',
              textinfo = 'label+percent',
              insidetextfont = list(color = '#FFFFFF'),
              hoverinfo = 'text',
              text = ~paste( values),
              marker = list(colors = colors,
                            line = list(color = '#FFFFFF', width = 1)),
              showlegend = F) %>%
  layout(title = 'Galaxy Sentiment', 
         xaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
         yaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE))


# Combining original OOB RC data
GalaxySMSentiment <- c(sum(dfGRC$galaxysentiment == "1"),
                       sum(dfGRC$galaxysentiment == "2"),
                       sum(dfGRC$galaxysentiment == "3"),
                       sum(dfGRC$galaxysentiment == "4"))
iPhoneSMSentiment <- c(sum(dfIRC$iphonesentiment == "1"),
                       sum(dfIRC$iphonesentiment == "2"),
                       sum(dfIRC$iphonesentiment == "3"),
                       sum(dfIRC$iphonesentiment == "4"))
complete_sentiment_df <- data.frame(SentimentGroups, GalaxyLMSentiment, GalaxySMSentiment, iPhoneLMSentiment, iPhoneSMSentiment)
complete_sentiment_df





#normalize
GalaxyLMSentiment_norm <- GalaxyLMSentiment / (length(c50GRCLM_Pred)+nrow(dfGRC))
GalaxySMSentiment_norm <- GalaxySMSentiment / (length(c50GRCLM_Pred)+nrow(dfGRC))
iPhoneLMSentiment_norm <- iPhoneLMSentiment / (length(c50IRCLM_Pred)+nrow(dfIRC))
iPhoneSMSentiment_norm <- iPhoneSMSentiment / (length(c50IRCLM_Pred)+nrow(dfIRC))

norm_complete_sentiment_df <- data.frame(SentimentGroups, GalaxyLMSentiment_norm, 
                                         GalaxySMSentiment_norm, iPhoneLMSentiment_norm, iPhoneSMSentiment_norm)
norm_complete_sentiment_df

melted <- melt(norm_complete_sentiment_df, "SentimentGroups")
melted
melted$cat <- ''
melted[melted$variable == 'GalaxyLMSentiment_norm',]$cat <- "Galaxy"
melted[melted$variable == 'GalaxySMSentiment_norm',]$cat <- "Galaxy"
melted[melted$variable == 'iPhoneLMSentiment_norm',]$cat <- "iPhone"
melted[melted$variable == 'iPhoneSMSentiment_norm',]$cat <- "iPhone"
#fix the x-axis ordering
melted$SentimentGroups <- factor(melted$SentimentGroups, levels=c("Negative","Somewhat Negative","Somewhat Positive","Positive"))

ggplot(melted, aes(x = cat, y = value, fill = variable)) + 
  geom_bar(stat = 'identity', position = 'stack') + 
  facet_grid(~ SentimentGroups)+
  scale_fill_manual(name = "Data", 
                    values=c("#5c84ff", "#0a46ff", "#fd65de","#f60ac6"),
                    labels = c("Modeled", "Manual", "Modeled", "Manual"))

#normalized LM only#
GLMSentiment_norm <- GalaxyLMSentiment/length(c50GRCLM_Pred)
ILMSentiment_norm <- iPhoneLMSentiment/length(c50IRCLM_Pred)
sentiment_df_norm <- data.frame(SentimentGroups,GLMSentiment_norm , ILMSentiment_norm)
sentiment_df_norm
meltedLM <-  melt(sentiment_df_norm, "SentimentGroups")
meltedLM
meltedLM$cat <- ''
meltedLM[meltedLM$variable == 'GLMSentiment_norm',]$cat <- "Galaxy"
meltedLM[meltedLM$variable == 'ILMSentiment_norm',]$cat <- "iPhone"
meltedLM$SentimentGroups <- factor(meltedLM$SentimentGroups, levels=c("Negative","Somewhat Negative","Somewhat Positive","Positive"))
ggplot(meltedLM, aes(x = cat, y = value, fill = variable)) + 
  geom_bar(stat = 'identity', position = 'stack', show.legend=FALSE) + 
  facet_grid(~ SentimentGroups)+
  scale_fill_manual(name = "Data", 
                    values=c("#5c84ff", "#fd65de"),
                    labels = c("Modeled", "Modeled"))



