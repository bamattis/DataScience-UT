# Title: C4T3 - Indoor Wifi positioning

# Last update: 7/2/2021
# Author: Brian Mattis
# File: C4T3_Pipeline.R


###################\
# Project Notes ####
###################/

# Summarize project: Investiaget using "wifi fingerprinting" to determine
# a person's location in indoor spaces.  This uses the signal strength from 
# multiple Wireless Access Points (WAPs) to determine location through a
# data science model.

# Outputs:
# 1) Written report with the comparison of models produced by 3 different
#    algorithms.  Show Kappa and Accuracy in a single chart to help evaluation (bar)
# 2) Recommendation on algorithm
# 3) Recommendations for how to further improve

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
library(caret)
library(corrplot)
library(readr)
library(mlbench)
library(doParallel)             # for Win
library(data.table)
library(dplyr)
library(DataExplorer)
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

# Data descriptions
	# DV: Location
	# each node: +100 - WAP not detected
	          #: 0 - strong WAP signal
	          #: -100 - weak WAP signal detected

dfOOB <- read.csv("data/trainingData.csv", stringsAsFactors = FALSE)
str(dfOOB)

dfValOOB <- read.csv("data/validationData.csv", stringsAsFactors = FALSE)
str(dfValOOB)
#Validation data set only has values for Building/Floor
# no values for SpaceID / RelativeID, so can only be used
# if we're using to model Building/Floor only

####################\
# Evaluate data #####
####################/

##--- Training Data ---##
# check for missing values 
anyNA(dfOOB) # FALSE

# check for duplicates
anyDuplicated((dfOOB))
# 2908 - this is perhaps because several readings were made with each phone?
# but with timestamp this shouldn't be possible.

#find grouped matched duplicate items
as.data.table(dfOOB)[, c("GRP", "N") := .(.GRP, .N), by = names(dfOOB)][
  N > 1, list(list(.I)), by = GRP]
#    GRP                V1
#1:  2335         2335,3216
#2:  2810         2810,2908
dfOOB[2335,]$TIMESTAMP  #[1] 1371719975
dfOOB[3216,]$TIMESTAMP  #[1] 1371719975
#Exact same time stamp!  Indeed duplicates!  let's remove them.
df <- dfOOB[!duplicated(dfOOB), ]
anyDuplicated((df)) # 0


##--- Validation Data ---##
# check for missing values 
anyNA(dfValOOB) # FALSE

anyDuplicated((dfValOOB)) # 0
dfVal <- dfValOOB

#################\
# Preprocess #####
#################/
str(df, list.len=ncol(df))

#factorization
df$FLOOR <- as.factor(df$FLOOR)
df$BUILDINGID <- as.factor(df$BUILDINGID)
df$SPACEID <- as.factor(df$SPACEID)
df$RELATIVEPOSITION <- as.factor(df$RELATIVEPOSITION)
df$USERID <- as.factor(df$USERID)
df$PHONEID <- as.factor(df$PHONEID)

# Turn unix date into readable Time
df$TIMESTAMP <- as.POSIXct(df$TIMESTAMP, origin="1970-01-01")
df$TIMESTAMP # "2013-06-20 02:35:33 CDT"

#combine into single DV for location
df$Location <- paste("B",df$BUILDINGID," F",df$FLOOR, 
                     " S",df$SPACEID, " R", df$RELATIVEPOSITION, sep="")
df$Location <- as.factor(df$Location)

#remove the original columns, so it doesn't train based on them
df$BUILDINGID <- NULL
df$FLOOR <- NULL
df$SPACEID <- NULL
df$RELATIVEPOSITION <- NULL

#########################\
# EDA/Visualizations #####
#########################/

plot_intro(df) #data health
plot_bar(df) #shows your categorical variables
  # shows that several WAPs don't actually have any non-100 values
plot_correlation(df) #giant correlation plot that does dummify for you (temporarily, just for the plot)



# statistics
summary(df)
#let's find the distribution of the non-100 values (real data)
idx <- which(df$WAP502<100)
summary(df$WAP502[idx])
#great! now let's see it for all WAP columns
allSignals <- NULL
#let's do it for all
for(i in 1:520){
  #append the current cols values <100 to the list
  allSignals <- append(allSignals, df[df[[i]]<100,][[i]])
}
allSignals

# plots
hist(allSignals, col="steelblue3", main="Histogram of all WAPs", xlab="Signal Strength (dB)")
hist(df$LATITUDE)
hist(df$LONGITUDE)
plot(df$LONGITUDE, df$LATITUDE)
qqnorm(df$LONGITUDE) 



################\
# Sampling  #####
################/

# Plan: use random sample for model selection, then run that model on 
# each building separately.

# create 25% sample 
set.seed(123) # set random seed
?createDataPartition
dfSamp_index <- createDataPartition(df$Location, p=0.25,list=FALSE)
dfSamp <- df[dfSamp_index,]
dfSamp <- droplevels(dfSamp)
nrow(dfSamp) #5047
levels(dfSamp$Location)
levels(df$Location)

#create a data set for only one building
dfB0 <- filter(df, substr(df$Location,0,2)=="B0") #5246 rows
dfB0 <- droplevels(dfB0)
levels(dfB0$Location) #257

#------\
#It could be more useful for the business problem to ignore the $RELATIVEPOSITION
# as part of the DV, since functionally getting our user the correct $SPACEID
# regardless of whether they're inside or outside the room should be good enough
#------/
# create a data set that ignores RELATIVEID and DV is just building/floor/SPace
dfSp <- df
dfSp$Location <- as.character(dfSp$Location)
nchar(dfSp$Location[1])
dfSp$Location <- substr(dfSp$Location, 0, (nchar(dfSp$Location)-3))
dfSp$Location <- as.factor(dfSp$Location) 
dfSp_Samp_index <- createDataPartition(dfSp$Location, p=0.25,list=FALSE)
dfSp_Samp <- dfSp[dfSamp_index,]
dfSp_Samp <- droplevels(dfSp_Samp)
nrow(dfSp_Samp) #5047

str(dfSp, list.len=ncol(dfSp))

#######################\
# Feature selection ####
#######################/

#TIMESTAMP really shouldn't have a bearing on signal strength, and this isn't
# a time series, so we likely wouldn't use it in the future
df$TIMESTAMP <- NULL
dfSamp$TIMESTAMP <- NULL
dfSp_Samp$TIMESTAMP <- NULL


##############################\
# Feature engineering #########
##############################/
str(df, list.len=ncol(df))
#Goal: create variables with highest signal strength value and WAP index
slice_max(df[1,],3)
df_dummy <- df[,1:520] #only the WAP columns
df_dummy[df_dummy==100] <- -101 # set the 100s to -101 so that they are not the max
slice_max(df_dummy[1,],3)
max_sig <- apply(df_dummy[1,],1,max) #find the max WAP value in the first row

max_sig #-53
max.col(df_dummy[1,]) #173 - column location of the highest value
colnames(df_dummy)[173]  #[1] "WAP173"
tail(sort(df_dummy[,1])) # highest values in column 1 (WAP001)

#let's apply this method to get the max WAP value and WAP# for each row
df_FE <- df[,521:525]
#highest signal strength
df_FE$sig1 <- apply(df_dummy,1,max)
#WAP of highest signal strength
df_FE$sig1_WAP <- max.col(df_dummy)
head(df_FE)

#Expand: let's do the top 3 ranked signal intensity node numbers and WAP values

tail(sort(as.numeric(df_dummy[1,])),3) #show the 3 highest values in row 1

fout <- NULL
top3_val_func <- function(i,j){
  fout <- tail(sort(as.numeric(df_dummy[i,])),3)[j]
  return(fout)
}
dftemp <- NULL
for(i in 1:nrow(df_dummy)){
  #dftemp$sig2 <- apply(df_dummy,1,top3_val_func(i,2))
  df_FE$sig2[i] <-top3_val_func(i,2)
  df_FE$sig2_WAP[i] <- which(as.numeric(df_dummy[i,])==df_FE$sig2[i])
  df_FE$sig3[i] <-top3_val_func(i,1)
  df_FE$sig3_WAP[i] <- which(as.numeric(df_dummy[i,])==df_FE$sig3[i])
}
head(df_FE)
#WAP ID's should be factors
df_FE$sig1_WAP <- as.factor(df_FE$sig1_WAP)
df_FE$sig2_WAP <- as.factor(df_FE$sig2_WAP)
df_FE$sig3_WAP <- as.factor(df_FE$sig3_WAP)

#reorder the df_FE columns
df_FE <- df_FE[,c(6,7,8,9,10,11,1,2,3,4,5)]
str(df_FE)
summary(df_FE) #keep in mind, 100s are still all as -101.  probably doesn't matter

#how many rows never show any WAP signal?
nrow(df_FE[df_FE$sig1==-101,]) #73
nrow(df_FE[df_FE$sig2==-101,]) #80
nrow(df_FE[df_FE$sig3==-101,]) #119

dfFE_index <- createDataPartition(df_FE$Location, p=0.50,list=FALSE)
dfFEmini <- df[dfFE_index,]
dfFEmini <- droplevels(dfFEmini)
nrow(dfFEmini) #5047
inTraining <- createDataPartition(dfFEmini$Location, p=0.75, list=FALSE)
dfFEminiTrain <- dfFEmini[inTraining,]   
dfFEminiTest <- dfFEmini[-inTraining,] 
dfFEminiTrain$Location <- droplevels(dfFEminiTrain$Location)
#also consider if we just did:
# stack WAPs with "gather", then filter to show those < 100
#only get one per row, but that could be okay?  Perhaps investigate if all models are poor

#####################\
# Train/test sets ####
#####################/

#dfSamp data set
set.seed(123) 
inTraining <- createDataPartition(dfSamp$Location, p=0.75, list=FALSE)
dfSampTrain <- dfSamp[inTraining,]   
dfSampTest <- dfSamp[-inTraining,] 
# verify number of obs 
nrow(dfSampTrain) # 4117
nrow(dfSampTest)  # 930

#dfSp_Samp data set (DV=$Location=building/floor/SpaceID)
inTraining <- createDataPartition(dfSp_Samp$Location, p=0.75, list=FALSE)
dfSp_SampTrain <- dfSp_Samp[inTraining,]   
dfSp_SampTest <- dfSp_Samp[-inTraining,] 
# verify number of obs 
nrow(dfSp_SampTrain) # 4032
nrow(dfSp_SampTest)  # 1015

# df_FE
inTraining <- createDataPartition(df_FE$Location, p=0.75, list=FALSE)
df_FETrain <- df_FE[inTraining,]   
df_FETest <- df_FE[-inTraining,]   
# verify number of obs 
nrow(df_FETrain) # 14543
nrow(df_FETest)  # 4757

###################\
# Train control ####
###################/

# set cross validation
fitControl <- trainControl(method="repeatedcv", number=5, repeats=1, verboseIter = TRUE) 

##################\
# Train models ####
##################/

?modelLookup()
modelLookup('rf')

## ------- KNN ------- ##

fitControl <- trainControl(method="repeatedcv", number=10, repeats=1, verboseIter = TRUE) 
#dfSamp data set
set.seed(123)
system.time(knn_Samp <- train(Location ~.,
                              data = dfSampTrain,
                              method = 'knn',
                              tuneLength = 20,
                              trControl = fitControl))
knn_Samp
plot(knn_Samp)

#dfSp_Samp data set (DV=$Location=building/floor/SpaceID)
system.time(knn_Sp_Samp <- train(Location ~.,
                              data = dfSp_SampTrain,
                              method = 'knn',
                              tuneLength = 20,
                              trControl = fitControl))
knn_Sp_Samp
plot(knn_Sp_Samp)

# df_FE data set
system.time(knn_FE <- train(Location ~.,
                                 data = df_FETrain,
                                 method = 'knn',
                                 tuneLength = 4,
                                 trControl = fitControl))
knn_FE
plot(knn_FE)


## ------- RF ------- ##

## training model with Random Forest, with 10 fold CV,tuneLength = 5, on sub1_train
fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 1, 
                           verboseIter = TRUE)

system.time(rfSamp <- train(Location ~.,
                             data = dfSampTrain,
                             method = 'rf',
                             tuneLength = 5,
                             trControl = fitControl))
rfSamp

plot(rfSamp)
fitControl <- trainControl(method = "repeatedcv",
                           number = 5,
                           repeats = 1, 
                           verboseIter = TRUE)

system.time(rfSp_Samp <- train(Location ~.,
                               data = dfSp_SampTrain,
                               method = 'rf',
                               tuneLength = 3,
                               trControl = fitControl))
rfSp_Samp

rfFEGrid <- expand.grid(mtry=c(2,4,7,10)) #since only 10 possible variables
system.time(rfFEmini <- train(Location ~.,
                              data = dfFEminiTrain,
                              method = 'rf',
                              tuneGrid = rfFEGrid,
                              trControl = fitControl))
rfFEmini

## ------- C50 ------- ##

fitControl <- trainControl(method = "repeatedcv",
                           number = 5,
                           repeats = 1, 
                           verboseIter = TRUE)

system.time(c50Samp <- train(Location ~.,
                              data = dfSampTrain,
                              method = 'C5.0',
                              tuneLength = 3,
                              trControl = fitControl))
c50Samp

system.time(c50Sp_Samp <- train(Location ~.,
                             data = dfSp_SampTrain,
                             method = 'C5.0',
                             tuneLength = 3,
                             trControl = fitControl))
c50Sp_Samp

system.time(c50_FE <- train(Location ~.,
                                data = df_FETrain,
                                method = 'C5.0',
                                tuneLength = 3,
                                trControl = fitControl))
c50_FE

#####################\
# Model selection ####
#####################/
KNNresample <- resamples(list(Samp=knn_Samp, Sp_Samp=knn_Sp_Samp, FE=knn_FE))
summary(KNNresample)
bwplot(KNNresample)

#have to group the rfSamp with the KNNs, since 10 folds were used
rfSamp_resample <- resamples(list(kSamp=knn_Samp, Samp=rfSamp))
summary(rfSamp_resample)

rfresample <- resamples(list(Sp_Samp=rfSp_Samp, FE=rfFEmini))
summary(rfresample)
bwplot(rfresample)

c50resample <- resamples(list(Samp=c50Samp, Sp_Samp=c50Sp_Samp, FE=c50_FE))
summary(c50resample)
bwplot(c50resample)


################################\
# Predict testSet/validation ####
################################/

knnSamp_Pred <- predict(knn_Samp, newdata = dfSampTest)
postResample(knnSamp_Pred, dfSampTest$Location)

knnSp_Samp_Pred <- predict(knn_Sp_Samp, newdata = dfSp_SampTest)
postResample(knnSp_Samp_Pred, dfSp_SampTest$Location)

knnFE_Pred <- predict(knn_FE, newdata = df_FETest)
postResample(knnFE_Pred, df_FETest$Location)

rfSamp_Pred <- predict(rfSamp, newdata = dfSampTest)
postResample(rfSamp_Pred, dfSampTest$Location)

rfSp_Samp_Pred <- predict(rfSp_Samp, newdata = dfSp_SampTest)
postResample(rfSp_Samp_Pred, dfSp_SampTest$Location)

rfFE_Pred <- predict(rfFEmini, newdata = dfFEminiTest)
postResample(rfFE_Pred, dfFEminiTest$Location)

c50Samp_Pred <- predict(c50Samp, newdata = dfSampTest)
postResample(c50Samp_Pred, dfSampTest$Location)

c50Sp_Samp_Pred <- predict(c50Sp_Samp, newdata = dfSp_SampTest)
postResample(c50Sp_Samp_Pred, dfSp_SampTest$Location)

c50FE_Pred <- predict(c50_FE, newdata = df_FETest)
postResample(c50FE_Pred, df_FETest$Location)

confusionMatrix(data = knnSamp_Pred, reference = dfSampTest$Location, mode = "prec_recall")
confusionMatrix(data = knnFE_Pred, reference = df_FETest$Location, mode = "prec_recall")
