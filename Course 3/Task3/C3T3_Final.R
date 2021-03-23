# DataScience Course 3, Task 3.

# Goal: use regression to predict the volumes of products on a complete data set  
    # Then, use this model on a data set with unknown volumes

# Secondary Goal: Assess the impact services reviews and customer 
    # reviews have on sales of different product types

##############
# Import data 
##############

# Standard Libraries:
library(caret)
library(corrplot)
library(readr)

dfOOB <- read.csv("data/existingproductattributes2017.csv",  stringsAsFactors = FALSE)

df_New_Products <- read.csv("data/newproductattributes2017.csv",  stringsAsFactors = FALSE)

str(dfOOB)

#############
# Preprocess
#############

#check for duplicate values
anyDuplicated((dfOOB))
# [1] 0  -- All good!

#check for missing 
anyNA(dfOOB)
# [1] TRUE
#look for which columns have the NA values
colSums(is.na(dfOOB))
# All in BestSellersRank.  15 out of 80 rows
is.na(dfOOB$BestSellersRank)
which(is.na(dfOOB$BestSellersRank))
#there's too limited of a data set to just delete these rows because of one column
# decision: Setting to zero could be distruptive since a rank of 1 denotes 
# a popular product.  Regardless, a model that bases importance on this variable 
# could be problematic (VarImp).  Will set to column mean as INT
summary(dfOOB$BestSellersRank)
#    Min. 1st Qu.  Median    Mean   3rd Qu.    Max.      NA's 
#       1       7      27    1126     281      17502      15 
plot(dfOOB$BestSellersRank, dfOOB$Volume, xlim=c(0,1200), ylim=c(0,400))
# yes, this value looks reasonable and shouldn't disrupt much

# set the NA values to the (rounded) column mean value
dfOOB$BestSellersRank[is.na(dfOOB$BestSellersRank)] <- round(mean(dfOOB$BestSellersRank, na.rm=TRUE), digits=0)
anyNA(dfOOB)
# FALSE

#Dummification
#this breaks out the ProductType into seperate binary columns
newDataFrame <- dummyVars(" ~ .", data = dfOOB)
df <- data.frame(predict(newDataFrame, newdata = dfOOB))

# Ah, teachers notes now tell us to delete columns with NA data.  Bye bye $BestSellersRank
df$BestSellersRank <- NULL

#######################
# Correlation analysis
#######################

# Creating a Tuned data set (compare to orig in modeling)
# Rule#1: If indep to dep  corr > 0.95, don’t use that indep in the modeling.
#
# Rule#2: If two independents corr >0.9 to eachother, get rid of the 
#         one with lower correlation to the dependent

corrData <- cor(df)
corrData
corrplot(corrData)
# make a corr plot of just reviews/service/volume
df[,c(15:21, 28)]
corrStarServ <- cor(df[,c(15:21, 28)])
corrplot(corrStarServ)
write.csv(corrStarServ, "CorrStarServ.csv", row.names=T)

#check for overly high correlations from IV to DV
cor(df[-1], df$Volume)
##ummm... x5StarReviews is 100% correlated.  That's not good.
plot(df$x5StarReviews, df$Volume)
#remove for "Tuned data set"
dfCorr <- df
dfCorr$x5StarReviews <- NULL


#look for high correlations in the independent variables
corrIV <- cor(df[,1:27])
corrIVhigh <- findCorrelation(corrIV, cutoff=0.9)   
# print indexes of highly correlated attributes
corrIVhigh
# [1] 17 18
# get var name of high corr IV
colnames(df[c(corrIVhigh)]) 
# [1] "x3StarReviews" "x2StarReviews".  They have a 0.93 correlation
#remove for "Tuned data set".  Remove the one with lower correlation to DV
cor(df[-1], df$Volume)
#x3StarReviews                0.763373189
#x2StarReviews                0.487279328
dfCorr$x2StarReviews <- NULL

############
# caret RFE 
############
# define the control using a random forest selection function (regression or classification)
RFcontrol <- rfeControl(functions=rfFuncs, method="cv", number=5, repeats=1)
LMcontrol <- rfeControl(functions=lmFuncs, method="cv", number=5, repeats=1)

# run the RFE algorithm
set.seed(7)
# rfe (independent v, dv, etc...)
rfeRF <- rfe(df[,1:27], df[,28], sizes=c(1:28), rfeControl=RFcontrol)
rfeRF 
# Variables  RMSE    Rsquared   MAE    RMSESD   RsquaredSD MAESD Selected
# 1          365.1   0.9873     105.5  663.9    0.02299    176.2        *
plot(rfeRF, type=c("g", "o"))
# show predictors used
predictors(rfeRF)
# [1] "x5StarReviews" 
varImp(rfeRF)
# Overall
# x5StarReviews 14.4437
# not a surprising result considering the 100% correlation of x5StarReviews

####  lm ##########
# rfe (independent v, dv, etc...)
rfeLM <- rfe(df[,1:27], df[,28], sizes=c(1:28), rfeControl=LMcontrol)
rfeLM
plot(rfeLM, type=c("g", "o"))
predictors(rfeLM)
#R^2 = 1 because direct 1.0 correlation to x5starreviews
# Variables   RMSE       Rsquared   MAE        RMSESD    RsquaredSD MAESD       Selected
#         4   2.515e-13      1      1.782e-13 2.264e-13          0  1.694e-13      *
varImp(rfeLM)
# x5StarReviews               4.000000e+00
# ProductTypeGameConsole      2.606736e-12
# ProfitMargin                2.460732e-12
#would limit df to x5starReviews - just like rfeRF

#############
## 2nd round, remove x5StarReviews (Nx5 = "no 5-star review attribute")
dfNx5 <- df
#remove it here and find what key remaining variables are
dfNx5$x5StarReviews <- NULL
rfeNx5RF <- rfe(dfNx5[,1:26], dfNx5[,27], sizes=c(1:27), rfeControl=RFcontrol)
rfeNx5RF
# Variables   RMSE   Rsquared   MAE   RMSESD   RsquaredSD MAESD Selected
# 2           840.0   0.8561    283.0  715.3    0.09187   217.8        *
predictors(rfeNx5RF)
varImp(rfeNx5RF)
# PositiveServiceReview 15.912155
# x4StarReviews          9.435332
rfeNx5LM <- rfe(dfNx5[,13:26], dfNx5[,27], sizes=c(1:14), rfeControl=LMcontrol)
predictors(rfeNx5LM)
varImp(rfeNx5LM)
#Overall
#ProfitMargin          599.67999
#Recommendproduct      204.19402
#NegativeServiceReview  53.54237
#x3StarReviews          51.24364
#ProductWidth           50.07855
#x2StarReviews          44.77885
#x4StarReviews          36.42763

rfeNx5LM
# Variables   RMSE   Rsquared   MAE   RMSESD   RsquaredSD MAESD Selected
# 6           840.2  0.63752   498.3  385.4    0.38302    187.7        *

# will use RF in feature reduction as R^2 is low and RMSE is high




##############################
# Feature engineering
# https://www.datavedas.com/feature-engineering-in-r/
##############################

####  Feature Construction #####
dfFE <- df
#build an average star score
dfFE$starScore <- ((dfFE$x5StarReviews*5) + (dfFE$x4StarReviews*4) + 
                    (dfFE$x3StarReviews*3) + (dfFE$x2StarReviews*2) + (dfFE$x1StarReviews*1))/
  (dfFE$x5StarReviews + dfFE$x4StarReviews + dfFE$x3StarReviews + 
     dfFE$x2StarReviews + dfFE$x1StarReviews)
#build an average service score
dfFE$servScore <- dfFE$PositiveServiceReview / 
  (dfFE$PositiveServiceReview + dfFE$NegativeServiceReview)
#set NANs to 0
dfFE$servScore[is.na(dfFE$servScore)] <- 0
anyNA(dfOOB)
# I'll also remove the existing star scores and service scores, since they're duplicating info
dfFE <- dfFE[-c(15:21)]
grep("x5StarReviews", colnames(dfFE))
plot(dfFE$starScore, dfFE$Volume)
plot(dfFE$servScore, dfFE$Volume)


#### #Feature Reduction ########
dfrfe <- data.frame(matrix(df[,predictors(rfeRF)], nrow=80, byrow=TRUE),stringsAsFactors=FALSE)
dfrfe <- df[,predictors(rfeRF)]
dfrfe$Volume <- df$Volume
str(dfrfe)
names(dfrfe) <- c("x5StarReviews", "Volume")

dfNx5rfe <- dfNx5[,predictors(rfeNx5RF)]
dfNx5rfe$Volume <- dfNx5$Volume

##################
# Train/test sets
##################
# df - ABANDON
# dfCorr (no x5 or x2)
# dfFE (no x5 + new vars)
# dfrfe (x5 only) - ABANDON
# dfNx5 (no x5, otherwise df)
# dfNx5 with rfe(2 IVs)

set.seed(123) 
inTraining <- createDataPartition(df$Volume, p=0.75, list=FALSE)
#all have 80 observations, so can use inTraining for all
dfTrain <- df[inTraining,]   
dfTest <- df[-inTraining,]   

dfCorrTrain <- dfCorr[inTraining,]   
dfCorrTest <- dfCorr[-inTraining,]  

dfFETrain <- dfFE[inTraining,]   
dfFETest <- dfFE[-inTraining,]  

dfrfeTrain <- dfrfe[inTraining,]   
dfrfeTest <- dfrfe[-inTraining,] 

dfNx5Train <- dfNx5[inTraining,]   
dfNx5Test <- dfNx5[-inTraining,] 

dfNx5rfeTrain <- dfNx5rfe[inTraining,]   
dfNx5rfeTest <- dfNx5rfe[-inTraining,] 

################
# Train control
################

# set cross validation - 10 folds desired
fitControl <- trainControl(method="repeatedcv", number=10, repeats=1) 


###############
# Train models
###############

dfLMfit <- train(Volume~., data=dfTrain, method="lm", trControl=fitControl)
#Warning messages:
#  1: In predict.lm(modelFit, newdata) :
#  prediction from a rank-deficient fit may be misleading
summary(dfLMfit)
# Warning message:
#  In summary.lm(object$finalModel, ...) :
#  essentially perfect fit: summary may be unreliable
dfLMfit
# RMSE          Rsquared  MAE         
# 8.026183e-13  1         5.545337e-13
#essentially a perfect fit.. Volume = 3* x5StarReviews
# we can't use data sets with x5StarReviews to build a model

#PLAN: 
# - remove x5 from df, rename to dfNx5 rerun
# - redo rfe on dfNx5

dfLMfit <- train(Volume~., data=dfNx5Train, method="lm", trControl=fitControl)
summary(dfLMfit)
dfLMfit
# RMSE    Rsquared   MAE     
# 810.38  0.7339729  485.1641

## Next step: run the various data sets with SVM, RF, and GBM models (stock)

Nx5SVMfit <- train(Volume~., data=dfNx5Train, method="svmLinear", trControl=fitControl)
Nx5GBMfit <- train(Volume~., data=dfNx5Train, method="gbm", trControl=fitControl)
Nx5RFfit <- train(Volume~., data=dfNx5Train, method="rf", trControl=fitControl)

CorrSVMfit <- train(Volume~., data=dfCorrTrain, method="svmLinear", trControl=fitControl)
CorrGBMfit <- train(Volume~., data=dfCorrTrain, method="gbm", trControl=fitControl)
CorrRFfit <- train(Volume~., data=dfCorrTrain, method="rf", trControl=fitControl)

FESVMfit <- train(Volume~., data=dfFETrain, method="svmLinear", trControl=fitControl)
FEGBMfit <- train(Volume~., data=dfFETrain, method="gbm", trControl=fitControl)
FERFfit <- train(Volume~., data=dfFETrain, method="rf", trControl=fitControl)

Nx5rfeSVMfit <- train(Volume~., data=dfNx5rfeTrain, method="svmLinear", trControl=fitControl)
Nx5rfeGBMfit <- train(Volume~., data=dfNx5rfeTrain, method="gbm", trControl=fitControl)
Nx5rfeRFfit <- train(Volume~., data=dfNx5rfeTrain, method="rf", trControl=fitControl)

##################
# Model selection
##################

SVMfits <- resamples(list(svmNx5=Nx5SVMfit, svmCorr=CorrSVMfit, svmFE=FESVMfit, svmNx5rfe=Nx5rfeSVMfit))
# output summary
summary(SVMfits)
#RMSE 
#             Min.    1st Qu.   Median      Mean   3rd Qu.     Max. NA's
# svmNx5    126.6349 223.0968 397.2114  743.1073  717.2829 3411.103    0
# svmCorr   150.0035 418.4849 702.3760  686.6013  988.5740 1099.173    0
# svmFE     256.9209 494.4110 681.4238 1268.9149 1919.4335 4319.257    0
# svmNx5rfe 163.0183 256.3910 431.5376  760.6530 1189.1620 2098.562    0

#Rsquared 
#                 Min.    1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#svmNx5    0.570228930 0.78169500 0.8766624 0.8443279 0.9456686 0.9822480    0
#svmCorr   0.035838630 0.71790915 0.8631081 0.7217399 0.9384808 0.9952632    0
#svmFE     0.004428442 0.04578309 0.4090101 0.3950229 0.7041831 0.8377532    0
#svmNx5rfe 0.536370232 0.86493491 0.9459968 0.8993420 0.9883488 0.9976948    0

GBMfits <- resamples(list(gbmNx5=Nx5GBMfit, gbmCorr=CorrGBMfit, gbmFE=FEGBMfit, gbmNx5rfe=Nx5rfeGBMfit))
# output summary
summary(GBMfits)
#RMSE 
#              Min.  1st Qu.   Median      Mean  3rd Qu.     Max. NA's
#gbmNx5    306.3627 336.4591 427.3958  965.6269 536.9538 4311.703    0
#gbmCorr   235.0750 281.4011 486.9198 1073.4017 733.5749 4846.080    0
#gbmFE     446.3848 554.0925 741.9898 1230.8935 960.8546 3914.843    0
#gbmNx5rfe 255.5862 313.3999 439.5973  893.6406 836.6963 3412.696    0

#Rsquared 
#          Min.        1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#gbmNx5    0.38816457 0.6688135 0.8140648 0.7496367 0.9059831 0.9532864    0
#gbmCorr   0.41967096 0.6434906 0.8635613 0.7752458 0.9273451 0.9546931    0
#gbmFE     0.08602415 0.1633968 0.2927232 0.3176054 0.3622451 0.9020723    0
#gbmNx5rfe 0.50227595 0.6423597 0.8933280 0.8110944 0.9389130 0.9880810    0

RFfits <- resamples(list(rfNx5=Nx5RFfit, rfCorr=CorrRFfit, rfFE=FERFfit, rfNx5rfe=Nx5rfeRFfit))
# output summary
summary(RFfits)
#RMSE 
#              Min.  1st Qu.   Median      Mean  3rd Qu.     Max.    NA's
#rfNx5     64.33598 107.19938 221.7803  746.4600 1203.0748 2788.220    0
#rfCorr    67.65924 172.08365 244.1500  723.2769 1244.6424 2718.937    0
#rfFE     282.68972 574.80420 738.2718 1170.7819  923.0011 5524.187    0
#rfNx5rfe  33.12237  56.74854 110.5123  541.2870  320.6308 2368.492    0

#Rsquared 
#               Min.    1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#rfNx5    0.44597794 0.75345487 0.9457267 0.8573254 0.9924033 0.9955957    0
#rfCorr   0.79680611 0.86701219 0.9538262 0.9246617 0.9763286 0.9985232    0
#rfFE     0.02575132 0.07545297 0.1542784 0.3559635 0.6485632 0.9526483    0
#rfNx5rfe 0.71484789 0.87335517 0.9711358 0.9238941 0.9928530 0.9973074    0

## RF with the Nx5rfe data set is the best so far.  Amazing considering it only uses two IVs
## Consider trying for model optimization

#####################
# Model Optimization
####################

#MANUAL GRID
rfGrid <- expand.grid(mtry=c(1,2,3,4,5))
Nx5rfeRFOptfit <- train(Volume~., data=dfNx5rfeTrain, method="rf", 
                        trControl=fitControl,
                        tuneGrid=rfGrid)
Nx5rfeRFOptfit 
plot(Nx5rfeRFOptfit)
#mtry  RMSE      Rsquared   MAE     
#1     657.1423  0.9515293  319.5370
#2     608.2148  0.9512731  297.6882
#3     608.7653  0.9522991  298.4844
#4     605.4546  0.9531272  297.3035  <----
#5     606.9752  0.9527503  297.6629

#RF RANDOM SEARCH
rfitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 1, search = 'random')
Nx5rfeRFOptRfit <- train(Volume~., data=dfNx5rfeTrain, method="rf", 
                        trControl=rfitControl)
Nx5rfeRFOptRfit
# RMSE      Rsquared   MAE     
# 593.3431  0.9203816  270.7603

#compare to the previous winner with rsamples to prevent overfitting
RFfitOpt <- resamples(list(rfNx5rfe=Nx5rfeRFfit, rfOpt=Nx5rfeRFOptfit, rfOprRandom=Nx5rfeRFOptRfit))
summary(RFfitOpt)
#RMSE 
#                Min.  1st Qu.   Median     Mean  3rd Qu.     Max.  NA's
#rfNx5rfe    33.12237 56.74854 110.5123 541.2870 320.6308 2368.492    0
#rfOpt       37.18554 64.35891 128.7624 547.2421 356.2354 2670.653    0
#rfOprRandom 38.69356 158.38924 214.2956 593.3431 350.4208 2522.755    0

#Rsquared 
#                 Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#rfNx5rfe    0.7148479 0.8733552 0.9711358 0.9238941 0.9928530 0.9973074    0
#rfOpt       0.6994161 0.9257591 0.9919498 0.9418458 0.9946163 0.9970919    0
#rfOprRandom 0.8472728 0.8763959 0.9150764 0.9203816 0.9742623 0.9959529    0
# ToDo: do predict with each of the model types to make sure they don't suddenly get better

############################
# Predict testSet/validation
############################
#check predicted values for both Nx5rfeRFfit and Nx5rfeRFOptfit to make sure
# it's not trying to predict negative values

rfPredNx5rfe <- predict(Nx5rfeRFfit, dfNx5rfeTest)
# performance measurement
postResample(rfPredNx5rfe, dfNx5rfeTest$Volume)
# RMSE        Rsquared        MAE 
#899.1651533   0.6403254 303.6224536 
# plot predicted verses actual
plot(rfPredNx5rfe, dfNx5rfeTest$Volume)
abline(a=0, b=1, col="blue")

rfPredNx5Opt <- predict(Nx5rfeRFOptfit, dfNx5rfeTest)
# performance measurement
postResample(rfPredNx5Opt, dfNx5rfeTest$Volume)
# RMSE        Rsquared        MAE 
#918.8679925   0.6385883 306.9319240 
# plot predicted verses actual
plot(rfPredNx5Opt, dfNx5rfeTest$Volume)
#plot(rfPredNx5Opt, dfNx5rfeTest$Volume, xlim=c(-100,500), ylim=c(1,500))
abline(a=0, b=1, col="blue")
# That was a much worse fit that expected from Train modeling.

#let's check the one with more columns
rfPredNx5 <- predict(Nx5RFfit, dfNx5Test)
postResample(rfPredNx5, dfNx5rfeTest$Volume)
# RMSE          Rsquared         MAE 
# 989.0797247   0.6139403 332.8928561 
#slighly worse, as it was before.  ruling out this model

##########
# Plan - go back and eval our good SVM/GBM models 
# - Nx5SVMfit , Nx5rfeSVMfit, CorrGBMfit, Nx5rfeGBMfit
svmPredNx5 <- predict(Nx5SVMfit, dfNx5Test)
postResample(svmPredNx5, dfNx5Test$Volume)
# RMSE     Rsquared          MAE 
# 1228.9807198    0.7025931  659.5561832
plot(svmPredNx5, dfNx5Test$Volume)
abline(a=0, b=1, col="blue")
#svmPredNx5 returns two negtive values - not great!

svmPredNx5rfe <- predict(Nx5rfeSVMfit, dfNx5rfeTest)
postResample(svmPredNx5rfe, dfNx5rfeTest$Volume)
# RMSE          Rsquared          MAE 
# 919.6155420   0.5656202    539.1271346 
plot(svmPredNx5rfe, dfNx5rfeTest$Volume)
abline(a=0, b=1, col="blue")
# Low R^2 AND negative values - quite bad.

# CorrGBMfit
gbmPredCorr <- predict(CorrGBMfit, dfCorrTest)
postResample(gbmPredCorr, dfCorrTest$Volume)
# RMSE          Rsquared         MAE 
# 496.9910715   0.8380456 395.9673147 
plot(gbmPredCorr, dfCorrTest$Volume)
abline(a=0, b=1, col="blue")
# great R^2 but predicts lots of negatives

# Nx5rfeGBMfit
gbmPredNx5rfe <- predict(Nx5rfeGBMfit, dfNx5rfeTest)
postResample(gbmPredNx5rfe, dfNx5rfeTest$Volume)
# RMSE    Rsquared         MAE 
# 592.0401902   0.8083254 344.5652442 
plot(gbmPredNx5rfe, dfNx5rfeTest$Volume)
abline(a=0, b=1, col="blue")
# no negatives, decent 

# Combine all of the outputs for evaluation
dfTestSummary <- dfTest
dfTestSummary$gbmCorr <- gbmPredCorr
dfTestSummary$gbmNx5rfe <- gbmPredNx5rfe
dfTestSummary$rfNx5Opt <- rfPredNx5Opt
dfTestSummary$rfNx5rfe <- rfPredNx5rfe
dfTestSummary$svmNx5 <- svmPredNx5
dfTestSummary$svmNx5rfe <- svmPredNx5rfe
write.csv(dfTestSummary, "dfTestSummary.csv", row.names = F)

#problem is that RF is good, but really far off on high-volume prediction
# which massivly hurts RMSE and R^2

plot(dfTestSummary$gbmNx5rfe, dfTestSummary$Volume, xlim=c(0,7000))
abline(a=0, b=1, col="green")
points(dfTestSummary$rfNx5Opt, dfTestSummary$Volume, col="red")
points(dfTestSummary$rfNx5rfe, dfTestSummary$Volume, col="blue")
legend( "topleft",legend=c("gbmNx5rfe", "rfNx5Opt","rfNx5rfe"),
       col=c("black","red", "blue"), title="models", inset=c(0.02,0.02), pch=21, cex=0.8)

saveRDS(Nx5rfeGBMfit, "gbmNx5rfe.rds")  
GBMfit <- readRDS("gbmNx5rfe.rds")

##########################################
# Predict new data (Incomplete Survey Data)
#########################################
#set up the NewProd to match our existing nx5rfe data frame
#nx5rfe only uses $PositiveServiceReview and $x4StarReviews

#eval results from both best GBM model and best RF model
df_new_rfe <- df_New_Products[, c(5,9)]

gbmPred_new <- predict(GBMfit,df_new_rfe)
head(gbmPred_new)

rfPred_new <- predict(Nx5rfeRFOptfit,df_new_rfe)
head(rfPred_new)

## add it to the df and export to csv
df_New_Products$VolumePredictionGBM <- gbmPred_new
df_New_Products$VolumePredictionRF <- rfPred_new
write.csv(df_New_Products, "df_New_Products_Predicted.csv", row.names = F)

