install.packages("readr")
library(readr)
IrisDataset <- read.csv("iris.csv")
attributes(IrisDataset)
summary(IrisDataset) 
str(IrisDataset)
names(IrisDataset)
hist(IrisDataset$Petal.Length)
plot(IrisDataset$Sepal.Length, IrisDataset$Sepal.Width)
qqnorm(IrisDataset$Sepal.Length)
IrisDataset$Species<- factor(IrisDataset$Species) 
set.seed(123)
trainSize <- round(nrow(IrisDataset) * 0.7)
testSize <- nrow(IrisDataset) - trainSize
trainSize
testSize
training_indices<-sample(seq_len(nrow(IrisDataset)),size =trainSize)
trainSet <- IrisDataset[training_indices, ]
testSet <- IrisDataset[-training_indices, ]
set.seed(405)
trainSet <- IrisDataset[training_indices, ]
testSet <- IrisDataset[-training_indices, ]
LinearModel<- lm(Petal.Length ~ Petal.Width, trainSet)
summary(LinearModel)
prediction<-predict(LinearModel, testSet)
prediction
write.csv(prediction, file="iris_pred_length.csv")
write.csv(testSet, file="iris_testSet_length.csv")
