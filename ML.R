library(ggplot2)
library(caret)

library(randomForest)
library(e1071)
library(gbm)
library(doParallel)

library(survival)
library(splines)
library(plyr)


# load data
training <- read.csv("pml-training.csv", na.strings=c("#DIV/0!"), row.names = 1)
testing <- read.csv("pml-testing.csv", na.strings=c("#DIV/0!"), row.names = 1)

training <- training[, 6:dim(training)[2]]

treshold <- dim(training)[1] * 0.95
#Remove columns with more than 95% of NA or "" values
goodColumns <- !apply(training, 2, function(x) sum(is.na(x)) > treshold  || sum(x=="") > treshold)

training <- training[, goodColumns]

badColumns <- nearZeroVar(training, saveMetrics = TRUE)

training <- training[, badColumns$nzv==FALSE]

training$classe = factor(training$classe)

#Partition rows into training and crossvalidation
inTrain <- createDataPartition(training$classe, p = 0.6)[[1]]
crossv <- training[-inTrain,]
training <- training[ inTrain,]
inTrain <- createDataPartition(crossv$classe, p = 0.75)[[1]]
crossv_test <- crossv[ -inTrain,]
crossv <- crossv[inTrain,]


testing <- testing[, 6:dim(testing)[2]]
testing <- testing[, goodColumns]
testing$classe <- NA
testing <- testing[, badColumns$nzv==FALSE]

#Train 3 different models
mod1 <- train(classe ~ ., data=training, method="rf")
#mod2 <- train(classe ~ ., data=training, method="gbm")
#mod3 <- train(classe ~ ., data=training, method="lda")

pred1 <- predict(mod1, crossv)
#pred2 <- predict(mod2, crossv)
#pred3 <- predict(mod3, crossv)

#show confusion matrices
confusionMatrix(pred1, crossv$classe)

#confusionMatrix(pred2, crossv$classe)
#confusionMatrix(pred3, crossv$classe)

#Create Combination Model

#predDF <- data.frame(pred1, pred2, pred3, classe=crossv$classe)
#predDF <- data.frame(pred1, pred2, classe=crossv$classe)

#combModFit <- train(classe ~ ., method="rf", data=predDF)
#in-sample error
#combPredIn <- predict(combModFit, predDF)
#confusionMatrix(combPredIn, predDF$classe)



#out-of-sample error
pred1 <- predict(mod1, crossv_test)
#pred3 <- predict(mod3, crossv_test)
accuracy <- sum(pred1 == crossv_test$classe) / length(pred1)

varImpRF <- train(classe ~ ., data = training, method = "rf")
varImpObj <- varImp(varImpRF)
# Top 40 plot
plot(varImpObj, main = "Importance of Top 40 Variables", top = 40)

# Top 25 plot
plot(varImpObj, main = "Importance of Top 25 Variables", top = 25)