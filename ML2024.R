install.packages("readr")
install.packages("ggplot2")
install.packages("forecast")
install.packages("urca")
install.packages("tseries")
install.packages("tsDyn")
install.packages("strucchange")
#install.packages("rpart.plot")

library(readr)
library(ggplot2)
library(forecast)
library(urca)
library(tseries)
library(tsDyn)
library(strucchange)

################################################################################
#INSTAL LIBRARY
################################################################################

install.packages("caret")
install.packages("randomForest")
install.packages("tictoc")
library(caret)
install.packages("randomForest")
library(randomForest)
install.packages("tictoc")
library(tictoc)
install.packages("MLmetrics")
library(MLmetrics)

################################################################################
#LOAD DATA
################################################################################
trainData <- read.csv("train.csv")

#SELECT PREDICTORS
predictors <- setdiff(names(trainData), "target")
X_train <- trainData[, predictors]

#FORMAT TARGET VARIABLE
trainData$target <- factor(trainData$target, levels = c("0","1"))


nobs <- nrow(trainData)
percentageAccuracy <-25*0.01

#DIVIDE DATA FROM TRAINING DATA FOR SIMPEL ACCURACY CALCULATION
#Training data
set.seed(777)
a<-sample(1:nobs,round(percentageAccuracy*nobs,0))
Data_train_Accuracy <- trainData[-a,]
X_train_Accuracy <- Data_train_Accuracy[, predictors]

#Validation data
X_val = X_train[a,]
Y_val = trainData[a,1]


################################################################################
#TUNE HYPERPARAMETER
################################################################################
tic("total")#START TIMER FOR TOTAL CODE
tree_sizes <- c(130, 135)
#tree_sizes <- c(130, 135,140,145,150)
Accuracies <- vector("numeric", length = length(tree_sizes))
BestMtries <- vector("numeric" , length = length(tree_sizes))
j=0
for (i in tree_sizes) {
  j = j+1
  #create parameter grid mtry
  param_grid <- expand.grid(
    #mtry = c(50, 80, 90,100,110) 
    mtry = c(150) # Number of variables randomly sampled for each split
    #ntree = c(100, 200, 300)  # Number of trees in the forest
  )
  
  #Tune the hyperparameter mtry
  ctrl <- trainControl(
    method = "cv",            # Cross-validation
    number = 3,               # Number of cross-validation folds
    verboseIter = TRUE        # Create log of cv
  )
  
  set.seed(456)
  
  rf_model <- train(
    target ~ .,    
    data = Data_train_Accuracy,
    method = "rf",           # Random Forest
    trControl = ctrl,
    tuneGrid = param_grid,
    ntree = i,
    metric = "Accuracy",
    parallel = T,
    cpus = 8,# Kan runtime verlagen, check of je ook 8 cpus beschikbaar hebt
    criterion = "entropy", 
    seed = set.seed(456)
  )
  
  print(rf_model)
  
  
  ################################################################################
  #PREDICT VALUES
  ################################################################################
  set.seed(456)
  rf_model3 = randomForest(X_train_Accuracy, y = Data_train_Accuracy$target , ntree = i, mtry = rf_model$finalModel$mtry,  importance = TRUE, criterion = "entropy")
  test_prediction = predict(rf_model3, X_val)
  
  ################################################################################
  #CALCULATE ACCURACY
  ################################################################################
  Accuracy = Accuracy(test_prediction, Y_val)
  Accuracy
  rf_model$finalModel$mtry
  Accuracies[j] <- Accuracy
  BestMtries[j] <-rf_model$finalModel$mtry
  
}
toc() #EINDE TOTAL TIMER
Accuracies
BestMtries



# MAJORITY ACCURACY FOR THE VALIDATION SET
Number1 <- nrow(subset(trainData[a,], target == 1))
Number0 <- nrow(subset(trainData[a,], target == 0))
majority_ACC = max(Number1, Number0)/round(percentageAccuracy*nobs,0)
majority_ACC




