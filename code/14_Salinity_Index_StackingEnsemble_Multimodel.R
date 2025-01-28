# ______________________________________________________________________________
# ~~ Stack Ensemble multiple models with Binary EC and stratified sampling ~~ #
# ______________________________________________________________________________

library(tidyverse)
library(ggplot2)
library(gtsummary) # For summary tables
library(modelsummary)# For summary tables
library(mgcv) # GAM model fit 
library(randomForest) # to apply machine learning frameworks
library(datawizard) # for normalize()
library(nnet) # For ANN
library(neuralnet) # For more control on the architecture of ANN
library(glmnet) # For lasso regression 
library(caret)  # For bagging
library(MASS) # for stepwise regression
library(dplyr)
library(scales) # for scaling data from 0 to 1
library(caretEnsemble) # For ensemble
library(kernlab)
library(naivebayes)
library(pROC)

# ================ 1. Read data ===============
getwd()

source("code/Funcs.R")

soil_data <- read.csv("data/soil_data_allindices.csv")
head(soil_data)


# ================ 2. Prepare data ===============

# Select the type of EC to be used here 
soil_data$EC <- soil_data$EC_bin
head(soil_data)

# Clean dataset
soil_data <- soil_data[, c("Name", "EC", "Blue_R", "Green_R", "Red_R", "NIR_R", "SWIR1_R", "SWIR2_R",
                           "NDVI", "NDWI", "NDSI1", "NDSI2", "SI1", "SI2", "SI3", "SI4", "SI5", "SAVI", "VSSI",
                           "NBR", "NBG", "NBNIR", "NBSWIR1", "NBSWIR2", "NRSWIR1", "NRSWIR2", "NGSWIR1", "NGSWIR2",
                           "NNIRSWIR1", "NNIRSWIR2")]


# Prepare soil data with only numeric fields
soil_data_numeric <- soil_data[, sapply(soil_data, is.numeric)]
soil_data_numeric$EC <- as.factor(soil_data_numeric$EC)

table(soil_data_numeric$EC)


# ================ 3. Stratified Sampling ===============

#### With Stratified sampling for training and testing data along with 5-fold###

  # Stratified sampling for train-test split
  set.seed(123)
  train_indices <- createDataPartition(soil_data_numeric$EC, p = 0.8, list = FALSE)
  train_data <- soil_data_numeric[train_indices, ]
  test_data <- soil_data_numeric[-train_indices, ]
  
  # Convert EC into Factor (0 and 1 to X0 and X1)
  levels(train_data$EC) <- make.names(levels(train_data$EC))
  levels(test_data$EC) <- make.names(levels(test_data$EC))
  train_data$EC <- factor(train_data$EC, levels = c("X0", "X1"))
  test_data$EC <- factor(test_data$EC, levels = c("X0", "X1"))
  
  # Debug: Check class balance
  print(paste("Train class distribution"))
  print(table(train_data$EC))
  print(paste("Test class distribution"))
  print(table(test_data$EC))
  
  # Check and remove near-zero variance predictors
  nzv <- nearZeroVar(train_data, saveMetrics = TRUE)
  if (any(nzv$nzv)) {
    print("Near-zero variance predictors found:")
    print(nzv[nzv$nzv, ])
    train_data <- train_data[, !nzv$nzv]
    test_data <- test_data[, !nzv$nzv]
  }
  
  # Define training control
  train_control <- trainControl(method = "cv", number = 5, savePredictions = "final", classProbs = TRUE)
  
  
  # ================ 4. Train models ===============
  
  # Train multiple models using caretList
  li_multi <- caretList(
    EC ~ Blue_R + Red_R + Green_R + NIR_R + SWIR1_R + SWIR2_R + 
      NDVI + NDWI + NDSI1 + NDSI2 + SI1 + SI2 + SI3 + SI4 + SI5 + SAVI + VSSI + 
      NBR + NBG + NBNIR + NBSWIR1 + NBSWIR2 + NRSWIR1 + NRSWIR2 + NGSWIR1 + NGSWIR2 + NNIRSWIR1 + NNIRSWIR2,
    data = train_data,
    trControl = train_control,
    methodList = c("rf", "rpart", "nnet", "svmRadial", "gbm", "naive_bayes", "xgbTree", "knn", "glmnet")  
  )
  
  # Check the list of models that they are functioning and not null. If null remove. 
  li_multi
  
  # ================ 5. Create the stack ensemble ===============
  # Define stack ensemble
  MultiStratEnsemble <- caretStack(
    li_multi,
    method = "glm",  # Stacking method
    metric = "Accuracy",
    trControl = trainControl(method = "cv", number = 5, classProbs = TRUE)
  )
  
  # # Check the ensemble object
  # MultiStratEnsemble
  # #str(MultiStratEnsemble)
  
  
  # ================ 6. Evaluate the Stack Ensemble ===============
  #   # Summary of the ensemble model
  # summary(MultiStratEnsemble)
  
  # Predict probabilities
  # Check individual model predictions ("rf", "rpart", "nnet", "svmRadial", "gbm", "naive_bayes", "xgbTree", "knn", "glmnet")
  rf_prob <- predict(li_multi$rf, newdata = test_data, type = "prob")
  class(li_multi$rf)
  rpart_prob <- predict(li_multi$rpart, newdata = test_data, type = "prob")
  nnet_prob <- predict(li_multi$nnet, newdata = test_data, type = "prob")
  svmRadial_prob <- predict(li_multi$svmRadial, newdata = test_data, type = "prob")
  gbm_prob <- predict(li_multi$gbm, newdata = test_data, type = "prob")
  naive_bayes_prob <- predict(li_multi$naive_bayes, newdata = test_data, type = "prob")
  xgb_prob <- predict(li_multi$xgbTree, newdata = test_data, type = "prob")
  knn_prob <- predict(li_multi$knn, newdata = test_data, type = "prob")
  glmnet_prob <- predict(li_multi$glmnet, newdata = test_data, type = "prob")
 
  # Combine probabilities (average)
  ensemble_probabilities <- (rf_prob + rpart_prob + nnet_prob + svmRadial_prob + gbm_prob + naive_bayes_prob + xgb_prob + knn_prob + glmnet_prob) / 9
  
  
  # determine the best threshold
  # Create ROC object
  roc_obj <- roc(test_data$EC, ensemble_probabilities[, "X1"])
  
  # Find best threshold using Youden's J statistic
  best_threshold <- coords(roc_obj, "best", best.method = "youden")
  
  # Alternative: find threshold that maximizes specificity + sensitivity
  best_threshold_alt <- coords(roc_obj, "best", best.method = "closest.topleft")
  
  # Print the threshold
  print(best_threshold)
 
  
  
  # Predict classes 
  predicted_class <- ifelse(ensemble_probabilities[, "X1"] > best_threshold$threshold, 1, 0)
  
  
  # Calculate metrics
  ensemble_metrics <- calculate_classification_metrics(test_data$EC, predicted_class)
  
  
  # Print ensemble metrics
  print(ensemble_metrics)

  
  # ================ 7. Evaluate the Stack Ensemble on Bangladesh ===============

  soil_data_bang <- read.csv("data/Bangladesh_Sample_Points/soil_data_bang.csv")

  # Predict probabilities
  # Check individual model predictions ("rf", "rpart", "nnet", "svmRadial", "gbm", "naive_bayes", "xgbTree", "knn", "glmnet")
  rf_prob <- predict(li_multi$rf, newdata = soil_data_bang, type = "prob")
  class(li_multi$rf)
  rpart_prob <- predict(li_multi$rpart, newdata = soil_data_bang, type = "prob")
  nnet_prob <- predict(li_multi$nnet, newdata = soil_data_bang, type = "prob")
  svmRadial_prob <- predict(li_multi$svmRadial, newdata = soil_data_bang, type = "prob")
  gbm_prob <- predict(li_multi$gbm, newdata = soil_data_bang, type = "prob")
  naive_bayes_prob <- predict(li_multi$naive_bayes, newdata = soil_data_bang, type = "prob")
  xgb_prob <- predict(li_multi$xgbTree, newdata = soil_data_bang, type = "prob")
  knn_prob <- predict(li_multi$knn, newdata = soil_data_bang, type = "prob")
  glmnet_prob <- predict(li_multi$glmnet, newdata = soil_data_bang, type = "prob")
  
  # Combine probabilities (average)
  ensemble_probabilities <- (rf_prob + rpart_prob + nnet_prob + svmRadial_prob + gbm_prob + naive_bayes_prob + xgb_prob + knn_prob + glmnet_prob) / 9
  
  
  # determine the best threshold
  # Create ROC object
  roc_obj <- roc(soil_data_bang$EC, ensemble_probabilities[, "X1"])
  
  # Find best threshold using Youden's J statistic
  best_threshold <- coords(roc_obj, "best", best.method = "youden")
  
  # Alternative: find threshold that maximizes specificity + sensitivity
  best_threshold_alt <- coords(roc_obj, "best", best.method = "closest.topleft")
  
  # Print the threshold
  print(best_threshold)
  
  
  # Predict classes 
  predicted_class <- ifelse(ensemble_probabilities[, "X1"] > best_threshold$threshold, 1, 0)
  
  
  # Calculate metrics
  ensemble_metrics_bang <- calculate_classification_metrics(soil_data_bang$EC, predicted_class)
  
  
  # Print ensemble metrics
  print(ensemble_metrics_bang)

    
#Bangladesh data accuracy is is about 71%. Decent. 
  


  

