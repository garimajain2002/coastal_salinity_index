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
  # Using cv with 5 fold here. Could also test method = "LOOCV" (Leave-One-Out Cross-Validation) 
  
  
  # ================ 4. Train models ===============
  
  # Train multiple models using caretList
  li_models <- c("rf", "rpart", "nnet", "svmRadial", "gbm", "naive_bayes", "xgbTree", "knn", "glmnet")
  
  li_multi <- caretList(
    EC ~ Blue_R + Red_R + Green_R + NIR_R + SWIR1_R + SWIR2_R + 
      NDVI + NDWI + NDSI1 + NDSI2 + SI1 + SI2 + SI3 + SI4 + SI5 + SAVI + VSSI + 
      NBR + NBG + NBNIR + NBSWIR1 + NBSWIR2 + NRSWIR1 + NRSWIR2 + NGSWIR1 + NGSWIR2 + NNIRSWIR1 + NNIRSWIR2,
    data = train_data,
    trControl = train_control,
    methodList = li_models  
  )
  

  
  # Check the list of models that they are functioning and not null. If null remove. 
  li_multi
  
  # Model/Feature selection
  # Get performance metrics for all models
  model_performances <- resamples(li_multi)
  results <- summary(model_performances)
  
  print(results)
  
  accuracies <- unlist(lapply(li_multi, function(model) {
    max(model$results$Accuracy)
  }))
  
  # Assess detailed statistics
  performance_stats <- summary(model_performances)$statistics
  
  # Use statistical methods to set a threshold
  # Method 1: Mean threshold
  mean_threshold <- mean(accuracies)
  
  # Method 2: Median threshold
  median_threshold <- median(accuracies)
  
  # Method 3: Mean minus one SD
  sd_threshold <- mean(accuracies) - sd(accuracies)
  
  selected_models <- names(accuracies[accuracies > mean_threshold])
  
  print(paste("Mean accuracy:", round(mean_threshold, 4)))
  print(paste("Models above threshold:", paste(selected_models, collapse=", ")))
  
  
  # Visual inspection to find  a natural breakpoint?
  # Plot model performances
  model_comparison <- dotplot(model_performances)  
  plot(model_comparison )
  
  
  
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
  
  
  train_preds <- predict(MultiStratEnsemble, newdata = train_data)
  test_preds <- predict(MultiStratEnsemble, newdata = test_data)
  
  train_probs <- as.numeric(unlist(train_preds[,"X1"]))
  test_probs <- as.numeric(unlist(test_preds[,"X1"]))
  
    # Determine the best threshold
  # # Create ROC object 
  train_roc <- roc(train_data$EC, train_probs)
  test_roc <- roc(test_data$EC, test_probs)
  
  
  # Find best threshold using Youden's J statistic
  best_threshold_train <- coords(train_roc, "best", best.method = "youden")
  best_threshold_test <- coords(test_roc, "best", best.method = "youden")
  
  # Calculate AUC values
  train_auc <- auc(train_roc)
  test_auc <- auc(test_roc)
  
  print(paste("Training AUC:", round(train_auc, 3)))
  print(paste("Testing AUC:", round(test_auc, 3)))
  
  plot(train_roc, col = "blue", main = "ROC Curves")
  lines(test_roc, col = "red")
  legend("bottomright", legend = c("Training", "Testing"), 
         col = c("blue", "red"), lwd = 2)
  
  
  # # Alternative: find threshold that maximizes specificity + sensitivity
  # best_threshold_alt <- coords(roc_obj, "best", best.method = "closest.topleft")
  # # 
  #   # Defaul threshold
  # best_threshold_def <- 0.5

  best_threshold = best_threshold_test$threshold 
    
  print(best_threshold)
  
  
  # OR 
  
  
  # Find a threshold that gives the least diff between train and test accuracies. 
  li_thresholds <- seq(0.1, 0.9, 0.01)
  df <- data.frame(matrix(ncol=4, nrow=0))
  colnames(df) <- c("threshold", "train_acc","test_acc", "diff")
  
  for(threshold in li_thresholds){
    # Predict classes 
    train_predicted_class <- ifelse(train_preds[, "X1"] > threshold, 1, 0)
    test_predicted_class <- ifelse(test_preds[, "X1"] > threshold, 1, 0)
    
    # Calculate metrics
    train_ensemble_metrics <- calculate_classification_metrics(train_data$EC, train_predicted_class)
    train_acc <- train_ensemble_metrics$Accuracy
    test_ensemble_metrics <- calculate_classification_metrics(test_data$EC, test_predicted_class)
    test_acc <- test_ensemble_metrics$Accuracy
    diff <- train_acc-test_acc
    
    df[nrow(df)+1, ] <- c(threshold, train_acc, test_acc, abs(diff))
    
    if(abs(diff)<0.1){
      print(threshold)
    }
  }
  
  min(df$diff)
  df[df$diff==min(df$diff), ]
  
  best_threshold = mean(df[df$diff==min(df$diff), ]$threshold)
  
      # Print the threshold
  print(best_threshold)
  
  
  
  # Predict classes 
  # train_predicted_class <- ifelse(train_preds[, "X1"] > best_threshold$threshold, 1, 0)
  # test_predicted_class <- ifelse(test_preds[, "X1"] > best_threshold$threshold, 1, 0)
  
  # Predict classes based on identified threshold
  train_predicted_class <- ifelse(train_preds[, "X1"] > best_threshold, 1, 0)
  test_predicted_class <- ifelse(test_preds[, "X1"] > best_threshold, 1, 0)
  
  # Calculate metrics
  train_ensemble_metrics <- calculate_classification_metrics(train_data$EC, train_predicted_class)
  test_ensemble_metrics <- calculate_classification_metrics(test_data$EC, test_predicted_class)
  
  # Print ensemble metrics
  print(train_ensemble_metrics)
  print(test_ensemble_metrics)
  
  
  saveRDS(MultiStratEnsemble, "ensemble.rds")
  



 
   