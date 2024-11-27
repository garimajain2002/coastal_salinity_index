# ========================= Test binary EC ==============================
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

getwd()

soil_data <- read.csv("data/soil_data_allindices.csv")
head(soil_data)

# Select the type of EC to be used here 
soil_data$EC <- soil_data$EC_bin
head(soil_data)


# Clean dataset
soil_data <- soil_data[, c("Name", "EC", "Blue_R", "Green_R", "Red_R", "NIR_R", "SWIR1_R", "SWIR2_R",
                           "NDVI", "NDWI", "NDSI1", "NDSI2", "SI1", "SI2", "SI3", "SI4", "SI5", "SAVI", "VSSI",
                           "NBR", "NBG", "NBNIR", "NBSWIR1", "NBSWIR2", "NRSWIR1", "NRSWIR2", "NGSWIR1", "NGSWIR2",
                           "NNIRSWIR1", "NNIRSWIR2")]

# ================ 2. Prepare data ===============
# Prepare soil data with only numeric fields
soil_data_numeric <- soil_data[, sapply(soil_data, is.numeric)]
soil_data_numeric$EC <- as.factor(soil_data_numeric$EC)

# ================= feature selection with LASSO =====================
head(soil_data_numeric)

drop.cols <- c('EC')
x <- soil_data_numeric %>% dplyr::select(-one_of(drop.cols))
# scale x
for(i in 1:ncol(x)){
  x[, i] <- rescale(x[, i])
}
x <- data.matrix(x)

y <- soil_data_numeric$EC
table(y)

cvmodel <- cv.glmnet(x, y, alpha=1, family='binomial') # did not converge
plot(cvmodel)
best_lambda <- cvmodel$lambda.min
best_lambda

# we can also tweak lambda to see
bestlasso <- glmnet(x, y, alpha=1, lambda=best_lambda, family='binomial')
coef(bestlasso)
# selected predictors
# Green_R, SWIR2_R, NDWI, NBNIR, NBSWIR2, NRSWIR1, NNIRSWIR1




#### Stratified Sampling  #####

#### With Stratified sampling for training and testing data along with 10-fold###

# Initialize an empty dataframe to store classification results
results_df <- data.frame(
  Iteration = integer(),
  Model_Type = character(),
  Accuracy_Train = numeric(),
  Accuracy_Test = numeric(),
  Precision = numeric(),
  Recall = numeric(),
  F1_Score = numeric()
)

# Define functions
calculate_classification_metrics <- function(actual, predicted) {
  confusion <- table(factor(actual, levels = c(0, 1)), factor(predicted, levels = c(0, 1)))
  
  TN <- ifelse(!is.na(confusion[1, 1]), confusion[1, 1], 0) # True Negative
  FP <- ifelse(!is.na(confusion[1, 2]), confusion[1, 2], 0) # False Positive
  FN <- ifelse(!is.na(confusion[2, 1]), confusion[2, 1], 0) # False Negative
  TP <- ifelse(!is.na(confusion[2, 2]), confusion[2, 2], 0) # True Positive
  
  accuracy <- (TP + TN) / (TN + FP + FN + TP)
  precision <- ifelse((TP + FP) > 0, TP / (TP + FP), NA)
  recall <- ifelse((TP + FN) > 0, TP / (TP + FN), NA)
  f1_score <- ifelse(!is.na(precision) && !is.na(recall) && (precision + recall) > 0,
                     2 * (precision * recall) / (precision + recall), NA)
  
  return(list(
    Accuracy = accuracy,
    Precision = precision,
    Recall = recall,
    F1_Score = f1_score
  ))
}

scale_numeric <- function(data) {
  # Get numeric columns
  numeric_cols <- sapply(data, is.numeric)
  
  # Scale numeric columns
  scaled_data <- data
  scaled_data[, numeric_cols] <- scale(data[, numeric_cols])
  
  return(scaled_data)
}

# Loop to run the models 100 times
for (i in 1:100) {
  print(paste("Iteration:", i))
  
  # Stratified sampling for train-test split
  set.seed(i)
  train_indices <- createDataPartition(soil_data_numeric$EC, p = 0.8, list = FALSE)
  train_data <- soil_data_numeric[train_indices, ]
  test_data <- soil_data_numeric[-train_indices, ]
  
  # Debug: Check class balance
  print(paste("Train class distribution (Iteration:", i, ")"))
  print(table(train_data$EC))
  print(paste("Test class distribution (Iteration:", i, ")"))
  print(table(test_data$EC))
  
  # Check and remove near-zero variance predictors
  nzv <- nearZeroVar(train_data, saveMetrics = TRUE)
  if (any(nzv$nzv)) {
    print("Near-zero variance predictors found:")
    print(nzv[nzv$nzv, ])
    train_data <- train_data[, !nzv$nzv]
    test_data <- test_data[, !nzv$nzv]
  }
  
  # Define training control for cross-validation
  train_control <- trainControl(method = "cv", number = 10)
  
  ### 1. Logistic Regression Model ###
  logistic_model <- tryCatch({
    train(
      EC ~ Green_R + SWIR2_R + NDWI + NBNIR + NBSWIR2 + NRSWIR1 + NNIRSWIR1, 
      data = train_data, method = "glm", family = binomial(link='logit'), trControl = train_control)
    
  }, error = function(e) {
    message("Logistic model failed on iteration: ", i)
    return(NULL)
  })
  
  if (!is.null(logistic_model)) {
    train_pred_class <- predict(logistic_model, newdata = train_data)
    test_pred_class <- predict(logistic_model, newdata = test_data)
    
    metrics_train_logistic <- calculate_classification_metrics(train_data$EC, train_pred_class)
    metrics_test_logistic <- calculate_classification_metrics(test_data$EC, test_pred_class)
    
    results_df <- rbind(results_df, data.frame(
      Iteration = i,
      Model_Type = "Logistic",
      Accuracy_Train = metrics_train_logistic$Accuracy,
      Accuracy_Test = metrics_test_logistic$Accuracy,
      Precision = metrics_test_logistic$Precision,
      Recall = metrics_test_logistic$Recall,
      F1_Score = metrics_test_logistic$F1_Score
    ))
  }
  
  ### 2. Random Forest Model ###
  rf_model <- tryCatch({
    # mtry <- 5
    # tunegrid <- expand.grid(.mtry=mtry)
    # 
    # for (maxnode in c(2, 5, 10, 15, 20)){
    #   set.seed(123)
    #   print(maxnode)
    #   fit <- train(EC ~ NDWI + Green_R + SWIR2_R + NBR + NBNIR + NBSWIR2 + 
    #                  NRSWIR2 + NGSWIR1 + NDSI1 + VSSI,
    #                data = train_data,
    #                method = 'rf',
    #                metric = 'Accuracy',
    #                tuneGrid = tunegrid,
    #                trControl = train_control,
    #                maxnodes = maxnode,
    #                ntree = 1500)
    #   # key <- toString(ntree)
    #   # modellist[[key]] <- fit
    #   print(fit)
    # }
 
    randomForest(
        EC ~ Green_R + SWIR2_R + NDWI + NBNIR + NBSWIR2 + NRSWIR1 + NNIRSWIR1,
      data = train_data, ntree = 1500,
      mtry = max(5, floor(sqrt(ncol(train_data) - 1))),
      nodesize = 1, maxnodes = 15
    )
  }, error = function(e) {
    cat("Random Forest failed on iteration:", i, "\n")
    cat("Random Forest Error Message:", conditionMessage(e), "\n")
    return(NULL)
  })
  
  if (!is.null(rf_model)) {
    train_pred_class <- predict(rf_model, newdata = train_data)
    test_pred_class <- predict(rf_model, newdata = test_data)
    
    metrics_train_rf <- calculate_classification_metrics(train_data$EC, train_pred_class)
    metrics_test_rf <- calculate_classification_metrics(test_data$EC, test_pred_class)
    
    results_df <- rbind(results_df, data.frame(
      Iteration = i,
      Model_Type = "Random Forest",
      Accuracy_Train = metrics_train_rf$Accuracy,
      Accuracy_Test = metrics_test_rf$Accuracy,
      Precision = metrics_test_rf$Precision,
      Recall = metrics_test_rf$Recall,
      F1_Score = metrics_test_rf$F1_Score
    ))
  }
  
  ### 3. Artificial Neural Network (ANN) ###
  train_data_scaled <- scale_numeric(train_data)
  test_data_scaled <- scale_numeric(test_data)
  
  ann_model <- tryCatch({
    nnet(
      EC ~ Green_R + SWIR2_R + NDWI + NBNIR + NBSWIR2 + NRSWIR1 + NNIRSWIR1, 
      data = train_data_scaled, size = 4, linout = FALSE, maxit = 100)
  }, error = function(e) {
    message("ANN failed on iteration: ", i)
    return(NULL)
  })
  
  if (!is.null(ann_model)) {
    train_pred_class <- predict(ann_model, newdata = train_data_scaled, type = "class")
    test_pred_class <- predict(ann_model, newdata = test_data_scaled, type = "class")
    
    metrics_train_ann <- calculate_classification_metrics(train_data$EC, train_pred_class)
    metrics_test_ann <- calculate_classification_metrics(test_data$EC, test_pred_class)
    
    results_df <- rbind(results_df, data.frame(
      Iteration = i,
      Model_Type = "ANN",
      Accuracy_Train = metrics_train_ann$Accuracy,
      Accuracy_Test = metrics_test_ann$Accuracy,
      Precision = metrics_test_ann$Precision,
      Recall = metrics_test_ann$Recall,
      F1_Score = metrics_test_ann$F1_Score
    ))
  }
  
  print(paste("Results so far:", nrow(results_df)))
}

# Print summary of results
print("Final Summary")
print(summary(results_df))

write.csv(results_df, "outputs/smalldata_model_results_ECbin_lasso_80_20strat.csv")





#### Random Sampling  #####
# # Initialize an empty dataframe to store results
results_df <- data.frame(
  Iteration = integer(),
  Model_Type = character(),
  Accuracy_Train = numeric(),
  Accuracy_Test = numeric(),
  Precision = numeric(),
  Recall = numeric(),
  F1_Score = numeric()
)

# Define functions
calculate_classification_metrics <- function(actual, predicted) {
  confusion <- table(factor(actual, levels = c(0, 1)), factor(predicted, levels = c(0, 1)))
  
  TN <- ifelse(!is.na(confusion[1, 1]), confusion[1, 1], 0) # True Negative
  FP <- ifelse(!is.na(confusion[1, 2]), confusion[1, 2], 0) # False Positive
  FN <- ifelse(!is.na(confusion[2, 1]), confusion[2, 1], 0) # False Negative
  TP <- ifelse(!is.na(confusion[2, 2]), confusion[2, 2], 0) # True Positive
  
  accuracy <- (TP + TN) / (TN + FP + FN + TP)
  precision <- ifelse((TP + FP) > 0, TP / (TP + FP), NA)
  recall <- ifelse((TP + FN) > 0, TP / (TP + FN), NA)
  f1_score <- ifelse(!is.na(precision) && !is.na(recall) && (precision + recall) > 0,
                     2 * (precision * recall) / (precision + recall), NA)
  
  return(list(
    Accuracy = accuracy,
    Precision = precision,
    Recall = recall,
    F1_Score = f1_score
  ))
}

scale_numeric <- function(data) {
  # Get numeric columns
  numeric_cols <- sapply(data, is.numeric)
  
  # Scale numeric columns
  scaled_data <- data
  scaled_data[, numeric_cols] <- scale(data[, numeric_cols])
  
  return(scaled_data)
}

for (i in 1:100) {
  print(paste("Iteration:", i))
  
  # Shuffle and split data
  set.seed(i)
  shuffled_indices <- sample(seq_len(nrow(soil_data_numeric)))
  train_size <- floor(0.8 * nrow(soil_data_numeric))
  train_indices <- shuffled_indices[1:train_size]
  test_indices <- shuffled_indices[(train_size + 1):nrow(soil_data_numeric)]
  
  train_data <- soil_data_numeric[train_indices, ]
  test_data <- soil_data_numeric[test_indices, ]
  
  # Check and remove near-zero variance predictors
  nzv <- nearZeroVar(train_data, saveMetrics = TRUE)
  if (any(nzv$nzv)) {
    print("Near-zero variance predictors found:")
    print(nzv[nzv$nzv, ])
    train_data <- train_data[, !nzv$nzv]
    test_data <- test_data[, !nzv$nzv]
  }
  
  # Define training control for cross-validation
  train_control <- trainControl(method = "cv", number = 10)
  
  ### 1. Logistic Regression Model ###
  logistic_model <- tryCatch({
    train(
      EC ~ Green_R + SWIR2_R + NDWI + NBNIR + NBSWIR2 + NRSWIR1 + NNIRSWIR1, 
      data = train_data, method = "glm", family = binomial(link='logit'), trControl = train_control)
    
  }, error = function(e) {
    message("Logistic model failed on iteration: ", i)
    return(NULL)
  })
  
  if (!is.null(logistic_model)) {
    train_pred_class <- predict(logistic_model, newdata = train_data)
    test_pred_class <- predict(logistic_model, newdata = test_data)
    
    metrics_train_logistic <- calculate_classification_metrics(train_data$EC, train_pred_class)
    metrics_test_logistic <- calculate_classification_metrics(test_data$EC, test_pred_class)
    
    results_df <- rbind(results_df, data.frame(
      Iteration = i,
      Model_Type = "Logistic",
      Accuracy_Train = metrics_train_logistic$Accuracy,
      Accuracy_Test = metrics_test_logistic$Accuracy,
      Precision = metrics_test_logistic$Precision,
      Recall = metrics_test_logistic$Recall,
      F1_Score = metrics_test_logistic$F1_Score
    ))
  }
  
  ### 2. Random Forest Model ###
  rf_model <- tryCatch({
    # mtry <- 5
    # tunegrid <- expand.grid(.mtry=mtry)
    # 
    # for (maxnode in c(2, 5, 10, 15, 20)){
    #   set.seed(123)
    #   print(maxnode)
    #   fit <- train(EC ~ NDWI + Green_R + SWIR2_R + NBR + NBNIR + NBSWIR2 + 
    #                  NRSWIR2 + NGSWIR1 + NDSI1 + VSSI,
    #                data = train_data,
    #                method = 'rf',
    #                metric = 'Accuracy',
    #                tuneGrid = tunegrid,
    #                trControl = train_control,
    #                maxnodes = maxnode,
    #                ntree = 1500)
    #   # key <- toString(ntree)
    #   # modellist[[key]] <- fit
    #   print(fit)
    # }
    
    randomForest(
      EC ~ Green_R + SWIR2_R + NDWI + NBNIR + NBSWIR2 + NRSWIR1 + NNIRSWIR1,
      data = train_data, ntree = 1500,
      mtry = max(5, floor(sqrt(ncol(train_data) - 1))),
      nodesize = 1, maxnodes = 15
    )
  }, error = function(e) {
    cat("Random Forest failed on iteration:", i, "\n")
    cat("Random Forest Error Message:", conditionMessage(e), "\n")
    return(NULL)
  })
  
  if (!is.null(rf_model)) {
    train_pred_class <- predict(rf_model, newdata = train_data)
    test_pred_class <- predict(rf_model, newdata = test_data)
    
    metrics_train_rf <- calculate_classification_metrics(train_data$EC, train_pred_class)
    metrics_test_rf <- calculate_classification_metrics(test_data$EC, test_pred_class)
    
    results_df <- rbind(results_df, data.frame(
      Iteration = i,
      Model_Type = "Random Forest",
      Accuracy_Train = metrics_train_rf$Accuracy,
      Accuracy_Test = metrics_test_rf$Accuracy,
      Precision = metrics_test_rf$Precision,
      Recall = metrics_test_rf$Recall,
      F1_Score = metrics_test_rf$F1_Score
    ))
  }
  
  ### 3. Artificial Neural Network (ANN) ###
  train_data_scaled <- scale_numeric(train_data)
  test_data_scaled <- scale_numeric(test_data)
  
  ann_model <- tryCatch({
    nnet(
      EC ~ Green_R + SWIR2_R + NDWI + NBNIR + NBSWIR2 + NRSWIR1 + NNIRSWIR1, 
      data = train_data_scaled, size = 4, linout = FALSE, maxit = 100)
  }, error = function(e) {
    message("ANN failed on iteration: ", i)
    return(NULL)
  })
  
  if (!is.null(ann_model)) {
    train_pred_class <- predict(ann_model, newdata = train_data_scaled, type = "class")
    test_pred_class <- predict(ann_model, newdata = test_data_scaled, type = "class")
    
    metrics_train_ann <- calculate_classification_metrics(train_data$EC, train_pred_class)
    metrics_test_ann <- calculate_classification_metrics(test_data$EC, test_pred_class)
    
    results_df <- rbind(results_df, data.frame(
      Iteration = i,
      Model_Type = "ANN",
      Accuracy_Train = metrics_train_ann$Accuracy,
      Accuracy_Test = metrics_test_ann$Accuracy,
      Precision = metrics_test_ann$Precision,
      Recall = metrics_test_ann$Recall,
      F1_Score = metrics_test_ann$F1_Score
    ))
  }
  
  print(paste("Results so far:", nrow(results_df)))
}

# Print summary of results
print("Final Summary")
print(summary(results_df))

write.csv(results_df, "outputs/smalldata_model_results_ECbin_lasso_80_20random.csv")

