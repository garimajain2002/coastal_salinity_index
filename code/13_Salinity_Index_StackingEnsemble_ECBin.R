# =================== Stacking Ensemble binary EC random forest ==================
# Note: we still need to discuss if this is the way to go
# 1. stacking ensemble predicts probabilities rather than zero/one, so we need to re-convert probabilities back to binary EC values (i.e., a threshold is needed)
# 2. stacking seems to be usually applied to different algorithms (i.e., rf, svm, etc.) rather than like this

# ref: https://cran.r-project.org/web/packages/caretEnsemble/vignettes/caretEnsemble-intro.html
# ================ 1. Read in packages and data ===============
pkgs <- c("tidyverse", "ggplot2", "gtsummary", "modelsummary", "mgcv", "randomForest", "datawizard", "nnet", "neuralnet", "glmnet", "caret", "MASS", "dplyr", "scales", "caretEnsemble")
lapply(pkgs, library, character.only=TRUE)

source("code/Funcs.R")

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

# Prepare soil data with only numeric fields
soil_data_numeric <- soil_data[, sapply(soil_data, is.numeric)]
soil_data_numeric$EC <- as.factor(soil_data_numeric$EC)
levels(soil_data_numeric$EC) <- make.names(levels(factor(soil_data_numeric$EC)))

# ================ 2. Stratified sampling ===============
# With Stratified sampling for training and testing data along with 10-fold
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

# create an empty list of models
li_rf <- c()

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
  train_control <- trainControl(method = "cv", number = 10,
                                savePredictions = "final",
                                classProbs = TRUE)
  tuneGrid <- expand.grid(.mtry=c(1:max(5, floor(sqrt(ncol(train_data)-1)))))
  
  ### Random Forest Model ###
  rf_model <- tryCatch({
    train(
      EC ~ Blue_R + Red_R + Green_R + NIR_R + SWIR1_R + SWIR2_R + 
        NDWI + NDSI2 + SAVI + NBNIR + NRSWIR1,
      data = train_data, method="rf", 
      ntree = 1500,
      nodesize = 1, maxnodes = 15,
      tuneGrid = tuneGrid,
      trControl=train_control
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
    
    li_rf <- append(li_rf, list(rf_model))
    
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
  
  print(paste("Results so far:", nrow(results_df)))
}

# Print summary of results
print("Final Summary")
print(summary(results_df))
#write.csv(results_df, "outputs/smalldata_model_results_ECbin_highcorr_80_20strat.csv")


# ====== 4. Stack ensemble ======
rfEnsemble <- caretStack(li_rf, method="rf")
# test if the stack ensemble model works
pred <- predict(rfEnsemble, newdata = train_data)
pred$EC <- train_data$EC

pred_test <- predict(rfEnsemble, newdata = test_data)
pred_test$EC <- test_data$EC
