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

soil_data_bang <- read.csv("data/Bangladesh_Sample_Points/soil_data_bang.csv")


# Select the type of EC to be used here 
soil_data$EC <- soil_data$EC_bin
head(soil_data)

soil_data_bang$EC <- soil_data_bang$EC_bin


# Clean datasets
soil_data <- soil_data[, c("Name", "EC", "Blue_R", "Green_R", "Red_R", "NIR_R", "SWIR1_R", "SWIR2_R",
                           "NDVI", "NDWI", "NDSI1", "NDSI2", "SI1", "SI2", "SI3", "SI4", "SI5", "SAVI", "VSSI",
                           "NBR", "NBG", "NBNIR", "NBSWIR1", "NBSWIR2", "NRSWIR1", "NRSWIR2", "NGSWIR1", "NGSWIR2",
                           "NNIRSWIR1", "NNIRSWIR2")]

soil_data_bang <- soil_data_bang[, c("Name", "EC", "Blue_R", "Green_R", "Red_R", "NIR_R", "SWIR1_R", "SWIR2_R",
                           "NDVI", "NDWI", "NDSI1", "NDSI2", "SI1", "SI2", "SI3", "SI4", "SI5", "SAVI", "VSSI",
                           "NBR", "NBG", "NBNIR", "NBSWIR1", "NBSWIR2", "NRSWIR1", "NRSWIR2", "NGSWIR1", "NGSWIR2",
                           "NNIRSWIR1", "NNIRSWIR2")]


soil_data_numeric <- soil_data[, sapply(soil_data, is.numeric)]
soil_data_bang_numeric <- soil_data_bang[, sapply(soil_data_bang, is.numeric)]
  
soil_data_numeric$EC <- as.factor(soil_data_numeric$EC)
soil_data_bang_numeric$EC <- as.factor(soil_data_bang_numeric$EC)


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
# Green_R, SWIR2_R, NDWI, NDSI1, SI1, SI2, VSSI, NBR, NBNIR, NBSWIR2, NGSWIR1


# Run the three models this time using soil_data as training data, and soil_data_bang as testing data 
train_data <- soil_data
test_data <-  soil_data_bang 

train_data$EC <- as.factor(train_data$EC)
test_data$EC <- as.factor(test_data$EC)  

# Record results in one data frame
results_df <- data.frame(
  Model_Type = character(),
  Accuracy_Train = numeric(),
  Accuracy_Test = numeric(),
  Precision = numeric(),
  Recall = numeric(),
  F1_Score = numeric()
)
  
### 1. Logistic model 
logistic_model <- train(EC ~ Blue_R + Red_R + Green_R + NIR_R + SWIR1_R + SWIR2_R + 
      NDVI + NDWI + NDSI1 + NDSI2 + SI1 + SI2 + SI3 + SI4 + SI5 + SAVI + VSSI + 
      NBR + NBG + NBNIR + NBSWIR1 + NBSWIR2 + NRSWIR1 + NRSWIR2 + NGSWIR1 + NGSWIR2 + NNIRSWIR1 + NNIRSWIR2, 
    data = train_data, method = "glm", family = binomial(link='logit'), trControl = train_control)


  train_pred_class <- predict(logistic_model, newdata = train_data)
  test_pred_class <- predict(logistic_model, newdata = test_data)
  
  metrics_train_logistic <- calculate_classification_metrics(train_data$EC, train_pred_class)
  metrics_test_logistic <- calculate_classification_metrics(test_data$EC, test_pred_class)
  
  results_df <- rbind(results_df, data.frame(
    Model_Type = "Logistic - all",
    Accuracy_Train = metrics_train_logistic$Accuracy,
    Accuracy_Test = metrics_test_logistic$Accuracy,
    Precision = metrics_test_logistic$Precision,
    Recall = metrics_test_logistic$Recall,
    F1_Score = metrics_test_logistic$F1_Score
  ))

  ### 2. Logistic model - Lasso (Green_R, SWIR2_R, NDWI, NDSI1, SI1, SI2, VSSI, NBR, NBNIR, NBSWIR2, NGSWIR1)
  logistic_model_lasso <- train(EC ~ Green_R + SWIR2_R + NDWI + NDSI1 + SI1 + SI2 + VSSI + NBR + NBNIR + NBSWIR2 + NGSWIR1, 
                          data = train_data, method = "glm", family = binomial(link='logit'), trControl = train_control)
  
  train_pred_class <- predict(logistic_model_lasso, newdata = train_data)
  test_pred_class <- predict(logistic_model_lasso, newdata = test_data)
  
  metrics_train_logistic_lasso <- calculate_classification_metrics(train_data$EC, train_pred_class)
  metrics_test_logistic_lasso <- calculate_classification_metrics(test_data$EC, test_pred_class)
  
  results_df <- rbind(results_df, data.frame(
    Model_Type = "Logistic - Lasso",
    Accuracy_Train = metrics_train_logistic_lasso$Accuracy,
    Accuracy_Test = metrics_test_logistic_lasso$Accuracy,
    Precision = metrics_test_logistic_lasso$Precision,
    Recall = metrics_test_logistic_lasso$Recall,
    F1_Score = metrics_test_logistic_lasso$F1_Score
  ))
  
  
### 3. Random Forest Model with bagging ###
  rf_model <- randomForest(
    EC ~ Blue_R + Red_R + Green_R + NIR_R + SWIR1_R + SWIR2_R + 
      NDVI + NDWI + NDSI1 + NDSI2 + SI1 + SI2 + SI3 + SI4 + SI5 + SAVI + VSSI + 
      NBR + NBG + NBNIR + NBSWIR1 + NBSWIR2 + NRSWIR1 + NRSWIR2 + NGSWIR1 + NGSWIR2 + NNIRSWIR1 + NNIRSWIR2,
    data = train_data, ntree = 1500,
    mtry = max(5, floor(sqrt(ncol(train_data) - 1))),
    nodesize = 1, maxnodes = 15)
  
  train_pred_class <- predict(rf_model, newdata = train_data)
  test_pred_class <- predict(rf_model, newdata = test_data)
  
  metrics_train_rf <- calculate_classification_metrics(train_data$EC, train_pred_class)
  metrics_test_rf <- calculate_classification_metrics(test_data$EC, test_pred_class)
  
  results_df <- rbind(results_df, data.frame(
    Model_Type = "Random Forest - All",
    Accuracy_Train = metrics_train_rf$Accuracy,
    Accuracy_Test = metrics_test_rf$Accuracy,
    Precision = metrics_test_rf$Precision,
    Recall = metrics_test_rf$Recall,
    F1_Score = metrics_test_rf$F1_Score
  ))

  ### 4. Random Forest Model with bagging - Lasso ###
  rf_model_lasso <- randomForest(
    EC ~ Green_R + SWIR2_R + NDWI + NDSI1 + SI1 + SI2 + VSSI + NBR + NBNIR + NBSWIR2 + NGSWIR1,
    data = train_data, ntree = 1500,
    mtry = max(5, floor(sqrt(ncol(train_data) - 1))),
    nodesize = 1, maxnodes = 15)
  
  train_pred_class <- predict(rf_model_lasso, newdata = train_data)
  test_pred_class <- predict(rf_model_lasso, newdata = test_data)
  
  metrics_train_rf_lasso <- calculate_classification_metrics(train_data$EC, train_pred_class)
  metrics_test_rf_lasso <- calculate_classification_metrics(test_data$EC, test_pred_class)
  
  results_df <- rbind(results_df, data.frame(
    Model_Type = "Random Forest - Lasso",
    Accuracy_Train = metrics_train_rf_lasso$Accuracy,
    Accuracy_Test = metrics_test_rf_lasso$Accuracy,
    Precision = metrics_test_rf_lasso$Precision,
    Recall = metrics_test_rf_lasso$Recall,
    F1_Score = metrics_test_rf_lasso$F1_Score
  ))

### 5. Artificial Neural Network (ANN) ###
train_data_scaled <- scale_numeric(train_data)
test_data_scaled <- scale_numeric(test_data)

train_data_scaled$EC <- as.factor(train_data_scaled$EC)
test_data_scaled$EC <- as.factor(test_data_scaled$EC)

ann_model <- nnet(EC ~ Blue_R + Red_R + Green_R + NIR_R + SWIR1_R + SWIR2_R + 
      NDVI + NDWI + NDSI1 + NDSI2 + SI1 + SI2 + SI3 + SI4 + SI5 + SAVI + VSSI + 
      NBR + NBG + NBNIR + NBSWIR1 + NBSWIR2 + NRSWIR1 + NRSWIR2 + NGSWIR1 + NGSWIR2 + NNIRSWIR1 + NNIRSWIR2, 
    data = train_data_scaled, size = 4, linout = FALSE, maxit = 100)

  train_pred_class <- predict(ann_model, newdata = train_data_scaled, type = "class")
  test_pred_class <- predict(ann_model, newdata = test_data_scaled, type = "class")
  
  metrics_train_ann <- calculate_classification_metrics(train_data$EC, train_pred_class)
  metrics_test_ann <- calculate_classification_metrics(test_data$EC, test_pred_class)
  
  results_df <- rbind(results_df, data.frame(
    Model_Type = "ANN - All",
    Accuracy_Train = metrics_train_ann$Accuracy,
    Accuracy_Test = metrics_test_ann$Accuracy,
    Precision = metrics_test_ann$Precision,
    Recall = metrics_test_ann$Recall,
    F1_Score = metrics_test_ann$F1_Score
  ))


  ### 6. Artificial Neural Network (ANN) - Lasso ###
  train_data_scaled <- scale_numeric(train_data)
  test_data_scaled <- scale_numeric(test_data)
  
  train_data_scaled$EC <- as.factor(train_data_scaled$EC)
  test_data_scaled$EC <- as.factor(test_data_scaled$EC)
  
  ann_model_lasso <- nnet(EC ~ Green_R + SWIR2_R + NDWI + NDSI1 + SI1 + SI2 + VSSI + NBR + NBNIR + NBSWIR2 + NGSWIR1, 
                    data = train_data_scaled, size = 4, linout = FALSE, maxit = 100)
  
  train_pred_class <- predict(ann_model_lasso, newdata = train_data_scaled, type = "class")
  test_pred_class <- predict(ann_model_lasso, newdata = test_data_scaled, type = "class")
  
  metrics_train_ann_lasso <- calculate_classification_metrics(train_data$EC, train_pred_class)
  metrics_test_ann_lasso <- calculate_classification_metrics(test_data$EC, test_pred_class)
  
  results_df <- rbind(results_df, data.frame(
    Model_Type = "ANN - Lasso",
    Accuracy_Train = metrics_train_ann_lasso$Accuracy,
    Accuracy_Test = metrics_test_ann_lasso$Accuracy,
    Precision = metrics_test_ann_lasso$Precision,
    Recall = metrics_test_ann_lasso$Recall,
    F1_Score = metrics_test_ann_lasso$F1_Score
  ))
  
  
write.csv(results_df, "outputs/bang_validation_results_ECbin_80_20_random.csv")

