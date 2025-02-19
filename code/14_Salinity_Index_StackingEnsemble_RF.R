# ______________________________________________________________________________
# ~~ Stack Ensemble multiple models with Binary EC and stratified sampling ~~ #
# ______________________________________________________________________________

# Load required libraries
library(caret)
library(caretEnsemble)
library(pROC)
library(data.table)
library(dplyr)
library(scales)
library(glmnet)
library(xgboost)
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
library(MASS) # for stepwise regression
library(scales) # for scaling data from 0 to 1
library(kernlab)
library(naivebayes)


# Load accuracy metrics function
source("code/0_AccuracyFunction.R")

# ================ 1. Load & Prepare Data ===============
soil_data <- read.csv("data/soil_data_allindices.csv")

# Select EC type for classification
soil_data$EC <- soil_data$EC_bin  

# Keep only necessary columns
soil_data <- soil_data[, c("Name", "EC", "Blue_R", "Green_R", "Red_R", "NIR_R", "SWIR1_R", "SWIR2_R",
                           "NDVI", "NDWI", "NDSI1", "NDSI2", "SI1", "SI2", "SI3", "SI4", "SI5", "SAVI", "VSSI",
                           "NBR", "NBG", "NBNIR", "NBSWIR1", "NBSWIR2", "NRSWIR1", "NRSWIR2", "NGSWIR1", "NGSWIR2",
                           "NNIRSWIR1", "NNIRSWIR2")]

# Keep only numeric fields
soil_data_numeric <- soil_data[, sapply(soil_data, is.numeric)]


# ================ 2. Reduce data dimensionality  ===============

# # Option 1: Drop highly correlated parameters
# cor_matrix <- cor(soil_data_numeric)
# findCorrelation(cor_matrix, cutoff = 0.8)
# 
# colnames(soil_data_numeric)[c(7, 6, 4, 29, 14, 13, 12, 8, 2, 28, 27, 11, 21, 15, 26, 18, 23, 25, 22, 5, 19)]
# 
# soil_data_numeric <- soil_data_numeric[, -c(7, 6, 4, 29, 14, 13, 12, 8, 2, 28, 27, 11, 21, 15, 26, 18, 23, 25, 22, 5, 19)]
# 
# # Only 7 paramtrs left 
# 


# Option 2: Lasso variable selection 
set.seed(123)
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
png(filename='outputs/lasso_lambda.png', width=800, height=400, res=96)
plot(cvmodel)
dev.off()

best_lambda <- cvmodel$lambda.min
best_lambda

# we can also tweak lambda to see
bestlasso <- glmnet(x, y, alpha=1, lambda=best_lambda, family='binomial')
coef(bestlasso)
# selected predictors
# Green_R, SWIR2_R, NDWI, NDSI1, VSSI, NBNIR, NBSWIR2, NRSWIR1, NGSWIR1, NNIRSWIR1

soil_data_numeric <- dplyr::select(soil_data_numeric,
  c('EC', 'Green_R', 'SWIR2_R', 'NDWI', 'NDSI1', 'VSSI', 'NBNIR', 'NBSWIR2', 'NRSWIR1', 'NGSWIR1', 'NNIRSWIR1'))


# ================ 3. Stratified Train-Test Split ===============
# Convert EC to factor for model classification
soil_data_numeric$EC <- as.factor(soil_data_numeric$EC)


set.seed(123)
train_indices <- createDataPartition(soil_data_numeric$EC, p = 0.8, list = FALSE)
train_data <- soil_data_numeric[train_indices, ]
test_data <- soil_data_numeric[-train_indices, ]

# Convert EC (0,1) to factor levels X0 and X1
levels(train_data$EC) <- make.names(levels(train_data$EC))
levels(test_data$EC) <- make.names(levels(test_data$EC))
train_data$EC <- factor(train_data$EC, levels = c("X0", "X1"))
test_data$EC <- factor(test_data$EC, levels = c("X0", "X1"))

# Debug: Check class balance
print("Train class distribution:")
print(table(train_data$EC))
print("Test class distribution:")
print(table(test_data$EC))

# ================ 4. Remove Near-Zero Variance Predictors ===============
nzv <- nearZeroVar(train_data, saveMetrics = TRUE)
if (any(nzv$nzv)) {
  print("Near-zero variance predictors found and removed:")
  print(nzv[nzv$nzv, ])
  train_data <- train_data[, !nzv$nzv]
  test_data <- test_data[, !nzv$nzv]
}

# Define cross-validation control
train_control <- trainControl(method = "cv", number = 5, savePredictions = "final", classProbs = TRUE)
# Using cv with 5 fold here. Could also test method = "LOOCV" (Leave-One-Out Cross-Validation) 
  

# ================ 5. Train Models Individually ===============
li_models <- c("rf", "rpart", "nnet", "svmRadial", "gbm", "naive_bayes", "xgbTree", "knn", "glmnet")

li_multi <- caretList(
  EC ~ .,
  data = train_data,
  trControl = train_control,
  methodList = li_models
)

# Warning comes from xgboost

# Check models - If null remove 
print(li_multi)



# ================ 6. Model Performance & Selection ===============
model_performances <- resamples(li_multi)
results <- summary(model_performances)
print(results)

# Extract accuracies
accuracies <- unlist(lapply(li_multi, function(model) {
  max(model$results$Accuracy)
}))

# Assess detailed statistics
performance_stats <- summary(model_performances)$statistics

# Use statistical methods to set a threshold
# Method 1: Mean threshold
mean_threshold <- mean(accuracies)

# # Method 2: Median threshold
# median_threshold <- median(accuracies)
# 
# # Method 3: Mean minus one SD
# sd_threshold <- mean(accuracies) - sd(accuracies)

selected_models <- names(accuracies[accuracies > mean_threshold])

print(paste("Mean accuracy:", round(mean_threshold, 4)))
print(paste("Models above threshold:", paste(selected_models, collapse=", ")))


# Visual inspection to find  a natural breakpoint?
# Plot model performances
model_comparison <- dotplot(model_performances)
plot(model_comparison)



# ================ 7. Create the Stacked Ensemble ===============
MultiStratEnsemble <- caretStack(
  li_multi,
  method = "rf",  
  metric = "ROC",
  trControl = trainControl(
    method = "cv", 
    number = 5, 
    classProbs = TRUE,
    summaryFunction = twoClassSummary
  )
)

# Check the ensemble object
print(summary(MultiStratEnsemble))

# Tried the glm model
#MultiStratEnsemble_glm <- caretStack(
#  li_multi,
#  method = "glm",  
#  metric = "ROC",
#  trControl = trainControl(
#    method = "cv", 
#    number = 5, 
#    classProbs = TRUE,
#    summaryFunction = twoClassSummary
#  )
#)



# ================ 8. Make Predictions & Fix Type Issues ===============
# Predict probabilities
train_preds <- predict(MultiStratEnsemble_glm, newdata=train_data)
test_preds <- predict(MultiStratEnsemble_glm, newdata=test_data)

# Ensure output is a data frame with X0 and X1 probabilities
print(str(train_preds))  
print(str(test_preds))  

# Extract probabilities of class X1
train_preds_numeric <- train_preds$X1
test_preds_numeric <- test_preds$X1

# Check lengths to avoid errors
print(length(train_preds_numeric))
print(length(test_preds_numeric))
print(length(train_data$EC))
print(length(test_data$EC))


# ================ 9. Determine Best Threshold (Youden's J) ===============
test_ec_numeric <- as.numeric(test_data$EC == "X1")
roc_obj <- roc(test_ec_numeric, test_preds_numeric)
best_threshold_df <- coords(roc_obj, "best", best.method = "youden")
best_threshold <- best_threshold_df$threshold

print(paste("Best threshold:", round(best_threshold, 4)))

# test sensitivity to thresholds
li_thresholds <- seq(0.1, 0.9, 0.01)
df <- data.frame(matrix(ncol=4, nrow=0))
colnames(df) <- c("threshold", "train_acc","test_acc", "diff")

for(threshold in li_thresholds){
  # Predict classes 
  train_predicted_class <- ifelse(train_preds[, "X1"] > threshold, "X1", "X0")
  test_predicted_class <- ifelse(test_preds[, "X1"] > threshold, "X1", "X0")
  
  # Calculate metrics
  train_ensemble_metrics <- calculate_classification_metrics(train_data$EC, train_predicted_class)
  train_acc <- train_ensemble_metrics$Accuracy
  test_ensemble_metrics <- calculate_classification_metrics(test_data$EC, test_predicted_class)
  test_acc <- test_ensemble_metrics$Accuracy
  diff <- train_acc-test_acc
  
  df[nrow(df)+1, ] <- c(threshold, train_acc, test_acc, abs(diff))
  
#  if(abs(diff)<0.1){
#    print(threshold)
#  }
}

write.csv(df, 'outputs/threshold_tests.csv')

# min(df$diff)

mean(df$diff)

# ================ 10. Prediction ===============

# Convert probabilities to class predictions
train_predicted_class <- ifelse(train_preds_numeric > best_threshold, "X1", "X0")
test_predicted_class <- ifelse(test_preds_numeric > best_threshold, "X1", "X0")

# Ensure predictions match EC factor levels
train_predicted_class <- factor(train_predicted_class, levels = c("X0", "X1"))
test_predicted_class <- factor(test_predicted_class, levels = c("X0", "X1"))

# Final length check before evaluation
print(length(train_predicted_class))
print(length(test_predicted_class))


# ================ 11. Evaluate Model Performance ===============
train_ensemble_metrics <- calculate_classification_metrics(train_data$EC, train_predicted_class)
test_ensemble_metrics <- calculate_classification_metrics(test_data$EC, test_predicted_class)

print("Train Metrics:")
print(train_ensemble_metrics)

print("Test Metrics:")
print(test_ensemble_metrics)



# ================ 12. Save Model ===============
saveRDS(MultiStratEnsemble, "ensemble_0502.rds")
# test <- readRDS("ensemble_0502.rds")

