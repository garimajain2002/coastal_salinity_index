# EC CONTINUOUS ALL VARIABLES Confidence interval validation # 

# Need to make the models simpler and less complex, given the small data set: 
# https://www.kaggle.com/code/rafjaa/dealing-with-very-small-datasets
# # Next steps: 
# 1. Do feature selection using regularization or model averaging (https://medium.com/rants-on-machine-learning/what-to-do-with-small-data-d253254d1a89)
# 2. OR Test with fewer parameters to improve overfitting issues: Red, Blue, Green, NIR, SWIR1, SWIR2, NBNIR, NDSI2, NRSWIR1, NDWI, SAVI
# 3. Account for outliers or try quantile regression (https://en.wikipedia.org/wiki/Quantile_regression)
# 4. Balance the dataset with SMOTE
# 5. Apply confidence intervals rather than point estimates for validation 
# 6. Adjust the RF parameters including reducing the depth 
# 7. Add more penalties to the models to avoid overfitting 
# 8. Try on binary and categorical (low, medium, high)
# 9. Test data smaller 90-10
# 10. Apply Leave one out cross validation method 
# 11. Add principal components of the bands instead of the bands directly. 
# 12. Add land_cover as a control/FE and then conduct the analysis. The same land cover categories can be used in the GEE LULC to apply in the predictive model. 
# 13. Try pooling data from other sources - from Bangladesh study? 

library(tidyverse)
library(ggplot2)
library(gtsummary) # For summary tables
library(modelsummary)# For summary tables
library(mgcv) # GAM model fit 
library(randomForest) # to apply machine learning frameworks
library(datawizard) # for normalize()
library(nnet) # For ANN
library(neuralnet) # For more control on the architecture of ANN
library(glmnet) # For Ridge and lasso regularization in regression 
library(caret)  # For bagging
library(MASS) # for step wise regression
library(dplyr)
library(MASS) # for robust regression
library(performanceEstimation) # for balanced sampling using SMOTE
library(ROSE) # oversampling technique for regressions
library(quantreg) # for quantile regression that tests for medians and not means and less affected by outliers 
library(factoextra)



# Set working directory 
# laptop 
setwd("C:\\Users\\Garima\\Google Drive (garimajain2002@gmail.com)\\03 PHD_Work\\Chapter 1\\Salinity and pH Indices\\")

soil_data <- read.csv("soil_data_allindices")

# Convert EC into binary (high low salinity - threshold at 1900) and three categorical variable (high (>3000), medium (1900-3000), low (<1900))
soil_data$EC_all <- soil_data$EC # create a copy of continuous EC values - EC will take on different values for analysis 
#soil_data$EC_bin <- ifelse(soil_data$EC >= 1900, 1, 0)
#soil_data$EC_cat <- ifelse(soil_data$EC < 1900, 1, ifelse(soil_data$EC < 3000, 2, 3))  

#table(soil_data$EC_cat)
#table(soil_data$EC_bin)

# Select the type of EC to be used here 
soil_data$EC <- soil_data$EC_all

# As Binary or categorical
#soil_data$EC <- soil_data$EC_bin
#soil_data$EC <- soil_data$EC_cat


# Select for full data
soil_data <- soil_data[, c("Name", "EC", "Blue_R", "Green_R", "Red_R", "NIR_R", "SWIR1_R", "SWIR2_R",
                           "NDVI", "NDWI", "NDSI1", "NDSI2", "SI1", "SI2", "SI3", "SI4", "SI5", "SAVI", "VSSI",
                           "NBR", "NBG", "NBNIR", "NBSWIR1", "NBSWIR2", "NRSWIR1", "NRSWIR2", "NGSWIR1", "NGSWIR2",
                           "NNIRSWIR1", "NNIRSWIR2")]



# select for highly correlated and pure band data 
# soil_data <- soil_data[, c("Name", "EC", "Blue_R", "Green_R", "Red_R", "NIR_R", "SWIR1_R", "SWIR2_R",
#                            "NBNIR", "NDSI2", "NRSWIR1", "NDWI", "SAVI")]


# Note: SMOTE didn't work despite multiple attempts. It anyway only works on binary or categorical variables. 
# Adapt this later for continuous variable using other functions like SMOGN (Synthetic Minority Oversampling for Regression) (package: smotefamily), an extension of SMOTE designed for continuous target variables


####~~~~~~~~~ EC AS BINARY VARIABLE ~~~~~~~~~####


# Prepare soil data with only numeric fields
soil_data_numeric <- soil_data[, sapply(soil_data, is.numeric)]

# Initialize results dataframe
results_df <- data.frame(
  Iteration = integer(),
  Model_Type = character(),
  R2_Train = numeric(),
  R2_Test = numeric(),
  RMSE_Train = numeric(),
  RMSE_Test = numeric(),
  CI_Lower_Test = numeric(),
  CI_Upper_Test = numeric()
)

# Define calculate_metrics function with confidence intervals (90%)
calculate_metrics <- function(actual, predicted, confidence_level = 0.90) {
  residuals <- actual - predicted
  mse <- mean(residuals^2)
  rmse <- sqrt(mse)
  r2 <- 1 - (sum(residuals^2) / sum((actual - mean(actual))^2))
  
  # Calculate 90% confidence interval for residuals
  alpha <- (1 - confidence_level) / 2
  se <- sd(residuals) / sqrt(length(residuals))
  ci <- qt(c(alpha, 1 - alpha), df = length(residuals) - 1) * se
  
  return(list(
    RMSE = rmse,
    R2 = r2,
    CI_Lower = mean(predicted) + ci[1],
    CI_Upper = mean(predicted) + ci[2]
  ))
}

# Function to handle outliers
handle_outliers <- function(data, threshold = 3) {
  numeric_cols <- sapply(data, is.numeric)
  data <- data %>%
    filter_all(all_vars(abs(scale(.)[numeric_cols]) <= threshold))
  return(data)
}

# Main loop
for (i in 1:100) {
  cat("Iteration:", i, "\n")
  
  # Handle outliers
  soil_data_clean <- handle_outliers(soil_data_numeric)
  
  # Split data into training and testing sets
  shuffled_indices <- sample(seq_len(nrow(soil_data_clean)))
  train_size <- floor(0.8 * nrow(soil_data_clean))
  train_indices <- shuffled_indices[1:train_size]
  test_indices <- shuffled_indices[(train_size + 1):nrow(soil_data_clean)]
  
  train_data <- soil_data_clean[train_indices, ]
  test_data <- soil_data_clean[test_indices, ]
  
  
  
  ### 1&2. Regularization (Lasso and Ridge) ###
  x_train <- as.matrix(train_data[, -which(names(train_data) == "EC")])
  y_train <- train_data$EC
  x_test <- as.matrix(test_data[, -which(names(test_data) == "EC")])
  y_test <- test_data$EC
  
  for (alpha in c(1, 0)) { # 1 = Lasso, 0 = Ridge
    model <- cv.glmnet(x_train, y_train, alpha = alpha, maxit = 200)
    train_pred <- predict(model, x_train, s = "lambda.min")
    test_pred <- predict(model, x_test, s = "lambda.min")
    
    # Metrics calculation
    metrics_train <- calculate_metrics(y_train, train_pred, confidence_level = 0.90)
    metrics_test <- calculate_metrics(y_test, test_pred, confidence_level = 0.90)
    
    # Append results
    results_df <- rbind(results_df, data.frame(
      Iteration = i,
      Model_Type = ifelse(alpha == 1, "Lasso", "Ridge"),
      R2_Train = metrics_train$R2,
      R2_Test = metrics_test$R2,
      RMSE_Train = metrics_train$RMSE,
      RMSE_Test = metrics_test$RMSE,
      CI_Lower_Test = metrics_test$CI_Lower,
      CI_Upper_Test = metrics_test$CI_Upper
    ))
  }
  
  
  ### 3. Quantile Regression ###
  quantile_model <- tryCatch({
    rq(EC ~ ., data = train_data, tau = 0.5)
  }, error = function(e) {
    NULL
  })
  
  if (!is.null(quantile_model)) {
    train_predictions_quant <- predict(quantile_model, newdata = train_data)
    test_predictions_quant <- predict(quantile_model, newdata = test_data)
    
    metrics_train_quant <- calculate_metrics(train_data$EC, train_predictions_quant, confidence_level = 0.90)
    metrics_test_quant <- calculate_metrics(test_data$EC, test_predictions_quant, confidence_level = 0.90)
    
    results_df <- rbind(results_df, data.frame(
      Iteration = i,
      Model_Type = "Quantile Regression",
      R2_Train = metrics_train_quant$R2,
      R2_Test = metrics_test_quant$R2,
      RMSE_Train = metrics_train_quant$RMSE,
      RMSE_Test = metrics_test$RMSE,
      CI_Lower_Test = metrics_test_quant$CI_Lower,
      CI_Upper_Test = metrics_test_quant$CI_Upper
    ))
  }
  
  
  ### 4. Logarithmic with all variables ### 
  log_linear_model <- tryCatch({
    lm(log(EC) ~ Blue_R + Red_R + Green_R + NIR_R + SWIR1_R + SWIR2_R + 
         NDVI + NDWI + NDSI1 + NDSI2 + SI1 + SI2 + SI3 + SI4 + SI5 + SAVI + VSSI + 
         NBR + NBG + NBNIR + NBSWIR1 + NBSWIR2 + NRSWIR1 + NRSWIR2 + NGSWIR1 + NGSWIR2 + NNIRSWIR1 + NNIRSWIR2, data = train_data)
  }, error = function(e) {
    message("Log-Linear model failed on iteration: ", i)
    return(NULL)
  })
  
  if (!is.null(log_linear_model)) {
    train_predictions_log_linear <- exp(predict(log_linear_model, newdata = train_data))
    test_predictions_log_linear <- exp(predict(log_linear_model, newdata = test_data))
    
    metrics_train_log_linear <- calculate_metrics(train_data$EC, train_predictions_log_linear, confidence_level = 0.90)
    metrics_test_log_linear <- calculate_metrics(test_data$EC, test_predictions_log_linear, confidence_level = 0.90)
    
    results_df <- rbind(results_df, data.frame(
      Iteration = i,
      Model_Type = "Log-Linear",
      R2_Train = metrics_train_log_linear$R2,
      R2_Test = metrics_test_log_linear$R2,
      RMSE_Train = metrics_train_log_linear$RMSE,
      RMSE_Test = metrics_test_log_linear$RMSE, 
      CI_Lower_Test = metrics_test_log_linear$CI_Lower,
      CI_Upper_Test = metrics_test_log_linear$CI_Upper
    ))
  }
  
  
  ### 5. OLS Model with all variables ###
  
  ols_model <- tryCatch({
    lm(EC ~ Blue_R + Red_R + Green_R + NIR_R + SWIR1_R + SWIR2_R + 
         NDVI + NDWI + NDSI1 + NDSI2 + SI1 + SI2 + SI3 + SI4 + SI5 + SAVI + VSSI + 
         NBR + NBG + NBNIR + NBSWIR1 + NBSWIR2 + NRSWIR1 + NRSWIR2 + NGSWIR1 + NGSWIR2 + NNIRSWIR1 + NNIRSWIR2, data = train_data)
  }, error = function(e) {
    message("OLS failed on iteration: ", i)
    return(NULL)
  })
  
  if (!is.null(ols_model)) {
    train_predictions_ols <- predict(ols_model, newdata = train_data)
    test_predictions_ols <- predict(ols_model, newdata = test_data)
    
    metrics_train_ols <- calculate_metrics(train_data$EC, train_predictions_ols, confidence_level = 0.90)
    metrics_test_ols <- calculate_metrics(test_data$EC, test_predictions_ols, confidence_level = 0.90)
    
    results_df <- rbind(results_df, data.frame(
      Iteration = i,
      Model_Type = "OLS",
      R2_Train = metrics_train_ols$R2,
      R2_Test = metrics_test_ols$R2,
      RMSE_Train = metrics_train_ols$RMSE,
      RMSE_Test = metrics_test_ols$RMSE, 
      CI_Lower_Test = metrics_test_ols$CI_Lower,
      CI_Upper_Test = metrics_test_ols$CI_Upper
    ))
  }
  
  
  
  ### 6. Polynomial with all variables ###
  
  poly_all <- tryCatch({
    lm(EC ~ poly(Blue_R,2) + poly(Red_R,2) + poly(Green_R,2) + poly(NIR_R,2) + poly(SWIR1_R,2) + poly(SWIR2_R,2) + 
         poly(NDVI,2) + poly(NDWI,2) + poly(NDSI1,2) + poly(NDSI2,2) + poly(SI1,2) + poly(SI2,2) + poly(SI3,2) + poly(SI4,2) + poly(SI5,2) + poly(SAVI,2) + poly(VSSI,2) + 
         poly(NBR,2) + poly(NBG,2) + poly(NBNIR,2) + poly(NBSWIR1,2) + poly(NBSWIR2,2) + poly(NRSWIR1,2) + poly(NRSWIR2,2) + poly(NGSWIR1,2) + poly(NGSWIR2,2) + poly(NNIRSWIR1,2) + poly(NNIRSWIR2,2) , data = train_data)
  }, error = function(e) {
    message("Polynomial OLS failed on iteration: ", i)
    return(NULL)
  })
  if (!is.null(poly_all)) {
    train_predictions_poly <- predict(poly_all, newdata = train_data)
    test_predictions_poly <- predict(poly_all, newdata = test_data)
    
    metrics_train_poly <- calculate_metrics(train_data$EC, train_predictions_poly, confidence_level = 0.90)
    metrics_test_poly <- calculate_metrics(test_data$EC, test_predictions_poly, confidence_level = 0.90)
    
    # Append results for Quadratic
    results_df <- rbind(results_df, data.frame(
      Iteration = i,
      Model_Type = "Quadratic_all",
      R2_Train = metrics_train_poly$R2,
      R2_Test = metrics_test_poly$R2,
      RMSE_Train = metrics_train_poly$RMSE,
      RMSE_Test = metrics_test_poly$RMSE, 
      CI_Lower_Test = metrics_test_poly$CI_Lower,
      CI_Upper_Test = metrics_test_poly$CI_Upper
    ))
  }
  
  
  ### 7. Polynomial with highly correlated variables ###  
  poly_corr <- tryCatch({
    lm(EC ~ poly(NIR_R,2) + poly(NDWI,2) + poly(NDSI2,2) + poly(SAVI,2) + poly(NBNIR,2) + poly(NRSWIR1,2), data = train_data)
  }, error = function(e) {
    message("Polynomial (correlated) failed on iteration: ", i)
    return(NULL)
  })
  if (!is.null(poly_corr)) {
    train_predictions_corr <- predict(poly_corr, newdata = train_data)
    test_predictions_corr <- predict(poly_corr, newdata = test_data)
    
    metrics_train_corr <- calculate_metrics(train_data$EC, train_predictions_corr, confidence_level = 0.90)
    metrics_test_corr <- calculate_metrics(test_data$EC, test_predictions_corr, confidence_level = 0.90)
    
    # Append results for Quadratic all
    results_df <- rbind(results_df, data.frame(
      Iteration = i,
      Model_Type = "Quadratic_correlated",
      R2_Train = metrics_train_corr$R2,
      R2_Test = metrics_test_corr$R2,
      RMSE_Train = metrics_train_corr$RMSE,
      RMSE_Test = metrics_test_corr$RMSE, 
      CI_Lower_Test = metrics_test_corr$CI_Lower,
      CI_Upper_Test = metrics_test_corr$CI_Upper
    ))
  }
  
  
  ### 8. Stepwise regression ###  
  stepwise_model <- tryCatch({
    stepAIC(lm(EC ~ Blue_R + Red_R + Green_R + NIR_R + SWIR1_R + SWIR2_R + 
                 NDVI + NDWI + NDSI1 + NDSI2 + SI1 + SI2 + SI3 + SI4 + SI5 + SAVI + VSSI + 
                 NBR + NBG + NBNIR + NBSWIR1 + NBSWIR2 + NRSWIR1 + NRSWIR2 + NGSWIR1 + NGSWIR2 + NNIRSWIR1 + NNIRSWIR2, data = train_data), direction = "both", trace = FALSE)
  }, error = function(e) {
    message("Stepwise regression failed on iteration: ", i)
    return(NULL)
  })
  
  if (!is.null(stepwise_model)) {
    train_predictions_stepwise <- predict(stepwise_model, newdata = train_data)
    test_predictions_stepwise <- predict(stepwise_model, newdata = test_data)
    
    metrics_train_stepwise <- calculate_metrics(train_data$EC, train_predictions_stepwise, confidence_level = 0.90)
    metrics_test_stepwise <- calculate_metrics(test_data$EC, test_predictions_stepwise, confidence_level = 0.90)
    
    results_df <- rbind(results_df, data.frame(
      Iteration = i,
      Model_Type = "Stepwise",
      R2_Train = metrics_train_stepwise$R2,
      R2_Test = metrics_test_stepwise$R2,
      RMSE_Train = metrics_train_stepwise$RMSE,
      RMSE_Test = metrics_test_stepwise$RMSE, 
      CI_Lower_Test = metrics_test_stepwise$CI_Lower,
      CI_Upper_Test = metrics_test_stepwise$CI_Upper
    ))
  }
  
  
  
  
  ### 9. Random Forest ###
  rf_model <- tryCatch({
    randomForest(EC ~ Blue_R + Red_R + Green_R + NIR_R + SWIR1_R + SWIR2_R + 
                   NDVI + NDWI + NDSI1 + NDSI2 + SI1 + SI2 + SI3 + SI4 + SI5 + SAVI + VSSI + 
                   NBR + NBG + NBNIR + NBSWIR1 + NBSWIR2 + NRSWIR1 + NRSWIR2 + NGSWIR1 + NGSWIR2 + NNIRSWIR1 + NNIRSWIR2, 
                 data = train_data, ntree = 500, maxnodes = 6) # use reduced tree depth for small dataset and tune the parameters till the results get better (ntree is higher the better, could also try sampsize to pick a certain number of observations in each try)
  }, error = function(e) {
    message("Random Forest failed on iteration: ", i)
    return(NULL)
  })
  
  if (!is.null(rf_model)) {
    train_predictions_rf <- predict(rf_model, newdata = train_data)
    test_predictions_rf <- predict(rf_model, newdata = test_data)
    
    metrics_train_rf <- calculate_metrics(train_data$EC, train_predictions_rf, confidence_level = 0.90)
    metrics_test_rf <- calculate_metrics(test_data$EC, test_predictions_rf, confidence_level = 0.90)
    
    # Append results for Random Forest
    results_df <- rbind(results_df, data.frame(
      Iteration = i,
      Model_Type = "RandomForest",
      R2_Train = metrics_train_rf$R2,
      R2_Test = metrics_test_rf$R2,
      RMSE_Train = metrics_train_rf$RMSE,
      RMSE_Test = metrics_test_rf$RMSE, 
      CI_Lower_Test = metrics_test_rf$CI_Lower,
      CI_Upper_Test = metrics_test_rf$CI_Upper
    ))
  }
  
  ### 10. Bagging (Random Forest with Bootstrap Aggregating) ###
  bagging_model <- tryCatch({
    randomForest(EC ~ Blue_R + Red_R + Green_R + NIR_R + SWIR1_R + SWIR2_R + 
                   NDVI + NDWI + NDSI1 + NDSI2 + SI1 + SI2 + SI3 + SI4 + SI5 + SAVI + VSSI + 
                   NBR + NBG + NBNIR + NBSWIR1 + NBSWIR2 + NRSWIR1 + NRSWIR2 + NGSWIR1 + NGSWIR2 + NNIRSWIR1 + NNIRSWIR2, data = train_data, 
                 ntree = 500, mtry = 5, maxnodes = 6) 
  }, error = function(e) {
    message("Random Forest with Bagging failed on iteration: ", i)
    return(NULL)
  })
  
  if (!is.null(bagging_model)) {
    train_predictions_bagging <- predict(bagging_model, newdata = train_data)
    test_predictions_bagging <- predict(bagging_model, newdata = test_data)
    
    metrics_train_bagging <- calculate_metrics(train_data$EC, train_predictions_bagging, confidence_level = 0.90)
    metrics_test_bagging <- calculate_metrics(test_data$EC, test_predictions_bagging, confidence_level = 0.90)
    
    results_df <- rbind(results_df, data.frame(
      Iteration = i,
      Model_Type = "Bagging",
      R2_Train = metrics_train_bagging$R2,
      R2_Test = metrics_test_bagging$R2,
      RMSE_Train = metrics_train_bagging$RMSE,
      RMSE_Test = metrics_test_bagging$RMSE, 
      CI_Lower_Test = metrics_test_bagging$CI_Lower,
      CI_Upper_Test = metrics_test_bagging$CI_Upper
    ))
  }
  
  ### 11. Artificial Neural Network (ANN) ###
  
  # Scale data for ANN
  train_data_scaled <- scale_numeric(train_data)
  test_data_scaled <- scale_numeric(test_data)
  
  ann_model <- tryCatch({
    nnet(EC ~ Blue_R + Red_R + Green_R + NIR_R + SWIR1_R + SWIR2_R + 
           NDVI + NDWI + NDSI1 + NDSI2 + SI1 + SI2 + SI3 + SI4 + SI5 + SAVI + VSSI + 
           NBR + NBG + NBNIR + NBSWIR1 + NBSWIR2 + NRSWIR1 + NRSWIR2 + NGSWIR1 + NGSWIR2 + NNIRSWIR1 + NNIRSWIR2, data = train_data_scaled, size = 4, linout = TRUE, maxit = 100)
  }, error = function(e) {
    message("ANN failed on iteration: ", i)
    return(NULL)
  })
  
  if (!is.null(ann_model)) {
    train_predictions_ann <- predict(ann_model, newdata = train_data_scaled)
    test_predictions_ann <- predict(ann_model, newdata = test_data_scaled)
    
    train_predictions_ann_rescaled <- reverse_scale(train_predictions_ann, train_data$EC)
    test_predictions_ann_rescaled <- reverse_scale(test_predictions_ann, test_data$EC)
    
    metrics_train_ann <- calculate_metrics(train_data$EC, train_predictions_ann_rescaled, confidence_level = 0.90)
    metrics_test_ann <- calculate_metrics(test_data$EC, test_predictions_ann_rescaled, confidence_level = 0.90)
    
    results_df <- rbind(results_df, data.frame(
      Iteration = i,
      Model_Type = "ANN",
      R2_Train = metrics_train_ann$R2,
      R2_Test = metrics_test_ann$R2,
      RMSE_Train = metrics_train_ann$RMSE,
      RMSE_Test = metrics_test_ann$RMSE, 
      CI_Lower_Test = metrics_test_ann$CI_Lower,
      CI_Upper_Test = metrics_test_ann$CI_Upper
    ))
  }
  
  ### ADD 12. Model Averaging / Ensemble
  
  
  # Print current size of results_df for diagnostics
  print(paste("Results so far: ", nrow(results_df))) 
}

# Save results to a CSV file
write.csv(results_df, "smalldata_model_results_continuous_all_CI_80_20.csv")




# ### 5. OLS Model with all variables ###
# ols_model <- tryCatch({
#   lm(EC ~ ., data = train_data)
# }, error = function(e) {
#   message("OLS failed on iteration: ", i)
#   return(NULL)
# })
# 
# if (!is.null(ols_model)) {
#   train_predictions <- predict(ols_model, newdata = train_data)
#   test_predictions <- predict(ols_model, newdata = test_data)
#   
#   metrics_train <- calculate_metrics(train_data$EC, train_predictions, confidence_level = 0.90)
#   metrics_test <- calculate_metrics(test_data$EC, test_predictions, confidence_level = 0.90)
#   
#   results_df <- rbind(results_df, data.frame(
#     Iteration = i,
#     Model_Type = "OLS",
#     R2_Train = metrics_train$R2,
#     R2_Test = metrics_test$R2,
#     RMSE_Train = metrics_train$RMSE,
#     RMSE_Test = metrics_test$RMSE,
#     CI_Lower_Test = metrics_test$CI_Lower,
#     CI_Upper_Test = metrics_test$CI_Upper
#   ))
# }
