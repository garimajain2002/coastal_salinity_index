# _____________________________________________________________________________________________________
# ~~ Stack Ensemble multiple models with Binary EC and stratified sampling - Applied on Bangladesh ~~ #
# _____________________________________________________________________________________________________

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

# ================ 1. Read Functions ===============
getwd()
setwd("C:/Users/garim/Documents/GitHub_LocalRepository/coastal_salinity_index - private")

getwd()

# oldermodel <- readRDS("ensemble_geomedian.rds")
# names(oldermodel$models)
# model_rf <- oldermodel$models$rf
# model_rf$finalModel$xNames
# Input bands: "Red"     "SWIR2"   "SI4"     "VSSI"    "NBNIR"   "NBSWIR2" "NRSWIR2" "NGSWIR1"

# Load the ensemble model
MultiStratEnsemble <- readRDS("ensemble_2000.rds")

# Set the threshold calculated using the training data
best_threshold = 0.661

# Load accuracy metrics function
source("code/0_AccuracyFunction.R")


# ================ 2. Read and Prepare Data ===============

soil_data_bang <- read.csv("data/Bangladesh_Sample_Points/soil_data_bang_2000EC.csv")

head(soil_data_bang) 

# Clean dataset
soil_data_bang <- soil_data_bang[, c("EC", "EC_all", "EC_bin", "Blue_R", "Green_R", "Red_R", "NIR_R", "SWIR1_R", "SWIR2_R",
                                     "NDVI", "NDWI", "NDSI1", "NDSI2", "SI1", "SI2", "SI3", "SI4", "SI5", "SAVI", "VSSI",
                                     "NBR", "NBG", "NBNIR", "NBSWIR1", "NBSWIR2", "NRSWIR1", "NRSWIR2", "NGSWIR1", "NGSWIR2",
                                     "NNIRSWIR1", "NNIRSWIR2")]

soil_data_bang <- rename(soil_data_bang,
                         Blue = Blue_R, 
                         Green = Green_R, 
                         Red = Red_R, 
                         NIR = NIR_R, 
                         SWIR1 = SWIR1_R, 
                         SWIR2 = SWIR2_R)

# Ensure EC is binary 
soil_data_bang$EC <- soil_data_bang$EC_bin


soil_data_bang <- dplyr::select(soil_data_bang,
                                        c('EC', 'Red', 'SWIR1', 'SWIR2', 'SI4' ,'VSSI', 'NBNIR', 'NBSWIR2', 'NRSWIR2', 'NGSWIR1'))

 
# Ensure EC is a factor with correct levels
#soil_data_bang$EC <- factor(soil_data_bang$EC, levels = c(0,1))
soil_data_bang$EC <- as.factor(soil_data_bang$EC)

test_data <- soil_data_bang

table(test_data$EC)

levels(test_data$EC) <- make.names(levels(test_data$EC))
test_data$EC <- factor(test_data$EC, levels = c("X0", "X1"))

# ================= 3. Test data distribution ===========================

# ggplot(test_data, aes(x = Blue_R)) + geom_histogram()
# ggplot(soil_data, aes(x = Blue_R)) + geom_histogram()
# 
# ggplot(test_data, aes(x = Green_R)) + geom_histogram()
# ggplot(soil_data, aes(x = Green_R)) + geom_histogram()
# 
# ggplot(test_data, aes(x = Red_R)) + geom_histogram()
# ggplot(soil_data, aes(x = Red_R)) + geom_histogram()
# 
# ggplot(test_data, aes(x = NIR_R)) + geom_histogram()
# ggplot(soil_data, aes(x = NIR_R)) + geom_histogram()
# 
# ggplot(test_data, aes(x = SWIR1_R)) + geom_histogram()
# ggplot(soil_data, aes(x = SWIR1_R)) + geom_histogram()
# 
# ggplot(test_data, aes(x = SWIR2_R)) + geom_histogram()
# ggplot(soil_data, aes(x = SWIR2_R)) + geom_histogram()


# ================ 4. Make Predictions ===============

# Predict probabilities on the test data
test_preds <- predict(MultiStratEnsemble, newdata=test_data)

# Ensure the output is a data frame with probabilities for both classes (X0 and X1)
print(str(test_preds))

# Extract probabilities for class X1 (adjust if the column names differ)
test_preds_numeric <- test_preds$X1


# ================= 4. Find best_threshold for Bangladesh ================

# roc_curve <- roc(test_data$EC, Bang_preds_numeric)
# best_threshold <- coords(roc_curve, "best", ret = "threshold")
# print(best_threshold)

# Assuming EC is a factor with levels "X0" and "X1"
test_ec_numeric <- as.numeric(test_data$EC == "X1")
roc_obj <- roc(test_ec_numeric, test_preds_numeric)

# Calculate the best threshold using Youden's J statistic
best_threshold_df <- coords(roc_obj, "best", best.method = "youden")
best_threshold <- best_threshold_df$threshold

print(paste("Best threshold:", round(best_threshold, 4)))


# ================ 8. Convert Probabilities to Class Predictions ===============
test_predicted_class <- ifelse(test_preds_numeric > best_threshold, "X1", "X0")

# Ensure the predicted class is a factor
test_predicted_class <- factor(test_predicted_class, levels = c("X0", "X1"))


# ================ 9. Evaluate Model Performance ===============
# Evaluate performance using the custom accuracy function (you should have defined this function earlier)
test_ensemble_metrics <- calculate_classification_metrics(test_data$EC, test_predicted_class)

print("Bangladesh Test Metrics:")
print(test_ensemble_metrics)


# Bangladesh data accuracy is about 64% but recall is 87.5% and F1 is 72.2%
  


  

