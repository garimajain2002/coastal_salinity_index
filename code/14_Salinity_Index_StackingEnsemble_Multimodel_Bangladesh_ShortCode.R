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

# ================ 1. Read data ===============
getwd()

source("code/Funcs.R")

ensemble <- readRDS("code/ensemble_2501.rds")

soil_data_bang <- read.csv("data/Bangladesh_Sample_Points/soil_data_bang.csv")

head(soil_data)



# ================ 2. Evaluate the Stack Ensemble on Bangladesh ===============

# Select the type of EC to be used here 
head(soil_data_bang)

# Clean dataset
soil_data_bang <- soil_data_bang[, c("Name", "EC", "Blue_R", "Green_R", "Red_R", "NIR_R", "SWIR1_R", "SWIR2_R",
                           "NDVI", "NDWI", "NDSI1", "NDSI2", "SI1", "SI2", "SI3", "SI4", "SI5", "SAVI", "VSSI",
                           "NBR", "NBG", "NBNIR", "NBSWIR1", "NBSWIR2", "NRSWIR1", "NRSWIR2", "NGSWIR1", "NGSWIR2",
                           "NNIRSWIR1", "NNIRSWIR2")]

bang_preds <- predict(ensemble, newdata=soil_data_bang)

  
  # Predict classes 
  predicted_class <- ifelse(bang_preds[, "X1"] > 0.75, 1, 0)
  
  
  # Calculate metrics
  ensemble_metrics_bang <- calculate_classification_metrics(soil_data_bang$EC, predicted_class)
  
  
  # Print ensemble metrics
  print(ensemble_metrics_bang)

    
#Bangladesh data accuracy is about 61.3%. 
  


  

