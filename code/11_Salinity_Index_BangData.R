# ========================= Prep Bangladesh Soil Data ==============================
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

soil_data_bang <- read.csv("data/Bangladesh_Sample_Points/Soil_Point_Data_Extracted.csv")
head(soil_data_bang)

soil_data_bang <- rename(soil_data_bang,
                         Name = sel_no, 
                         EC_d = Soil_Salin,
                         Aqua = RASTERVALU, 
                         Blue = b1_BandDN_Landsat, 
                         Green = b2_BandDN_Landsat, 
                         Red = b3_BandDN_Landsat, 
                         NIR = b4_BandDN_Landsat, 
                         SWIR1 = b5_BandDN_Landsat, 
                         SWIR2 = b6_BandDN_Landsat)

table(soil_data_bang$Aqua)

# Drop the points that are on mixed aqua cells (Aqua == 2) 
soil_data_bang <- subset(soil_data_bang, Aqua != 2)


# Convert Digital Number (DN) Values to Reflectance values using offset = -0.2 and Scale factor = 2.75e-05 (Reflectance = ((DN * scale_factor) + offset )
# Link: https://www.usgs.gov/faqs/how-do-i-use-a-scale-factor-landsat-level-2-science-products
soil_data_bang$Blue_R <- (soil_data_bang$Blue * 0.0000275) - 0.2
soil_data_bang$Red_R <- (soil_data_bang$Red * 0.0000275) - 0.2
soil_data_bang$Green_R <- (soil_data_bang$Green * 0.0000275) - 0.2
soil_data_bang$NIR_R <- (soil_data_bang$NIR * 0.0000275) - 0.2
soil_data_bang$SWIR1_R <- (soil_data_bang$SWIR1 * 0.0000275) - 0.2
soil_data_bang$SWIR2_R <- (soil_data_bang$SWIR2 * 0.0000275) - 0.2

# Convert EC_d decisemens to EC in msemens to be similar to the India data 
soil_data_bang$EC <- (soil_data_bang$EC_d * 1000)
soil_data_bang$EC_all <- soil_data_bang$EC # create a copy of continuous EC values 
soil_data_bang$EC_bin <- ifelse(soil_data_bang$EC >= 1900, 1, 0)
soil_data_bang$EC_cat <- ifelse(soil_data_bang$EC < 1400, 0, 
                           ifelse(soil_data_bang$EC < 3000, 1, 2))  

head(soil_data_bang)

# Calculate Indices 
# Create Normalised differences for all bands apart from the regularly known indices
# Blue and red 
soil_data_bang$NBR <- (soil_data_bang$Blue_R-soil_data_bang$Red_R)/(soil_data_bang$Blue_R+soil_data_bang$Red_R)

# Blue and green 
soil_data_bang$NBG <- (soil_data_bang$Blue_R-soil_data_bang$Green_R)/(soil_data_bang$Blue_R+soil_data_bang$Green_R)

# Blue and NIR
soil_data_bang$NBNIR <- (soil_data_bang$Blue_R-soil_data_bang$NIR_R)/(soil_data_bang$Blue_R+soil_data_bang$NIR_R)

# Blue and SWIR1
soil_data_bang$NBSWIR1 <- (soil_data_bang$Blue_R-soil_data_bang$SWIR1_R)/(soil_data_bang$Blue_R+soil_data_bang$SWIR1_R)

# Blue and SWIR2 
soil_data_bang$NBSWIR2 <- (soil_data_bang$Blue_R-soil_data_bang$SWIR2_R)/(soil_data_bang$Blue_R+soil_data_bang$SWIR2_R)

# Red and Green (also NDVI) 
soil_data_bang$NDVI <- (soil_data_bang$Red_R-soil_data_bang$Green_R)/(soil_data_bang$Red_R+soil_data_bang$Green_R)

# Red and NIR (NDSI2 or Normalised Difference Salinity Index 2 as per as per Khan et al 2001 in Nguyen et al 2020)
soil_data_bang$NDSI2 <- (soil_data_bang$Red_R-soil_data_bang$NIR_R)/(soil_data_bang$Red_R+soil_data_bang$NIR_R)

# Red and SWIR1
soil_data_bang$NRSWIR1 <- (soil_data_bang$Red_R-soil_data_bang$SWIR1_R)/(soil_data_bang$Red_R+soil_data_bang$SWIR1_R)

# Red and SWIR2
soil_data_bang$NRSWIR2 <- (soil_data_bang$Red_R-soil_data_bang$SWIR2_R)/(soil_data_bang$Red_R+soil_data_bang$SWIR2_R)

# Green and NIR (also NDWI) 
soil_data_bang$NDWI <- (soil_data_bang$Green_R-soil_data_bang$NIR_R)/(soil_data_bang$Green_R+soil_data_bang$NIR_R)

# Green and SWIR1
soil_data_bang$NGSWIR1 <- (soil_data_bang$Green_R-soil_data_bang$SWIR1_R)/(soil_data_bang$Green_R+soil_data_bang$SWIR1_R)

# Green and SWIR2
soil_data_bang$NGSWIR2 <- (soil_data_bang$Green_R-soil_data_bang$SWIR2_R)/(soil_data_bang$Green_R+soil_data_bang$SWIR2_R)

# NIR and SWIR1
soil_data_bang$NNIRSWIR1 <- (soil_data_bang$NIR_R-soil_data_bang$SWIR1_R)/(soil_data_bang$NIR_R+soil_data_bang$SWIR1_R)

# NIR and SWIR2 
soil_data_bang$NNIRSWIR2 <- (soil_data_bang$NIR_R-soil_data_bang$SWIR2_R)/(soil_data_bang$NIR_R+soil_data_bang$SWIR2_R)

# SWIR1 and SWIR2 (also NDSI as per the Index Database: https://www.indexdatabase.de/db/is.php?sensor_id=168 )
soil_data_bang$NDSI1 <- (soil_data_bang$SWIR1_R-soil_data_bang$SWIR2_R)/(soil_data_bang$SWIR1_R+soil_data_bang$SWIR2_R)


# Calculate other widely used Salinity indices (listed in Nguyen et al. 2020)
# Salinity Index 1 = sqrt(green^2+red^2)
soil_data_bang$SI1 <- sqrt((soil_data_bang$Green_R)^2 + (soil_data_bang$Red_R)^2) 

# Salinity Index 2 = sqrt(green x red)
soil_data_bang$SI2 <- sqrt(soil_data_bang$Green_R * soil_data_bang$Red_R)

# Salinity Index 3 = sqrt(blue x red) 
soil_data_bang$SI3 <- sqrt(soil_data_bang$Blue_R * soil_data_bang$Red_R)

# salinity index 4 = red x NIR / green 
soil_data_bang$SI4 <- (soil_data_bang$Red_R * soil_data_bang$NIR_R / soil_data_bang$Green_R)  

# salinity index 5 = blue/red 
soil_data_bang$SI5 <- (soil_data_bang$Blue_R / soil_data_bang$Red_R)   

# Soil Adjusted Vegetation Index (SAVI) = ((1.5)x NIR) - (red/0.5) + NIR + Red 
soil_data_bang$SAVI <- (1.5 * soil_data_bang$NIR_R) - (0.5 * soil_data_bang$Red_R) + soil_data_bang$NIR_R + soil_data_bang$Red_R

# Vegetation Soil Salinity Index (VSSI) = (2 x green) - 5 x (red + NIR) 
soil_data_bang$VSSI <- (2 * soil_data_bang$Green_R) - 5 * (soil_data_bang$Red_R + soil_data_bang$NIR_R)





# Clean data 
soil_data_bang <- soil_data_bang[, c("Name", "EC", "EC_all", "EC_bin", "EC_cat", "Blue_R", "Green_R", "Red_R", "NIR_R", "SWIR1_R", "SWIR2_R",
                           "NDVI", "NDWI", "NDSI1", "NDSI2", "SI1", "SI2", "SI3", "SI4", "SI5", "SAVI", "VSSI",
                           "NBR", "NBG", "NBNIR", "NBSWIR1", "NBSWIR2", "NRSWIR1", "NRSWIR2", "NGSWIR1", "NGSWIR2",
                           "NNIRSWIR1", "NNIRSWIR2")]

head(soil_data_bang)

write.csv(soil_data_bang, "data/Bangladesh_Sample_Points/soil_data_bang.csv")

