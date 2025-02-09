#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Soil Salinity Index for Tropical Coastal Contexts with Aquaculture 

# Step 1. Read the csv with soil field data and band information
# Step 2. Convert raw data to Reflectance and create all possible indices  
# Step 3. Visualise the data to understand the relationship between different bands, band combinations, indices and EC
# Step 4. Conduct statistical correlation and model fit for 80% of field data points
# Step 5. Conduct accuracy testing with the remaining 20% points 
# Step 6. Identify most appropriate bands and indices along with coefficients 

# Step 7: Alternate approach using Machine learning
# Use machine learning to fit the best model using all possible bands information 
# In this case, we do not need to know the specific relationship a band (or a combination of bands) has with EC
# This can be done in a way to find, let's say, which band/combinations are best suited to predict High, Medium and Low EC. 
# Once the model is trained with 80% of the data, it can be tested with the remaining 20%, and once the accuracy is tested, it can be applied to a larger area. 

# Step 8: Fit these models 100 times to get the average fit between different training/testing datasets 

# Other variations to try)
# 1. Try EC as (1) continuous, (2) binary, and (3) categorical variable to take away potential noise
# 2. Add quadratic and exponential values (e^) of select bands in the model too, to improve it further. 
# 3. Add principal components of the bands instead of the bands directly. 
# 4. Add land_cover as a control/FE and then conduct the analysis. The same land cover categories can be used in the GEE LULC to apply in the predictive model. 

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

# Set working directory 
getwd()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Step 1. Read the csv with soil field data and band information (extracted from ArcGIS)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
soil_data <- read.csv("data/Soil_data_raw.csv")

# Remove observations with missing Band values from Soil Data (7 points were on the masked areas)
soil_data <- soil_data[!is.na(soil_data$Blue), ]

# Keep soil moisture data for reflectance assessment
head(soil_data)
summary(soil_data$Soil_Moisture_Lab)
table(soil_data$Soil_moisture_field) # have field measures to fill the 40 missing lab measures, but leaving out for now. 

# Make a clean subset of data
soil_data <- soil_data[, c("Name", "EC", "pH", "Soil_Moisture_Lab", "Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2")]


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Step 2. Convert raw data to Reflectance and create all possible indices 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Convert Digital Number (DN) Values to Reflectance values using offset = -0.2 and Scale factor = 2.75e-05 (Reflectance = ((DN * scale_factor) + offset )
# Link: https://www.usgs.gov/faqs/how-do-i-use-a-scale-factor-landsat-level-2-science-products
soil_data$Blue_R <- (soil_data$Blue * 0.0000275) - 0.2
soil_data$Red_R <- (soil_data$Red * 0.0000275) - 0.2
soil_data$Green_R <- (soil_data$Green * 0.0000275) - 0.2
soil_data$NIR_R <- (soil_data$NIR * 0.0000275) - 0.2
soil_data$SWIR1_R <- (soil_data$SWIR1 * 0.0000275) - 0.2
soil_data$SWIR2_R <- (soil_data$SWIR2 * 0.0000275) - 0.2

# # Normalize Band Values
# soil_data$Blue_RN <- normalize(soil_data$Blue_R)
# summary(soil_data$Blue_RN)
# soil_data$Red_RN <- normalize(soil_data$Red_R)
# soil_data$Green_RN <- normalize(soil_data$Green_R)
# soil_data$NIR_RN <- normalize(soil_data$NIR_R)
# soil_data$SWIR1_RN <- normalize(soil_data$SWIR1_R)
# soil_data$SWIR2_RN <- normalize(soil_data$SWIR2_R)

# Convert EC into binary (high low salinity - threshold at 1900) and three categorical variable (high (>3000), medium (1900-3000), low (<1900))
soil_data$EC_all <- soil_data$EC # create a copy of continuous EC values - EC will take on different values for analysis 
soil_data$EC_bin <- ifelse(soil_data$EC >= 1900, 1, 0)
soil_data$EC_cat <- ifelse(soil_data$EC < 1400, 0, 
                           ifelse(soil_data$EC < 3000, 1, 2))  
# Note- binary is 1 and 2 and not 0 and 1 to make log functions work
table(soil_data$EC_bin)
table(soil_data$EC_cat)


# Create Normalised differences for all bands apart from the regularly known indices
# Blue and red 
soil_data$NBR <- (soil_data$Blue_R-soil_data$Red_R)/(soil_data$Blue_R+soil_data$Red_R)

# Blue and green 
soil_data$NBG <- (soil_data$Blue_R-soil_data$Green_R)/(soil_data$Blue_R+soil_data$Green_R)

# Blue and NIR
soil_data$NBNIR <- (soil_data$Blue_R-soil_data$NIR_R)/(soil_data$Blue_R+soil_data$NIR_R)

# Blue and SWIR1
soil_data$NBSWIR1 <- (soil_data$Blue_R-soil_data$SWIR1_R)/(soil_data$Blue_R+soil_data$SWIR1_R)

# Blue and SWIR2 
soil_data$NBSWIR2 <- (soil_data$Blue_R-soil_data$SWIR2_R)/(soil_data$Blue_R+soil_data$SWIR2_R)

# Red and Green (also NDVI) 
soil_data$NDVI <- (soil_data$Red_R-soil_data$Green_R)/(soil_data$Red_R+soil_data$Green_R)

# Red and NIR (NDSI2 or Normalised Difference Salinity Index 2 as per as per Khan et al 2001 in Nguyen et al 2020)
soil_data$NDSI2 <- (soil_data$Red_R-soil_data$NIR_R)/(soil_data$Red_R+soil_data$NIR_R)

# Red and SWIR1
soil_data$NRSWIR1 <- (soil_data$Red_R-soil_data$SWIR1_R)/(soil_data$Red_R+soil_data$SWIR1_R)

# Red and SWIR2
soil_data$NRSWIR2 <- (soil_data$Red_R-soil_data$SWIR2_R)/(soil_data$Red_R+soil_data$SWIR2_R)

# Green and NIR (also NDWI) 
soil_data$NDWI <- (soil_data$Green_R-soil_data$NIR_R)/(soil_data$Green_R+soil_data$NIR_R)

# Green and SWIR1
soil_data$NGSWIR1 <- (soil_data$Green_R-soil_data$SWIR1_R)/(soil_data$Green_R+soil_data$SWIR1_R)

# Green and SWIR2
soil_data$NGSWIR2 <- (soil_data$Green_R-soil_data$SWIR2_R)/(soil_data$Green_R+soil_data$SWIR2_R)

# NIR and SWIR1
soil_data$NNIRSWIR1 <- (soil_data$NIR_R-soil_data$SWIR1_R)/(soil_data$NIR_R+soil_data$SWIR1_R)

# NIR and SWIR2 
soil_data$NNIRSWIR2 <- (soil_data$NIR_R-soil_data$SWIR2_R)/(soil_data$NIR_R+soil_data$SWIR2_R)

# SWIR1 and SWIR2 (also NDSI as per the Index Database: https://www.indexdatabase.de/db/is.php?sensor_id=168 )
soil_data$NDSI1 <- (soil_data$SWIR1_R-soil_data$SWIR2_R)/(soil_data$SWIR1_R+soil_data$SWIR2_R)

# # Normalise band combinations NBR, NBG, NBNIR, NBSWIR1, NBSWIR2, NRSWIR1, NRSWIR2, NGSWIR1, NGSWIR2, NNIRSWIR1, NNIRSWIR2
# soil_data$NBR_RN <- normalize(soil_data$NBR)
# soil_data$NBG_RN <- normalize(soil_data$NBG)
# soil_data$NBNIR_RN <- normalize(soil_data$NBNIR)
# soil_data$NBSWIR1_RN <- normalize(soil_data$NBSWIR1)
# soil_data$NBSWIR2_RN <- normalize(soil_data$NBSWIR2)
# soil_data$NRSWIR1_RN <- normalize(soil_data$NRSWIR1)
# soil_data$NRSWIR2_RN <- normalize(soil_data$NRSWIR2)
# soil_data$NGSWIR1_RN <- normalize(soil_data$NGSWIR1)
# soil_data$NGSWIR2_RN <- normalize(soil_data$NGSWIR2)
# soil_data$NNIRSWIR1_RN <- normalize(soil_data$NNIRSWIR1)
# soil_data$NNIRSWIR2_RN <- normalize(soil_data$NNIRSWIR2)


# Calculate other widely used Salinity indices (listed in Nguyen et al. 2020)
# Salinity Index 1 = sqrt(green^2+red^2)
soil_data$SI1 <- sqrt((soil_data$Green_R)^2 + (soil_data$Red_R)^2) 
  
# Salinity Index 2 = sqrt(green x red)
soil_data$SI2 <- sqrt(soil_data$Green_R * soil_data$Red_R)

# Salinity Index 3 = sqrt(blue x red) 
soil_data$SI3 <- sqrt(soil_data$Blue_R * soil_data$Red_R)
  
# salinity index 4 = red x NIR / green 
soil_data$SI4 <- (soil_data$Red_R * soil_data$NIR_R / soil_data$Green_R)  

# salinity index 5 = blue/red 
soil_data$SI5 <- (soil_data$Blue_R / soil_data$Red_R)   

# Soil Adjusted Vegetation Index (SAVI) = ((1.5)x NIR) - (red/0.5) + NIR + Red 
soil_data$SAVI <- (1.5 * soil_data$NIR_R) - (0.5 * soil_data$Red_R) + soil_data$NIR_R + soil_data$Red_R

# Vegetation Soil Salinity Index (VSSI) = (2 x green) - 5 x (red + NIR) 
soil_data$VSSI <- (2 * soil_data$Green_R) - 5 * (soil_data$Red_R + soil_data$NIR_R)



# # Normalise Popular Indices: NDVI, NDWI, NDSI1, NDSI2, SI1, SI2, SI3, SI4, SI5, SAVI, VSSI
# soil_data$NDVI_RN <- normalize(soil_data$NDVI)
# soil_data$NDWI_RN <- normalize(soil_data$NDWI)
# soil_data$NDSI1_RN <- normalize(soil_data$NDSI1)
# soil_data$NDSI2_RN <- normalize(soil_data$NDSI2)
# soil_data$SI1_RN <- normalize(soil_data$SI1)
# soil_data$SI2_RN <- normalize(soil_data$SI2)
# soil_data$SI3_RN <- normalize(soil_data$SI3)
# soil_data$SI4_RN <- normalize(soil_data$SI4)
# soil_data$SI5_RN <- normalize(soil_data$SI5)
# soil_data$SAVI_RN <- normalize(soil_data$SAVI)
# soil_data$VSSI_RN <- normalize(soil_data$VSSI)


write.csv(soil_data, "data/soil_data_allindices.csv")




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Step 3. Visualize the data 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# First just simply check the linear correlation between EC and other bands/indices using a correlation matrix 
cor_matrix <- cor(soil_data[, c("EC","pH" ,"Soil_Moisture_Lab","Blue_R", "Red_R", "Green_R", "NIR_R", "SWIR1_R", "SWIR2_R",
                                "NBR", "NBG", "NBNIR", "NBSWIR1", "NBSWIR2", 
                                "NDVI", "NDSI2", "NRSWIR1", "NRSWIR2", 
                                "NDWI", "NGSWIR1", "NGSWIR2", 
                                "NNIRSWIR1", "NNIRSWIR2", "NDSI1", 
                                "SI1", "SI2", "SI3", "SI4", "SI5", "SAVI", "VSSI"
                                
                                )], use = "complete.obs")

print(cor_matrix)
write.csv(cor_matrix, "outputs/EC_bands_correlation.csv")
# As per this, the best linear predictors are NIR, NBNIR, NDSI2, NRSWIR1, NDWI, SAVI
# But that maybe because the best fit is non-linear. 
# Correlation does not change by normalising the data.
# Also, notably, while EC is strongly correlated with moisture, it is not as strongly correlated with pH


# Now try some descriptive visualizations to see the relationship between band reflectances and salinity (e.g.linear or logarithmic) 
# Create the scatter plots between EC and Bands+indices and fit different curves
# Blue Linear 
ggplot(soil_data, aes(x = Blue_R, y = EC)) +
  geom_point(color = "orange") +  # Scatter points
  geom_smooth(method = "lm", color = "black", se = FALSE) +  # Linear regression line
  labs(title = "Scatterplot of EC vs Blue Band: Linear Fit", x = "Blue", y = "EC") +
  theme_minimal()
ggsave("outputs/EC_Blue_Linear.png", width = 8, height = 6, dpi = 300, bg = "white")

# Blue Curve (smooth Loess)
ggplot(soil_data, aes(x = Blue_R, y = EC)) +
  geom_point(color = "orange") +  # Scatter points
  geom_smooth(method = "loess", color = "black", se = FALSE) +  # Smooth curve
  labs(title = "Scatterplot of EC vs Blue Band: Loess Fit", x = "Blue", y = "EC") +
  theme_minimal()
ggsave("outputs/EC_Blue_Loess.png", width = 8, height = 6, dpi = 300, bg = "white")


# Blue Curve (Generalised Additive Model - fit various curves)
ggplot(soil_data, aes(x = Blue_R, y = EC)) +
  geom_point(color = "orange") +  # Scatter points
  geom_smooth(method = "gam", color = "black", se = FALSE) +  # fitted
  labs(title = "Scatterplot of EC vs Blue Band: GAM fit", x = "Blue", y = "EC") +
  theme_minimal()
ggsave("outputs/EC_Blue_GAM.png", width = 8, height = 6, dpi = 300, bg = "white")

#looks like the best fit is linear maybe mildly quadratic

# Blue polynomial curve 
ggplot(soil_data, aes(x = Blue_R, y = EC)) +
  geom_point(color = "orange") +  # Scatter points
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), color = "black") +  # quadratic curve
  labs(title = "Scatterplot of EC vs Blue Band: Polynomial Fit", x = "Blue", y = "EC") +
  theme_minimal()
ggsave("outputs/EC_Blue_Poly.png", width = 8, height = 6, dpi = 300, bg = "white")


# Polynomial and gam fits seem like the best options. Use for the rest of the bands. 
# First fit gam, if it is of relevance, only then fit polynomial

# Red Curve (Generalised Additive Model - fit various curves)
ggplot(soil_data, aes(x = Red_R, y = EC)) +
  geom_point(color = "orange") +  # Scatter points
  geom_smooth(method = "gam", color = "black", se = FALSE) +  # fitted
  labs(title = "Scatterplot of EC vs Red Band: GAM Fit", x = "Red", y = "EC") +
  theme_minimal()
ggsave("outputs/EC_Red_GAM.png", width = 8, height = 6, dpi = 300, bg = "white")


#looks like the best fit is linear maybe mildly quadratic
# Red polynomial curve 
ggplot(soil_data, aes(x = Red_R, y = EC)) +
  geom_point(color = "orange") +  # Scatter points
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), color = "black") +  # quadratic curve
  labs(title = "Scatterplot of EC vs Red Band: Polynomial Fit", x = "Red", y = "EC") +
  theme_minimal()
ggsave("outputs/EC_Red_Poly.png", width = 8, height = 6, dpi = 300, bg = "white")


# Green Curve (Generalised Additive Model - fit various curves)
ggplot(soil_data, aes(x = Green_R, y = EC)) +
  geom_point(color = "orange") +  # Scatter points
  geom_smooth(method = "gam", color = "black", se = FALSE) +  # fitted
  labs(title = "Scatterplot of EC vs Green Band: GAM Fit", x = "Green", y = "EC") +
  theme_minimal()
ggsave("outputs/EC_Green_GAM.png", width = 8, height = 6, dpi = 300, bg = "white")

#looks like the best fit is linear maybe mildly quadratic
# Green polynomial curve 
ggplot(soil_data, aes(x = Green_R, y = EC)) +
  geom_point(color = "orange") +  # Scatter points
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), color = "black") +  # quadratic curve
  labs(title = "Scatterplot of EC vs Green Band: Polynomial Fit", x = "Green", y = "EC") +
  theme_minimal()
ggsave("outputs/EC_Green_Poly.png", width = 8, height = 6, dpi = 300, bg = "white")


# NIR Curve (Generalised Additive Model - fit various curves)
ggplot(soil_data, aes(x = NIR_R, y = EC)) +
  geom_point(color = "orange") +  # Scatter points
  geom_smooth(method = "gam", color = "black", se = FALSE) +  # fitted
  labs(title = "Scatterplot of EC vs NIR Band: GAM Fit", x = "NIR", y = "EC") +
  theme_minimal()
ggsave("outputs/EC_NIR_GAM.png", width = 8, height = 6, dpi = 300, bg = "white")

# Quadratic fit
# NIR polynomial curve 
ggplot(soil_data, aes(x = NIR_R, y = EC)) +
  geom_point(color = "orange") +  # Scatter points
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), color = "black") +  # quadratic curve
  labs(title = "Scatterplot of EC vs NIR Band: Polynomial Fit", x = "NIR", y = "EC") +
  theme_minimal()
ggsave("outputs/EC_NIR_poly.png", width = 8, height = 6, dpi = 300, bg = "white")



# SWIR1 Curve (Generalised Additive Model - fit various curves)
ggplot(soil_data, aes(x = SWIR1_R, y = EC)) +
  geom_point(color = "orange") +  # Scatter points
  geom_smooth(method = "gam", color = "black", se = FALSE) +  # fitted
  labs(title = "Scatterplot of EC vs SWIR1 Band: GAM Fit", x = "SWIR1", y = "EC") +
  theme_minimal()
ggsave("outputs/EC_SWIR1_GAM.png", width = 8, height = 6, dpi = 300, bg = "white")
# seems to have no relation


# SWIR2 Curve (Generalised Additive Model - fit various curves)
ggplot(soil_data, aes(x = SWIR2_R, y = EC)) +
  geom_point(color = "orange") +  # Scatter points
  geom_smooth(method = "gam", color = "black", se = FALSE) +  # fitted
  labs(title = "Scatterplot of EC vs SWIR2 Band: GAM Fit", x = "SWIR2", y = "EC") +
  theme_minimal()
ggsave("outputs/EC_SWIR2_GAM.png", width = 8, height = 6, dpi = 300, bg = "white")
# seems to have no relation


# Visualize indices calculated
# Blue and red 
ggplot(soil_data, aes(x = NBR, y = EC)) +
  geom_point(color = "orange") +  # Scatter points
  geom_smooth(method = "gam", color = "black", se = FALSE) +  # fitted
  labs(title = "Scatterplot of EC vs NBR: GAM Fit", x = "NBR", y = "EC") +
  theme_minimal()
ggsave("outputs/EC_NBR_GAM.png", width = 8, height = 6, dpi = 300, bg = "white")
# seems to have no relation


# NBG Blue and green 
ggplot(soil_data, aes(x = NBG, y = EC)) +
  geom_point(color = "orange") +  # Scatter points
  geom_smooth(method = "gam", color = "black", se = FALSE) +  # fitted
  labs(title = "Scatterplot of EC vs NBG: GAM Fit", x = "NBG", y = "EC") +
  theme_minimal()
ggsave("outputs/EC_NBG_GAM.png", width = 8, height = 6, dpi = 300, bg = "white")
# seems to have a linear relationship



# Blue and NIR
#gam
ggplot(soil_data, aes(x = NBNIR, y = EC)) +
  geom_point(color = "orange") +  # Scatter points
  geom_smooth(method = "gam", color = "black", se = FALSE) +  # fitted
  labs(title = "Scatterplot of EC vs NBNIR: GAM Fit", x = "NBNIR", y = "EC") +
  theme_minimal()
ggsave("outputs/EC_NBNIR_GAM.png", width = 8, height = 6, dpi = 300, bg = "white")
#definitely a quadratic relationship
#polynomial
ggplot(soil_data, aes(x = NBNIR, y = EC)) +
  geom_point(color = "orange") +  # Scatter points
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), color = "black") +  # quadratic curve
  labs(title = "Scatterplot of EC vs NBNIR: Polynomial Fit", x = "NBNIR", y = "EC") +
  theme_minimal()
ggsave("outputs/EC_NBNIR_Poly.png", width = 8, height = 6, dpi = 300, bg = "white")



# Blue and SWIR1
#gam
ggplot(soil_data, aes(x = NBSWIR1, y = EC)) +
  geom_point(color = "orange") +  # Scatter points
  geom_smooth(method = "gam", color = "black", se = FALSE) +  # fitted
  labs(title = "Scatterplot of EC vs NBSWIR1: GAM Fit ", x = "NBSWIR1", y = "EC") +
  theme_minimal()
ggsave("outputs/EC_NBSWRIR1_GAM.png", width = 8, height = 6, dpi = 300, bg = "white")
# What fit? 
#polynomial
ggplot(soil_data, aes(x = NBSWIR1, y = EC)) +
  geom_point(color = "orange") +  # Scatter points
  geom_smooth(method = "lm", formula = y ~ poly(x, 3), color = "black") +  # cubic curve
  labs(title = "Scatterplot of EC vs NBSWIR1", x = "NBSWIR1: Polynomial Fit", y = "EC") +
  theme_minimal()
ggsave("outputs/EC_NBSWRIR1_Poly.png", width = 8, height = 6, dpi = 300, bg = "white")

# Blue and SWIR2 
#gam
ggplot(soil_data, aes(x = NBSWIR2, y = EC)) +
  geom_point(color = "orange") +  # Scatter points
  geom_smooth(method = "gam", color = "black", se = FALSE) +  # fitted
  labs(title = "Scatterplot of EC vs NBSWIR2: GAM Fit ", x = "NBSWIR2", y = "EC") +
  theme_minimal()
ggsave("outputs/EC_NBSWRIR2_GAM.png", width = 8, height = 6, dpi = 300, bg = "white")
# gaussian fit?  
#polynomial
ggplot(soil_data, aes(x = NBSWIR2, y = EC)) +
  geom_point(color = "orange") +  # Scatter points
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), color = "black") +  # quadratic curve
  labs(title = "Scatterplot of EC vs NBSWIR2 : Polynomial Fit", x = "NBSWIR2", y = "EC") +
  theme_minimal()
ggsave("outputs/EC_NBSWRIR2_Poly.png", width = 8, height = 6, dpi = 300, bg = "white")


# Red and Green (also NDVI) 
# gam
ggplot(soil_data, aes(x = NDVI, y = EC)) +
  geom_point(color = "orange") +  # Scatter points
  geom_smooth(method = "gam", color = "black", se = FALSE) +  # fitted
  labs(title = "Scatterplot of EC vs NDVI: GAM Fit", x = "NDVI", y = "EC") +
  theme_minimal()
ggsave("outputs/EC_NDVI_GAM.png", width = 8, height = 6, dpi = 300, bg = "white")
# looks like a linear fit 


# Red and NIR (Normalised Difference Salinity Index 2 as per Khan et al 2001 in Nguyen et al 2020)
#gam
ggplot(soil_data, aes(x = NDSI2, y = EC)) +
  geom_point(color = "orange") +  # Scatter points
  geom_smooth(method = "gam", color = "black", se = FALSE) +  # fitted
  labs(title = "Scatterplot of EC vs NDSI2: GAM Fit", x = "NDSI2", y = "EC") +
  theme_minimal()
ggsave("outputs/EC_NDSI2_GAM.png", width = 8, height = 6, dpi = 300, bg = "white")
# Definitely a quadratic fit

#polynomial
ggplot(soil_data, aes(x = NDSI2, y = EC)) +
  geom_point(color = "orange") +  # Scatter points
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), color = "black") +  # quadratic curve
  labs(title = "Scatterplot of EC vs NDSI2: Polynomial Fit", x = "NDSI2", y = "EC") +
  theme_minimal()
ggsave("outputs/EC_NDSI2_Poly.png", width = 8, height = 6, dpi = 300, bg = "white")


# Red and SWIR1
#gam
ggplot(soil_data, aes(x = NRSWIR1, y = EC)) +
  geom_point(color = "orange") +  # Scatter points
  geom_smooth(method = "gam", color = "black", se = FALSE) +  # fitted
  labs(title = "Scatterplot of EC vs NRSWIR1: GAM Fit", x = "NRSWIR1", y = "EC") +
  theme_minimal()
ggsave("outputs/EC_NRSWIR1_GAM.png", width = 8, height = 6, dpi = 300, bg = "white") 

#polynomial
ggplot(soil_data, aes(x = NRSWIR1, y = EC)) +
  geom_point(color = "orange") +  # Scatter points
  geom_smooth(method = "lm", formula = y ~ poly(x, 3), color = "black") +  # cubic curve
  labs(title = "Scatterplot of EC vs NRSWIR1: Polynomial Fit", x = "NRSWIR1", y = "EC") +
  theme_minimal()
ggsave("outputs/EC_NRSWIR1_Poly.png", width = 8, height = 6, dpi = 300, bg = "white")


# Red and SWIR2
#gam
ggplot(soil_data, aes(x = NRSWIR2, y = EC)) +
  geom_point(color = "orange") +  # Scatter points
  geom_smooth(method = "gam", color = "black", se = FALSE) +  # fitted
  labs(title = "Scatterplot of EC vs NRSWIR2: GAM Fit", x = "NRSWIR2", y = "EC") +
  theme_minimal()
ggsave("outputs/EC_NRSWIR2_GAM.png", width = 8, height = 6, dpi = 300, bg = "white")
# gaussian fit? 

#polynomial
ggplot(soil_data, aes(x = NRSWIR2, y = EC)) +
  geom_point(color = "orange") +  # Scatter points
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), color = "black") +  # quadratic curve
  labs(title = "Scatterplot of EC vs NRSWIR2: Polynomial", x = "NRSWIR2", y = "EC") +
  theme_minimal()
ggsave("outputs/EC_NRSWIR2_Poly.png", width = 8, height = 6, dpi = 300, bg = "white")


# Green and NIR (also NDWI) 
#gam
ggplot(soil_data, aes(x = NDWI, y = EC)) +
  geom_point(color = "orange") +  # Scatter points
  geom_smooth(method = "gam", color = "black", se = FALSE) +  # fitted
  labs(title = "Scatterplot of EC vs NDWI: GAM Fit", x = "NDWI", y = "EC") +
  theme_minimal()
ggsave("outputs/EC_NDWI_GAM.png", width = 8, height = 6, dpi = 300, bg = "white")

#polynomial
ggplot(soil_data, aes(x = NDWI, y = EC)) +
  geom_point(color = "orange") +  # Scatter points
  geom_smooth(method = "lm", formula = y ~ poly(x, 3), color = "black") +  # cubic curve
  labs(title = "Scatterplot of EC vs NDWI: Polynomial Fit", x = "NDWI", y = "EC") +
  theme_minimal()
ggsave("outputs/EC_NDWI_Poly.png", width = 8, height = 6, dpi = 300, bg = "white")

# Green and SWIR1
#gam
ggplot(soil_data, aes(x = NGSWIR1, y = EC)) +
  geom_point(color = "orange") +  # Scatter points
  geom_smooth(method = "gam", color = "black", se = FALSE) +  # fitted
  labs(title = "Scatterplot of EC vs NGSWIR1: GAM Fit", x = "NGSWIR1", y = "EC") +
  theme_minimal()
ggsave("outputs/EC_NGSWIR1_GAM.png", width = 8, height = 6, dpi = 300, bg = "white")

# Gaussian fit? 
#polynomial
ggplot(soil_data, aes(x = NGSWIR1, y = EC)) +
  geom_point(color = "orange") +  # Scatter points
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), color = "black") +  # quadratic curve
  labs(title = "Scatterplot of EC vs NGSWIR1: Polynomial Fit", x = "NGSWIR1", y = "EC") +
  theme_minimal()
ggsave("outputs/EC_NGSWIR1_Poly.png", width = 8, height = 6, dpi = 300, bg = "white")

# Green and SWIR2
#gam
ggplot(soil_data, aes(x = NGSWIR2, y = EC)) +
  geom_point(color = "orange") +  # Scatter points
  geom_smooth(method = "gam", color = "black", se = FALSE) +  # fitted
  labs(title = "Scatterplot of EC vs NGSWIR2: GAM Fit", x = "NGSWIR2", y = "EC") +
  theme_minimal()
ggsave("outputs/EC_NGSWIR2_GAM.png", width = 8, height = 6, dpi = 300, bg = "white")
#Gaussian fit? 

#polynomial
ggplot(soil_data, aes(x = NGSWIR2, y = EC)) +
  geom_point(color = "orange") +  # Scatter points
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), color = "black") +  # quadratic curve
  labs(title = "Scatterplot of EC vs NGSWIR2: Polynomial Fit", x = "NGSWIR2", y = "EC") +
  theme_minimal()
ggsave("outputs/EC_NGSWIR2_Poly.png", width = 8, height = 6, dpi = 300, bg = "white")


# NIR and SWIR1
#gam
ggplot(soil_data, aes(x = NNIRSWIR1 , y = EC)) +
  geom_point(color = "orange") +  # Scatter points
  geom_smooth(method = "gam", color = "black", se = FALSE) +  # fitted
  labs(title = "Scatterplot of EC vs NNIRSWIR1: GAM Fit ", x = "NNIRSWIR1 ", y = "EC") +
  theme_minimal()
ggsave("outputs/EC_NNIRSWIR1_GAM.png", width = 8, height = 6, dpi = 300, bg = "white")
# mostly linear but a little exponential? 
#polynomial
ggplot(soil_data, aes(x = NNIRSWIR1 , y = EC)) +
  geom_point(color = "orange") +  # Scatter points
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), color = "black") +  # quadratic curve
  labs(title = "Scatterplot of EC vs NNIRSWIR1: Polynomial Fit ", x = "NNIRSWIR1 ", y = "EC") +
  theme_minimal()
ggsave("outputs/EC_NNIRSWIR1_Poly.png", width = 8, height = 6, dpi = 300, bg = "white")


# NIR and SWIR2 
#gam
ggplot(soil_data, aes(x = NNIRSWIR2 , y = EC)) +
  geom_point(color = "orange") +  # Scatter points
  geom_smooth(method = "gam", color = "black", se = FALSE) +  # fitted
  labs(title = "Scatterplot of EC vs NNIRSWIR2: GAM Fit ", x = "NNIRSWIR2 ", y = "EC") +
  theme_minimal()
ggsave("outputs/EC_NNIRSWIR2_GAM.png", width = 8, height = 6, dpi = 300, bg = "white")
# mostly linear but a little exponential? 
#polynomial
ggplot(soil_data, aes(x = NNIRSWIR2 , y = EC)) +
  geom_point(color = "orange") +  # Scatter points
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), color = "black") +  # quadratic curve
  labs(title = "Scatterplot of EC vs NNIRSWIR2: Polynomial Fit", x = "NNIRSWIR2 ", y = "EC") +
  theme_minimal()
ggsave("outputs/EC_NNIRSWIR2_Poly.png", width = 8, height = 6, dpi = 300, bg = "white")

# SWIR1 and SWIR2 (also NDSI as per the Index Database: https://www.indexdatabase.de/db/is.php?sensor_id=168 )
#gam
ggplot(soil_data, aes(x = NDSI1 , y = EC)) +
  geom_point(color = "orange") +  # Scatter points
  geom_smooth(method = "gam", color = "black", se = FALSE) +  # fitted
  labs(title = "Scatterplot of EC vs NDSI1: GAM Fit ", x = "NDSI1", y = "EC") +
  theme_minimal()
ggsave("outputs/EC_NDSI1_GAM.png", width = 8, height = 6, dpi = 300, bg = "white")
# essentially linear

ggplot(soil_data, aes(x = NDSI1 , y = EC)) +
  geom_point(color = "orange") +  # Scatter points
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), color = "black") +  # quadratic curve
  labs(title = "Scatterplot of EC vs NDSI1: Polynomial Fit", x = "NDSI1 ", y = "EC") +
  theme_minimal()
ggsave("outputs/EC_NDSI1_Poly.png", width = 8, height = 6, dpi = 300, bg = "white")



# Other widely used Salinity indices (listed in Nguyen et al. 2020)
# Salinity Index 1 = sqrt(green^2+red^2)
#gam
ggplot(soil_data, aes(x = SI1 , y = EC)) +
  geom_point(color = "orange") +  # Scatter points
  geom_smooth(method = "gam", color = "black", se = FALSE) +  # fitted
  labs(title = "Scatterplot of EC vs SI1: GAM Fit ", x = "SI1", y = "EC") +
  theme_minimal()
ggsave("outputs/EC_SI1_GAM.png", width = 8, height = 6, dpi = 300, bg = "white")
# linear with slight quadratic fit
#polynomial
ggplot(soil_data, aes(x = SI1 , y = EC)) +
  geom_point(color = "orange") +  # Scatter points
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), color = "black") +  # quadratic curve
  labs(title = "Scatterplot of EC vs SI1: Polynomial Fit ", x = "SI1", y = "EC") +
  theme_minimal()
ggsave("outputs/EC_SI1_Poly.png", width = 8, height = 6, dpi = 300, bg = "white")

# Salinity Index 2 = sqrt(green x red)
#gam
ggplot(soil_data, aes(x = SI2 , y = EC)) +
  geom_point(color = "orange") +  # Scatter points
  geom_smooth(method = "gam", color = "black", se = FALSE) +  # fitted
  labs(title = "Scatterplot of EC vs SI2: GAM Fit", x = "SI2", y = "EC") +
  theme_minimal()
ggsave("outputs/EC_SI2_GAM.png", width = 8, height = 6, dpi = 300, bg = "white")
# linear with slight quadratic fit
#polynomial
ggplot(soil_data, aes(x = SI2 , y = EC)) +
  geom_point(color = "orange") +  # Scatter points
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), color = "black") +  # quadratic curve
  labs(title = "Scatterplot of EC vs SI2: Polynomial Fit ", x = "SI2", y = "EC") +
  theme_minimal()
ggsave("outputs/EC_SI2_Poly.png", width = 8, height = 6, dpi = 300, bg = "white")


# Salinity Index 3 = sqrt(blue x red) 
#gam
ggplot(soil_data, aes(x = SI3 , y = EC)) +
  geom_point(color = "orange") +  # Scatter points
  geom_smooth(method = "gam", color = "black", se = FALSE) +  # fitted
  labs(title = "Scatterplot of EC vs SI3: GAM Fit", x = "SI3", y = "EC") +
  theme_minimal()
ggsave("outputs/EC_SI3_GAM.png", width = 8, height = 6, dpi = 300, bg = "white")
# essentially linear



# salinity index 4 = red x NIR / green 
#gam
ggplot(soil_data, aes(x = SI4 , y = EC)) +
  geom_point(color = "orange") +  # Scatter points
  geom_smooth(method = "gam", color = "black", se = FALSE) +  # fitted
  labs(title = "Scatterplot of EC vs SI4: GAM Fit", x = "SI4", y = "EC") +
  theme_minimal()
ggsave("outputs/EC_SI4_GAM.png", width = 8, height = 6, dpi = 300, bg = "white")
# essentially linear


# salinity index 5 = blue/red 
#gam
ggplot(soil_data, aes(x = SI5 , y = EC)) +
  geom_point(color = "orange") +  # Scatter points
  geom_smooth(method = "gam", color = "black", se = FALSE) +  # fitted
  labs(title = "Scatterplot of EC vs SI5: GAM Fit", x = "SI5", y = "EC") +
  theme_minimal()
ggsave("outputs/EC_SI5_GAM.png", width = 8, height = 6, dpi = 300, bg = "white")
# no relationship


# Soil Adjusted Vegetation Index (SAVI) = ((1.5)x NIR) - (red/0.5) + NIR + Red 
#gam
ggplot(soil_data, aes(x = SAVI , y = EC)) +
  geom_point(color = "orange") +  # Scatter points
  geom_smooth(method = "gam", color = "black", se = FALSE) +  # fitted
  labs(title = "Scatterplot of EC vs SAVI: GAM Fit", x = "SAVI", y = "EC") +
  theme_minimal()
ggsave("outputs/EC_SAVI_GAM.png", width = 8, height = 6, dpi = 300, bg = "white")
# quadratic?
#polynomial
ggplot(soil_data, aes(x = SAVI , y = EC)) +
  geom_point(color = "orange") +  # Scatter points
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), color = "black") +  # quadratic curve
  labs(title = "Scatterplot of EC vs SAVI: Polynomial Fit ", x = "SAVI", y = "EC") +
  theme_minimal()
ggsave("outputs/EC_SAVI_Poly.png", width = 8, height = 6, dpi = 300, bg = "white")


# Vegetation Soil Salinity Index (VSSI) = (2 x green) - 5 x (red + NIR) 
#gam
ggplot(soil_data, aes(x = VSSI , y = EC)) +
  geom_point(color = "orange") +  # Scatter points
  geom_smooth(method = "gam", color = "black", se = FALSE) +  # fitted
  labs(title = "Scatterplot of EC vs VSSI: GAM Fit", x = "VSSI", y = "EC") +
  theme_minimal()
ggsave("outputs/EC_VSSI_GAM.png", width = 8, height = 6, dpi = 300, bg = "white")
# essentially linear fit
# poly
ggplot(soil_data, aes(x = VSSI , y = EC)) +
  geom_point(color = "orange") +  # Scatter points
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), color = "black") +  # quadratic curve
  labs(title = "Scatterplot of EC vs VSSI: Polynomial Fit ", x = "VSSI", y = "EC") +
  theme_minimal()
ggsave("outputs/EC_VSSI_Poly.png", width = 8, height = 6, dpi = 300, bg = "white")


# Plot polynomial relationship of the select indices with moisture
# NDWI
ggplot(soil_data, aes(x = NDWI , y = Soil_Moisture_Lab)) +
  geom_point(color = "blue") +  # Scatter points
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), color = "black") +  # quadratic curve
  labs(title = "Scatterplot of Moisture vs NDWI: Polynomial Fit ", x = "NDWI", y = "Soil Moisture") +
  theme_minimal()
ggsave("outputs/Moisture_NDWI_Poly.png", width = 8, height = 6, dpi = 300, bg = "white")

# NBNIR
ggplot(soil_data, aes(x = NBNIR , y = Soil_Moisture_Lab)) +
  geom_point(color = "blue") +  # Scatter points
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), color = "black") +  # quadratic curve
  labs(title = "Scatterplot of Moisture vs NBNIR: Polynomial Fit ", x = "NBNIR", y = "Soil Moisture") +
  theme_minimal()
ggsave("outputs/Moisture_NBNIR_Poly.png", width = 8, height = 6, dpi = 300, bg = "white")


# SWIR2_R
ggplot(soil_data, aes(x = SWIR2_R , y = Soil_Moisture_Lab)) +
  geom_point(color = "blue") +  # Scatter points
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), color = "black") +  # quadratic curve
  labs(title = "Scatterplot of Moisture vs SWIR2: Polynomial Fit ", x = "SWIR2", y = "Soil Moisture") +
  theme_minimal()
ggsave("outputs/Moisture_SWIR2_Poly.png", width = 8, height = 6, dpi = 300, bg = "white")


# Green_R
ggplot(soil_data, aes(x = Green_R , y = Soil_Moisture_Lab)) +
  geom_point(color = "blue") +  # Scatter points
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), color = "black") +  # quadratic curve
  labs(title = "Scatterplot of Moisture vs Green: Polynomial Fit ", x = "Green", y = "Soil Moisture") +
  theme_minimal()
ggsave("outputs/Moisture_Green_Poly.png", width = 8, height = 6, dpi = 300, bg = "white")


# NRSWIR1
ggplot(soil_data, aes(x = NRSWIR1 , y = Soil_Moisture_Lab)) +
  geom_point(color = "blue") +  # Scatter points
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), color = "black") +  # quadratic curve
  labs(title = "Scatterplot of Moisture vs NRSWIR1: Polynomial Fit ", x = "NRSWIR1", y = "Soil Moisture") +
  theme_minimal()
ggsave("outputs/Moisture_NRSWIR1_Poly.png", width = 8, height = 6, dpi = 300, bg = "white")


# NRSWIR1
ggplot(soil_data, aes(x = NRSWIR1 , y = Soil_Moisture_Lab)) +
  geom_point(color = "blue") +  # Scatter points
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), color = "black") +  # quadratic curve
  labs(title = "Scatterplot of Moisture vs NRSWIR1: Polynomial Fit ", x = "NRSWIR1", y = "Soil Moisture") +
  theme_minimal()
ggsave("outputs/Moisture_NRSWIR1_Poly.png", width = 8, height = 6, dpi = 300, bg = "white")


# VSSI
ggplot(soil_data, aes(x = VSSI , y = Soil_Moisture_Lab)) +
  geom_point(color = "blue") +  # Scatter points
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), color = "black") +  # quadratic curve
  labs(title = "Scatterplot of Moisture vs VSSI: Polynomial Fit ", x = "VSSI", y = "Soil Moisture") +
  theme_minimal()
ggsave("outputs/Moisture_VSSI_Poly.png", width = 8, height = 6, dpi = 300, bg = "white")


# VSSI
ggplot(soil_data, aes(x = VSSI , y = Soil_Moisture_Lab)) +
  geom_point(color = "blue") +  # Scatter points
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), color = "black") +  # quadratic curve
  labs(title = "Scatterplot of Moisture vs VSSI: Polynomial Fit ", x = "VSSI", y = "Soil Moisture") +
  theme_minimal()
ggsave("outputs/Moisture_VSSI_Poly.png", width = 8, height = 6, dpi = 300, bg = "white")


# NNIRSWIR1
ggplot(soil_data, aes(x = NNIRSWIR1 , y = Soil_Moisture_Lab)) +
  geom_point(color = "blue") +  # Scatter points
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), color = "black") +  # quadratic curve
  labs(title = "Scatterplot of Moisture vs NNIRSWIR1: Polynomial Fit ", x = "NNIRSWIR1", y = "Soil Moisture") +
  theme_minimal()
ggsave("outputs/Moisture_NNIRSWIR1_Poly.png", width = 8, height = 6, dpi = 300, bg = "white")


# NDSI1
ggplot(soil_data, aes(x = NDSI1 , y = Soil_Moisture_Lab)) +
  geom_point(color = "blue") +  # Scatter points
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), color = "black") +  # quadratic curve
  labs(title = "Scatterplot of Moisture vs NDSI1: Polynomial Fit ", x = "NDSI1", y = "Soil Moisture") +
  theme_minimal()
ggsave("outputs/Moisture_NDSI1_Poly.png", width = 8, height = 6, dpi = 300, bg = "white")


# NGSWIR1
ggplot(soil_data, aes(x = NGSWIR1 , y = Soil_Moisture_Lab)) +
  geom_point(color = "blue") +  # Scatter points
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), color = "black") +  # quadratic curve
  labs(title = "Scatterplot of Moisture vs NGSWIR1: Polynomial Fit ", x = "NGSWIR1", y = "Soil Moisture") +
  theme_minimal()
ggsave("outputs/Moisture_NGSWIR1_Poly.png", width = 8, height = 6, dpi = 300, bg = "white")


# NBSWIR2
ggplot(soil_data, aes(x = NBSWIR2 , y = Soil_Moisture_Lab)) +
  geom_point(color = "blue") +  # Scatter points
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), color = "black") +  # quadratic curve
  labs(title = "Scatterplot of Moisture vs NBSWIR2: Polynomial Fit ", x = "NBSWIR2", y = "Soil Moisture") +
  theme_minimal()
ggsave("outputs/Moisture_NBSWIR2_Poly.png", width = 8, height = 6, dpi = 300, bg = "white")





#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Step 4. Conduct statistical correlation and model fit for 80% of field data points 
# Step 5. Conduct accuracy testing with the remaining 20% points 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Make a 80-20 split data for training and testing 
# Set seed for reproducibility
set.seed(123)  # set a seed number to repeat the same each time

# Calculate the number of rows for 80% of the data
train_size <- floor(0.8 * nrow(soil_data))

# Randomly shuffle row indices
shuffled_indices <- sample(seq_len(nrow(soil_data)))

# Split the data
train_indices <- shuffled_indices[1:train_size]
test_indices <- shuffled_indices[(train_size + 1):nrow(soil_data)]

# Create the 80% and 20% dataframes
train_data <- soil_data[train_indices, ]
test_data <- soil_data[test_indices, ]

# View the results
head(train_data)
head(test_data)


# Save soil data with all the indices and the train and test data as csv 

# write.csv(soil_data, "data/soil_data_reflectance.csv")
# write.csv(train_data, "data/train_data.csv")
# write.csv(test_data, "data/test_data.csv")

##### Conduct the statistical modelling with the train data #####
# Fit different models for each predictor and collate their R-squared, p-value, AIC and BIC
# For ref: 
# Adjusted R-squared adjusts for the number of predictors and is useful for comparing models with different numbers of terms.
# AIC (Akaike Information Criterion) helps to compare models and penalizes more complex models. Lower AIC values indicate a better fit.
# Note: Logistic models need EC to be either 0 or 1 or a proportion between 0 and 1. So logistic model is not suitable for these relationships

# Popular Indices: NDVI, NDWI, NDSI1, NDSI2, SI1, SI2, SI3, SI4, SI5, SAVI, VSSI
# Additional Normalised Band Combinations NBR, NBG, NBNIR, NBSWIR1, NBSWIR2, NRSWIR1, NRSWIR2, NGSWIR1, NGSWIR2, NNIRSWIR1, NNIRSWIR2

#~~~~~
# IMPORTANT
# Pick an appropriate y metric here to run the following analysis (EC_bin, EC_cat, EC_all)

#EC as continuous
soil_data$EC <- soil_data$EC_all
train_data$EC <- train_data$EC_all
test_data$EC <- test_data$EC_all

# OR 

#EC as binary
soil_data$EC <- soil_data$EC_bin
train_data$EC <- train_data$EC_bin
test_data$EC <- test_data$EC_bin

# OR 

#EC as categorical (high, medium, low variables)
soil_data$EC <- soil_data$EC_cat
train_data$EC <- train_data$EC_cat
test_data$EC <- test_data$EC_cat
#~~~~~


##### LINEAR MODEL FITS #####
# Bands
linear_Blue <- lm(EC ~ Blue_R, data = train_data)
linear_Red <- lm(EC ~ Red_R, data = train_data)
linear_Green <- lm(EC ~ Green_R, data = train_data)
linear_NIR <- lm(EC ~ NIR_R, data = train_data)
linear_SWIR1 <- lm(EC ~ SWIR1_R, data = train_data)
linear_SWIR2 <- lm(EC ~ SWIR2_R, data = train_data)

models <- list()
models[['Blue']] <- linear_Blue
models[['Red']] <- linear_Red
models[['Green']] <- linear_Green
models[['NIR']] <- linear_NIR
models[['SWIR-1']] <- linear_SWIR1
models[['SWIR-2']] <- linear_SWIR2

msummary(models, stars = c('*' = .1, '**' = .05, '***' = .01), filename = 'table.rtf')

# Note: Changing to normalized bands does nothing to R2. Returning to use Reflactances as is. 

# Popular Indices : NDVI, NDWI, NDSI1, NDSI2, SI1, SI2, SI3, SI4, SI5, SAVI, VSSI
linear_NDVI <- lm(EC ~ NDVI, data = train_data)
linear_NDWI <- lm(EC ~ NDWI, data = train_data)
linear_NDSI1 <- lm(EC ~ NDSI1, data = train_data)
linear_NDSI2 <- lm(EC ~ NDSI2, data = train_data)
linear_SI1 <- lm(EC ~ SI1, data = train_data)
linear_SI2 <- lm(EC ~ SI2, data = train_data)
linear_SI3 <- lm(EC ~ SI3, data = train_data)
linear_SI4 <- lm(EC ~ SI4, data = train_data)
linear_SI5 <- lm(EC ~ SI5, data = train_data)
linear_SAVI <- lm(EC ~ SAVI, data = train_data)
linear_VSSI <- lm(EC ~ VSSI, data = train_data)

models <- list()
models[['NDVI']] <- linear_NDVI
models[['NDWI']] <- linear_NDWI
models[['NDSI1']] <- linear_NDSI1
models[['NDSI2']] <- linear_NDSI2
models[['SI1']] <- linear_SI1
models[['SI2']] <- linear_SI2
models[['SI3']] <- linear_SI3
models[['SI4']] <- linear_SI4
models[['SI5']] <- linear_SI5
models[['SAVI']] <- linear_SAVI
models[['VSSI']] <- linear_VSSI

msummary(models, stars = c('*' = .1, '**' = .05, '***' = .01), filename = 'table.rtf')


# Additional combinations: NBR, NBG, NBNIR, NBSWIR1, NBSWIR2, NRSWIR1, NRSWIR2, NGSWIR1, NGSWIR2, NNIRSWIR1, NNIRSWIR2
linear_NBR <- lm(EC ~ NBR, data = train_data)
linear_NBG <- lm(EC ~ NBG, data = train_data)
linear_NBNIR <- lm(EC ~ NBNIR, data = train_data)
linear_NBSWIR1 <- lm(EC ~ NBSWIR1, data = train_data)
linear_NBSWIR2 <- lm(EC ~ NBSWIR2, data = train_data)
linear_NRSWIR1 <- lm(EC ~ NRSWIR1, data = train_data)
linear_NRSWIR2 <- lm(EC ~ NRSWIR2, data = train_data)
linear_NGSWIR1 <- lm(EC ~ NGSWIR1, data = train_data)
linear_NGSWIR2 <- lm(EC ~ NGSWIR2, data = train_data)
linear_NNIRSWIR1 <- lm(EC ~ NNIRSWIR1, data = train_data)
linear_NNIRSWIR2 <- lm(EC ~ NNIRSWIR2, data = train_data)

models <- list()
models[['NBR']] <- linear_NBR
models[['NBG']] <- linear_NBG
models[['NBNIR']] <- linear_NBNIR
models[['NBSWIR1']] <- linear_NBSWIR1
models[['NBSWIR2']] <- linear_NBSWIR2
models[['NRSWIR1']] <- linear_NRSWIR1
models[['NRSWIR2']] <- linear_NRSWIR2
models[['NGSWIR1']] <- linear_NGSWIR1
models[['NGSWIR2']] <- linear_NGSWIR2
models[['NNIRSWIR1']] <- linear_NNIRSWIR1
models[['NNIRSWIR2']] <- linear_NNIRSWIR2

msummary(models, stars = c('*' = .1, '**' = .05, '***' = .01), filename = 'table.rtf')



##### POLYNOMIAL MODEL FITS #####
# Bands
poly_Blue <- lm(EC ~ poly(Blue_R,2), data = train_data)
poly_Red <- lm(EC ~ poly(Red_R,2), data = train_data)
poly_Green <- lm(EC ~ poly(Green_R,2), data = train_data)
poly_NIR <- lm(EC ~ poly(NIR_R,2), data = train_data)
poly_SWIR1 <- lm(EC ~ poly(SWIR1_R,2), data = train_data)
poly_SWIR2 <- lm(EC ~ poly(SWIR2_R,2), data = train_data)

models <- list()
models[['Blue']] <- poly_Blue
models[['Red']] <- poly_Red
models[['Green']] <- poly_Green
models[['NIR']] <- poly_NIR
models[['SWIR-1']] <- poly_SWIR1
models[['SWIR-2']] <- poly_SWIR2

msummary(models, stars = c('*' = .1, '**' = .05, '***' = .01), filename = 'table.rtf')
# Indicates that Red and Green fit better as linear than quadratic, but NIR overall model fit is better with quadratic. 
# Note: using normalized bands does nothing to R2. Using reflanctances to keep the interpretation easier. 


# Popular Indices : NDVI, NDWI, NDSI1, NDSI2, SI1, SI2, SI3, SI4, SI5, SAVI, VSSI
poly_NDVI <- lm(EC ~ poly(NDVI,2), data = train_data)
poly_NDWI <- lm(EC ~ poly(NDWI,2), data = train_data)
poly_NDSI1 <- lm(EC ~ poly(NDSI1,2), data = train_data)
poly_NDSI2 <- lm(EC ~ poly(NDSI2,2), data = train_data)
poly_SI1 <- lm(EC ~ poly(SI1,2), data = train_data)
poly_SI2 <- lm(EC ~ poly(SI2,2), data = train_data)
poly_SI3 <- lm(EC ~ poly(SI3,2), data = train_data)
poly_SI4 <- lm(EC ~ poly(SI4,2), data = train_data)
poly_SI5 <- lm(EC ~ poly(SI5,2), data = train_data)
poly_SAVI <- lm(EC ~ poly(SAVI,2), data = train_data)
poly_VSSI <- lm(EC ~ poly(VSSI,2), data = train_data)

models <- list()
models[['NDVI']] <- poly_NDVI
models[['NDWI']] <- poly_NDWI
models[['NDSI1']] <- poly_NDSI1
models[['NDSI2']] <- poly_NDSI2
models[['SI1']] <- poly_SI1
models[['SI2']] <- poly_SI2
models[['SI3']] <- poly_SI3
models[['SI4']] <- poly_SI4
models[['SI5']] <- poly_SI5
models[['SAVI']] <- poly_SAVI
models[['VSSI']] <- poly_VSSI

msummary(models, stars = c('*' = .1, '**' = .05, '***' = .01), filename = 'table.rtf')


# Additional combinations: NBR, NBG, NBNIR, NBSWIR1, NBSWIR2, NRSWIR1, NRSWIR2, NGSWIR1, NGSWIR2,NNIRSWIR1, NNIRSWIR2
poly_NBR <- lm(EC ~ poly(NBR,2), data = train_data)
poly_NBG <- lm(EC ~ poly(NBG,2), data = train_data)
poly_NBNIR <- lm(EC ~ poly(NBNIR,2), data = train_data)
poly_NBSWIR1 <- lm(EC ~ poly(NBSWIR1,2), data = train_data)
poly_NBSWIR2 <- lm(EC ~ poly(NBSWIR2,2), data = train_data)
poly_NRSWIR1 <- lm(EC ~ poly(NRSWIR1,2), data = train_data)
poly_NRSWIR2 <- lm(EC ~ poly(NRSWIR2,2), data = train_data)
poly_NGSWIR1 <- lm(EC ~ poly(NGSWIR1,2), data = train_data)
poly_NGSWIR2 <- lm(EC ~ poly(NGSWIR2,2), data = train_data)
poly_NNIRSWIR1 <- lm(EC ~ poly(NNIRSWIR1,2), data = train_data)
poly_NNIRSWIR2 <- lm(EC ~ poly(NNIRSWIR2,2), data = train_data)

models <- list()
models[['NBR']] <- poly_NBR
models[['NBG']] <- poly_NBG
models[['NBNIR']] <- poly_NBNIR
models[['NBSWIR1']] <- poly_NBSWIR1
models[['NBSWIR2']] <- poly_NBSWIR2
models[['NRSWIR1']] <- poly_NRSWIR1
models[['NRSWIR2']] <- poly_NRSWIR2
models[['NGSWIR1']] <- poly_NGSWIR1
models[['NGSWIR2']] <- poly_NGSWIR2
models[['NNIRSWIR1']] <- poly_NNIRSWIR1
models[['NNIRSWIR2']] <- poly_NNIRSWIR2

msummary(models, stars = c('*' = .1, '**' = .05, '***' = .01), filename = 'table.rtf')



##### LOGARITHMIC FITS (NGUYEN ET AL. 2020 MODELS) #####
# Log function results interpretation [EC = (e^β₀) * e^(β₁ * band)]
# !! Will not work for EC_bin and EC_cat  
# Bands 
log_Blue <- lm(log(EC) ~ Blue_R, data = train_data)
log_Red <- lm(log(EC) ~ Red_R, data = train_data)
log_Green <- lm(log(EC) ~ Green_R, data = train_data)
log_NIR <- lm(log(EC) ~ NIR_R, data = train_data)
log_SWIR1 <- lm(log(EC) ~ SWIR1_R, data = train_data)
log_SWIR2 <- lm(log(EC) ~ SWIR2_R, data = train_data)

models <- list()
models[['Blue']] <- log_Blue
models[['Red']] <- log_Red
models[['Green']] <- log_Green
models[['NIR']] <- log_NIR
models[['SWIR-1']] <- log_SWIR1
models[['SWIR-2']] <- log_SWIR2

msummary(models, stars = c('*' = .1, '**' = .05, '***' = .01), filename = 'table.rtf')

# Note: Using normalized bands does nothing to the R2. Using reflactance values to keep the interpretation easier

# Popular Indices : NDVI, NDWI, NDSI1, NDSI2, SI1, SI2, SI3, SI4, SI5, SAVI, VSSI
log_NDVI <- lm(log(EC) ~ NDVI, data = train_data)
log_NDWI <- lm(log(EC) ~ NDWI, data = train_data)
log_NDSI1 <- lm(log(EC) ~ NDSI1, data = train_data)
log_NDSI2 <- lm(log(EC) ~ NDSI2, data = train_data)
log_SI1 <- lm(log(EC) ~ SI1, data = train_data)
log_SI2 <- lm(log(EC) ~ SI2, data = train_data)
log_SI3 <- lm(log(EC) ~ SI3, data = train_data)
log_SI4 <- lm(log(EC) ~ SI4, data = train_data)
log_SI5 <- lm(log(EC) ~ SI5, data = train_data)
log_SAVI <- lm(log(EC) ~ SAVI, data = train_data)
log_VSSI <- lm(log(EC) ~ VSSI, data = train_data)

models <- list()
models[['NDVI']] <- log_NDVI
models[['NDWI']] <- log_NDWI
models[['NDSI1']] <- log_NDSI1
models[['NDSI2']] <- log_NDSI2
models[['SI1']] <- log_SI1
models[['SI2']] <- log_SI2
models[['SI3']] <- log_SI3
models[['SI4']] <- log_SI4
models[['SI5']] <- log_SI5
models[['SAVI']] <- log_SAVI
models[['VSSI']] <- log_VSSI

msummary(models, stars = c('*' = .1, '**' = .05, '***' = .01), filename = 'table.rtf')


# Additional combinations: NBR, NBG, NBNIR, NBSWIR1, NBSWIR2, NRSWIR1, NRSWIR2, NGSWIR1, NGSWIR2, NNIRSWIR1, NNIRSWIR2
log_NBR <- lm(log(EC) ~ NBR, data = train_data)
log_NBG <- lm(log(EC) ~ NBG, data = train_data)
log_NBNIR <- lm(log(EC) ~ NBNIR, data = train_data)
log_NBSWIR1 <- lm(log(EC) ~ NBSWIR1, data = train_data)
log_NBSWIR2 <- lm(log(EC) ~ NBSWIR2, data = train_data)
log_NRSWIR1 <- lm(log(EC) ~ NRSWIR1, data = train_data)
log_NRSWIR2 <- lm(log(EC) ~ NRSWIR2, data = train_data)
log_NGSWIR1 <- lm(log(EC) ~ NGSWIR1, data = train_data)
log_NGSWIR2 <- lm(log(EC) ~ NGSWIR2, data = train_data)
log_NNIRSWIR1 <- lm(log(EC) ~ NNIRSWIR1, data = train_data)
log_NNIRSWIR2 <- lm(log(EC) ~ NNIRSWIR2, data = train_data)

models <- list()
models[['NBR']] <- log_NBR
models[['NBG']] <- log_NBG
models[['NBNIR']] <- log_NBNIR
models[['NBSWIR1']] <- log_NBSWIR1
models[['NBSWIR2']] <- log_NBSWIR2
models[['NRSWIR1']] <- log_NRSWIR1
models[['NRSWIR2']] <- log_NRSWIR2
models[['NGSWIR1']] <- log_NGSWIR1
models[['NGSWIR2']] <- log_NGSWIR2
models[['NNIRSWIR1']] <- log_NNIRSWIR1
models[['NNIRSWIR2']] <- log_NNIRSWIR2

msummary(models, stars = c('*' = .1, '**' = .05, '***' = .01), filename = 'table.rtf')



##### All bands in one model #####
linear_all <- lm(EC ~ Blue_R + Red_R + Green_R + NIR_R + SWIR1_R + SWIR2_R + 
                   NDVI + NDWI + NDSI1 + NDSI2 + SI1 + SI2 + SI3 + SI4 + SI5 + SAVI + VSSI + 
                   NBR + NBG + NBNIR + NBSWIR1 + NBSWIR2 + NRSWIR1 + NRSWIR2 + NGSWIR1 + NGSWIR2 + NNIRSWIR1 + NNIRSWIR2, data = train_data)

poly_all <- lm(EC ~ poly(Blue_R,2) + poly(Red_R,2) + poly(Green_R,2) + poly(NIR_R,2) + poly(SWIR1_R,2) + poly(SWIR2_R,2) + 
                 poly(NDVI,2) + poly(NDWI,2) + poly(NDSI1,2) + poly(NDSI2,2) + poly(SI1,2) + poly(SI2,2) + poly(SI3,2) + poly(SI4,2) + poly(SI5,2) + poly(SAVI,2) + poly(VSSI,2) + 
                 poly(NBR,2) + poly(NBG,2) + poly(NBNIR,2) + poly(NBSWIR1,2) + poly(NBSWIR2,2) + poly(NRSWIR1,2) + poly(NRSWIR2,2) + poly(NGSWIR1,2) + poly(NGSWIR2,2) + poly(NNIRSWIR1,2) + poly(NNIRSWIR2,2) , data = train_data)

#Note: log will not work on EC_bin and EC_cat
log_all <- lm(log(EC) ~ Blue_R + Red_R + Green_R + NIR_R + SWIR1_R + SWIR2_R + 
                NDVI + NDWI + NDSI1 + NDSI2 + SI1 + SI2 + SI3 + SI4 + SI5 + SAVI + VSSI + 
                NBR + NBG + NBNIR + NBSWIR1 + NBSWIR2 + NRSWIR1 + NRSWIR2 + NGSWIR1 + NGSWIR2 + NNIRSWIR1 + NNIRSWIR2, data = train_data)


models <- list()
models[['Linear']] <- linear_all
models[['Quadratic']] <- poly_all
models[['Log Linear']] <- log_all

msummary(models, stars = c('*' = .1, '**' = .05, '***' = .01), filename = 'table.rtf')
# While R2 is high, adjusted R2 is lower since too many predictors are added
# Result: R2(quadratic) = 0.891 / Adj R2 = 0.705 
# Most bands are also not significant, except NBSWIR2 in the linear and log fits and to some extent NRSWIR2 in the linear fit 

# Predict and test R2 


### Test poly_all 
# 1. Make predictions on test data
predictions <- predict(poly_all, newdata = test_data)

# 2. Calculate prediction metrics manually

# Create a data frame with actual and predicted values
actual <- test_data$EC  # Actual EC values in test data
results <- data.frame(
  Actual = actual,
  Predicted = predictions
)
# R2 = 1 - (Sum of Squared Residuals / Total Sum of Squares)
SSR <- sum((actual - predictions)^2)  # Sum of squared residuals
SST <- sum((actual - mean(actual))^2)  # Total sum of squares
R2 <- 1 - (SSR/SST)

# Calculate RMSE (Root Mean Square Error)
RMSE <- sqrt(mean((results$Actual - results$Predicted)^2))

# Calculate MAE (Mean Absolute Error)
MAE <- mean(abs(results$Actual - results$Predicted))

# Print results
print(paste("R-squared:", round(R2, 4)))
print(paste("RMSE:", round(RMSE, 4)))
print(paste("MAE:", round(MAE, 4)))

# R2 for test being in the negative suggests that the model is overfitted to the training data. 




#### Try with top 6 correlated bands and indices ####
# As per the correlation matrix, the best linear predictors are NIR, NBNIR, NDSI2, NRSWIR1, NDWI, SAVI
# Even though some of these are somewhat collinear with each other, try a model with just these 

linear_highcor <- lm(EC ~ NIR_R + NDWI + NDSI2 + SAVI + NBNIR + NRSWIR1, data = train_data)
poly_highcor <- lm(EC ~ poly(NIR_R,2) + poly(NDWI,2) + poly(NDSI2,2) + poly(SAVI,2) + poly(NBNIR,2) + poly(NRSWIR1,2), data = train_data)
log_highcor <- lm(log(EC) ~ NIR_R + NDWI + NDSI2 + SAVI + NBNIR + NRSWIR1, data = train_data)

models <- list()
models[['Linear']] <- linear_highcor
models[['Quadratic']] <- poly_highcor
models[['Log Linear']] <- log_highcor

msummary(models, stars = c('*' = .1, '**' = .05, '***' = .01), filename = 'table.rtf')
# R2 of the quadratic fit is the highest amongst these, although individually, none of the bands have significant coefficients 
# R2 (quadratic) = 0.480
# Still lower R2 than poly_all

### Test poly_highcor
predictions <- predict(poly_highcor, newdata = test_data)

# 2. Calculate prediction metrics manually

# Create a data frame with actual and predicted values
actual <- test_data$EC  # Actual EC values in test data
results <- data.frame(
  Actual = actual,
  Predicted = predictions
)
# R2 = 1 - (Sum of Squared Residuals / Total Sum of Squares)
SSR <- sum((actual - predictions)^2)  # Sum of squared residuals
SST <- sum((actual - mean(actual))^2)  # Total sum of squares
R2 <- 1 - (SSR/SST)

# Calculate RMSE (Root Mean Square Error)
RMSE <- sqrt(mean((results$Actual - results$Predicted)^2))

# Calculate MAE (Mean Absolute Error)
MAE <- mean(abs(results$Actual - results$Predicted))

# Print results
print(paste("R-squared:", round(R2, 4)))
print(paste("RMSE:", round(RMSE, 4)))
print(paste("MAE:", round(MAE, 4)))

# R2 for test data becomes better and =ve but still low


# Try these multi-band relationships after dropping any collinear bands
# (although the above models are not automatically dropping any predictors based on collinearity because no two are perfectly collinera)


#### Try Stepwise Regression which selects most relevant predictors based on AIC and BIC. 
# It systematically adds/removes predictors to find the best subset and fits a linear model.
# Fit a full linear model with all bands/indices
full_stepwise_model <- lm(EC ~ Blue_R + Red_R + Green_R + NIR_R + SWIR1_R + SWIR2_R + 
                            NDVI + NDWI + NDSI1 + NDSI2 + SI1 + SI2 + SI3 + SI4 + SI5 + SAVI + VSSI + 
                            NBR + NBG + NBNIR + NBSWIR1 + NBSWIR2 + NRSWIR1 + NRSWIR2 + NGSWIR1 + NGSWIR2 + NNIRSWIR1 + NNIRSWIR2, data = train_data)

# Perform stepwise selection
stepwise_model <- step(full_stepwise_model, direction = "both")

# Summary of the selected model
summary(stepwise_model)


# Higher R2 
# Results (EC_cont): 
# Residual standard error: 1364 on 67 degrees of freedom
# Multiple R-squared:  0.6348,	Adjusted R-squared:  0.5421 
# F-statistic:  6.85 on 17 and 67 DF,  p-value: 3.944e-09


### Test full_stepwise_model 
# 1. Make predictions on test data
predictions <- predict(full_stepwise_model, newdata = test_data)

# 2. Calculate prediction metrics manually
# Create a data frame with actual and predicted values
actual <- test_data$EC  # Actual EC values in test data
results <- data.frame(
  Actual = actual,
  Predicted = predictions
)
# R2 = 1 - (Sum of Squared Residuals / Total Sum of Squares)
SSR <- sum((actual - predictions)^2)  # Sum of squared residuals
SST <- sum((actual - mean(actual))^2)  # Total sum of squares
R2 <- 1 - (SSR/SST)

# Calculate RMSE (Root Mean Square Error)
RMSE <- sqrt(mean((results$Actual - results$Predicted)^2))

# Calculate MAE (Mean Absolute Error)
MAE <- mean(abs(results$Actual - results$Predicted))

# Print results
print(paste("R-squared:", round(R2, 4)))
print(paste("RMSE:", round(RMSE, 4)))
print(paste("MAE:", round(MAE, 4)))

# R2 for test data becomes better than poly_highcor (for both EC_all)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Step 6. Identify most appropriate bands and indices along with coefficients 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# See summarised in Notes.xls
# Best way forward would be to apply some machine learning methods to these input bands


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Step 7. Apply machine learning models - Random Forest, Bagging with RF, and Artificial Neural Networks
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#### RANDOM FOREST #####

# Fit a random forest model
rf_model <- randomForest(EC ~ Blue_R + Red_R + Green_R + NIR_R + SWIR1_R + SWIR2_R + 
                           NDVI + NDWI + NDSI1 + NDSI2 + SI1 + SI2 + SI3 + SI4 + SI5 + SAVI + VSSI + 
                           NBR + NBG + NBNIR + NBSWIR1 + NBSWIR2 + NRSWIR1 + NRSWIR2 + NGSWIR1 + NGSWIR2 + NNIRSWIR1 + NNIRSWIR2, data = train_data, ntree = 500)

# View variable importance
importance(rf_model)
varImpPlot(rf_model)

# 1. Calculate metrics for training data
predictions <- predict(rf_model, newdata = train_data)

# 2. Calculate prediction metrics manually
# Create a data frame with actual and predicted values
actual <- train_data$EC  # Actual EC values in test data
results <- data.frame(
  Actual = actual,
  Predicted = predictions
)
# R2 = 1 - (Sum of Squared Residuals / Total Sum of Squares)
SSR <- sum((actual - predictions)^2)  # Sum of squared residuals
SST <- sum((actual - mean(actual))^2)  # Total sum of squares
R2 <- 1 - (SSR/SST)

# Calculate RMSE (Root Mean Square Error)
RMSE <- sqrt(mean((results$Actual - results$Predicted)^2))

# Calculate MAE (Mean Absolute Error)
MAE <- mean(abs(results$Actual - results$Predicted))

# Print results
print(paste("R-squared (training):", round(R2, 4)))
print(paste("RMSE (training):", round(RMSE, 4)))
print(paste("MAE (training):", round(MAE, 4)))




# 1. Make predictions on test data
predictions <- predict(rf_model, newdata = test_data)

# 2. Calculate prediction metrics manually
# Create a data frame with actual and predicted values
actual <- test_data$EC  # Actual EC values in test data
results <- data.frame(
  Actual = actual,
  Predicted = predictions
)
# R2 = 1 - (Sum of Squared Residuals / Total Sum of Squares)
SSR <- sum((actual - predictions)^2)  # Sum of squared residuals
SST <- sum((actual - mean(actual))^2)  # Total sum of squares
R2 <- 1 - (SSR/SST)

# Calculate RMSE (Root Mean Square Error)
RMSE <- sqrt(mean((results$Actual - results$Predicted)^2))

# Calculate MAE (Mean Absolute Error)
MAE <- mean(abs(results$Actual - results$Predicted))

# Print results
print(paste("R-squared (testing):", round(R2, 4)))
print(paste("RMSE (testing):", round(RMSE, 4)))
print(paste("MAE (testing):", round(MAE, 4)))

# R2 for testing data maybe getting worse with EC_bin and EC_cat



#### BAGGING WITH RANDOM FOREST (Bootstrat Aggregating) ####

# Set mtry equal to the total number of predictors to implement bagging
bagging_model <- randomForest(EC ~ Blue_R + Red_R + Green_R + NIR_R + SWIR1_R + SWIR2_R + 
                                NDVI + NDWI + NDSI1 + NDSI2 + SI1 + SI2 + SI3 + SI4 + SI5 + SAVI + VSSI + 
                                NBR + NBG + NBNIR + NBSWIR1 + NBSWIR2 + NRSWIR1 + NRSWIR2 + NGSWIR1 + NGSWIR2 + NNIRSWIR1 + NNIRSWIR2, data = train_data, ntree = 500, mtry = 6)

# View variable importance
importance(bagging_model)
varImpPlot(bagging_model)

# 1. Calculate metrics for training data
predictions <- predict(bagging_model, newdata = train_data)

# 2. Calculate prediction metrics manually
# Create a data frame with actual and predicted values
actual <- train_data$EC  # Actual EC values in test data
results <- data.frame(
  Actual = actual,
  Predicted = predictions
)
# R2 = 1 - (Sum of Squared Residuals / Total Sum of Squares)
SSR <- sum((actual - predictions)^2)  # Sum of squared residuals
SST <- sum((actual - mean(actual))^2)  # Total sum of squares
R2 <- 1 - (SSR/SST)

# Calculate RMSE (Root Mean Square Error)
RMSE <- sqrt(mean((results$Actual - results$Predicted)^2))

# Calculate MAE (Mean Absolute Error)
MAE <- mean(abs(results$Actual - results$Predicted))

# Print results
print(paste("R-squared (training):", round(R2, 4)))
print(paste("RMSE (training):", round(RMSE, 4)))
print(paste("MAE (training):", round(MAE, 4)))




# 1. Make predictions on test data
predictions <- predict(bagging_model, newdata = test_data)

# 2. Calculate prediction metrics manually
# Create a data frame with actual and predicted values
actual <- test_data$EC  # Actual EC values in test data
results <- data.frame(
  Actual = actual,
  Predicted = predictions
)
# R2 = 1 - (Sum of Squared Residuals / Total Sum of Squares)
SSR <- sum((actual - predictions)^2)  # Sum of squared residuals
SST <- sum((actual - mean(actual))^2)  # Total sum of squares
R2 <- 1 - (SSR/SST)

# Calculate RMSE (Root Mean Square Error)
RMSE <- sqrt(mean((results$Actual - results$Predicted)^2))

# Calculate MAE (Mean Absolute Error)
MAE <- mean(abs(results$Actual - results$Predicted))

# Print results
print(paste("R-squared (testing):", round(R2, 4)))
print(paste("RMSE (testing):", round(RMSE, 4)))
print(paste("MAE (testing):", round(MAE, 4)))

# R2 for testing data maybe getting worse with EC_bin and EC_cat



#### ANN / Artificial Neural Networks #### 
# Scale data between 0 and 1 for ANN
train <- subset(train_data, select = c("EC", "Blue_R", "Red_R", "Green_R", "NIR_R", "SWIR1_R", "SWIR2_R", 
                                       "NDVI", "NDWI", "NDSI1", "NDSI2", "SI1", "SI2", "SI3", "SI4", "SI5", "SAVI", "VSSI", 
                                       "NBR", "NBG", "NBNIR", "NBSWIR1", "NBSWIR2", "NRSWIR1", "NRSWIR2", "NGSWIR1", "NGSWIR2", "NNIRSWIR1", "NNIRSWIR2"))

test <- subset(test_data, select = c("EC", "Blue_R", "Red_R", "Green_R", "NIR_R", "SWIR1_R", "SWIR2_R", 
                                     "NDVI", "NDWI", "NDSI1", "NDSI2", "SI1", "SI2", "SI3", "SI4", "SI5", "SAVI", "VSSI", 
                                     "NBR", "NBG", "NBNIR", "NBSWIR1", "NBSWIR2", "NRSWIR1", "NRSWIR2", "NGSWIR1", "NGSWIR2", "NNIRSWIR1", "NNIRSWIR2"))

train_scaled <- as.data.frame(scale(train))
test_scaled <- as.data.frame(scale(test))


# Option 1
# Fit an Artificial Neural Network
# size: number of neurons in the hidden layer, linout=TRUE for regression
ann_model <- nnet(EC ~ Blue_R + Red_R + Green_R + NIR_R + SWIR1_R + SWIR2_R + 
                    NDVI + NDWI + NDSI1 + NDSI2 + SI1 + SI2 + SI3 + SI4 + SI5 + SAVI + VSSI + 
                    NBR + NBG + NBNIR + NBSWIR1 + NBSWIR2 + NRSWIR1 + NRSWIR2 + NGSWIR1 + NGSWIR2 + NNIRSWIR1 + NNIRSWIR2, data = train_scaled, size = 4, linout = TRUE, maxit = 100)



# 1. Calculate metrics for training data
predictions <- predict(ann_model, newdata = train_scaled)

# 2. Calculate prediction metrics manually
# Create a data frame with actual and predicted values
actual <- train_scaled$EC  # Actual EC values in test data
results <- data.frame(
  Actual = actual,
  Predicted = predictions
)
# R2 = 1 - (Sum of Squared Residuals / Total Sum of Squares)
SSR <- sum((actual - predictions)^2)  # Sum of squared residuals
SST <- sum((actual - mean(actual))^2)  # Total sum of squares
R2 <- 1 - (SSR/SST)

# Calculate RMSE (Root Mean Square Error)
RMSE <- sqrt(mean((results$Actual - results$Predicted)^2))

# Calculate MAE (Mean Absolute Error)
MAE <- mean(abs(results$Actual - results$Predicted))

# Print results
print(paste("R-squared (training):", round(R2, 4)))
print(paste("RMSE (training):", round(RMSE, 4)))
print(paste("MAE (training):", round(MAE, 4)))




# 1. Make predictions on test data
predictions <- predict(ann_model, newdata = test_scaled)

# 2. Calculate prediction metrics manually
# Create a data frame with actual and predicted values
actual <- test_scaled$EC  # Actual EC values in test data
results <- data.frame(
  Actual = actual,
  Predicted = predictions
)
# R2 = 1 - (Sum of Squared Residuals / Total Sum of Squares)
SSR <- sum((actual - predictions)^2)  # Sum of squared residuals
SST <- sum((actual - mean(actual))^2)  # Total sum of squares
R2 <- 1 - (SSR/SST)

# Calculate RMSE (Root Mean Square Error)
RMSE <- sqrt(mean((results$Actual - results$Predicted)^2))

# Calculate MAE (Mean Absolute Error)
MAE <- mean(abs(results$Actual - results$Predicted))

# Print results
print(paste("R-squared (testing):", round(R2, 4)))
print(paste("RMSE (testing):", round(RMSE, 4)))
print(paste("MAE (testing):", round(MAE, 4)))

# Training might be getting better with EC_bin but testing is still -ve (overfitted). 




# #~~~~~~~~~~~~~ BELOW CODES ARE MOVED TO SEPARATE SCRIPTS ~~~~~~~~~~~
# #~~~~~~~~~~~~~ SEE Files Salinity_Index_Model_Testing.R Series for different iterations ~~~~~~~~~~~~~~
# 
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # Step 8. Repeat all models x 100 to test the average fit 
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# #~~~~~
# # IMPORTANT
# # Pick an appropriate y metric here to run the following analysis (EC_bin, EC_cat, EC_all)
# soil_data$EC <- soil_data$EC_all
# # soil_data$EC <- soil_data$EC_bin
# # soil_data$EC <- soil_data$EC_cat
# #~~~~~
# 
# 
# # Create soil data with only numeric fields 
# soil_data_numeric <- soil_data[, sapply(soil_data, is.numeric)]
# 
# # Initialize an empty dataframe to store results
# results_df <- data.frame(
#   Iteration = integer(),
#   Model_Type = character(),
#   R2_Train = numeric(),
#   R2_Test = numeric(),
#   RMSE_Train = numeric(),
#   RMSE_Test = numeric()
# )
# 
# # Define calculate_metrics function 
# calculate_metrics <- function(actual, predicted) {
#   # Calculate MSE (Mean Squared Error)
#   MSE <- mean((actual - predicted)^2)
#   
#   # Calculate RMSE (Root Mean Square Error)
#   RMSE <- sqrt(MSE)
#   
#   # Calculate MAE (Mean Absolute Error)
#   MAE <- mean(abs(actual - predicted))
#   
#   # Calculate R-squared
#   ss_tot <- sum((actual - mean(actual))^2)
#   ss_res <- sum((actual - predicted)^2)
#   R2 <- 1 - (ss_res/ss_tot)
#   
#   return(list(
#     MSE = MSE,
#     RMSE = RMSE,
#     MAE = MAE,
#     R2 = R2
#   ))
#   
# }
# 
# 
# # Define scale_numeric and reverse_numeric functions 
# 
# # Function to scale numeric data
# scale_numeric <- function(data) {
#   # Get numeric columns
#   numeric_cols <- sapply(data, is.numeric)
#   
#   # Scale numeric columns
#   scaled_data <- data
#   scaled_data[,numeric_cols] <- scale(data[,numeric_cols])
#   
#   return(scaled_data)
# }
# 
# # Function to reverse scaling
# reverse_scale <- function(scaled_values, original_values) {
#   # Get mean and sd of original values
#   orig_mean <- mean(original_values, na.rm = TRUE)
#   orig_sd <- sd(original_values, na.rm = TRUE)
#   
#   # Reverse scaling
#   reversed <- scaled_values * orig_sd + orig_mean
#   
#   return(reversed)
# }
# 
# 
# 
# 
# # Loop to run the models 100 times
# for (i in 1:100) {
#   
#   # Print iteration number
#   print(paste("Iteration:", i))
#   
#   # Shuffle and split data
#   shuffled_indices <- sample(seq_len(nrow(soil_data_numeric)))
#   train_size <- floor(0.8 * nrow(soil_data_numeric))
#   train_indices <- shuffled_indices[1:train_size]
#   test_indices <- shuffled_indices[(train_size + 1):nrow(soil_data_numeric)]
#   
#   train_data <- soil_data_numeric[train_indices, ]
#   test_data <- soil_data_numeric[test_indices, ]
#   
# 
#     
# ### 1. Logarithmic with all variables ### 
#     log_linear_model <- tryCatch({
#     lm(log(EC) ~ Blue_R + Red_R + Green_R + NIR_R + SWIR1_R + SWIR2_R + 
#          NDVI + NDWI + NDSI1 + NDSI2 + SI1 + SI2 + SI3 + SI4 + SI5 + SAVI + VSSI + 
#          NBR + NBG + NBNIR + NBSWIR1 + NBSWIR2 + NRSWIR1 + NRSWIR2 + NGSWIR1 + NGSWIR2 + NNIRSWIR1 + NNIRSWIR2, data = train_data)
#   }, error = function(e) {
#     message("Log-Linear model failed on iteration: ", i)
#     return(NULL)
#   })
#   
#   if (!is.null(log_linear_model)) {
#     train_predictions_log_linear <- exp(predict(log_linear_model, newdata = train_data))
#     test_predictions_log_linear <- exp(predict(log_linear_model, newdata = test_data))
#     
#     metrics_train_log_linear <- calculate_metrics(train_data$EC, train_predictions_log_linear)
#     metrics_test_log_linear <- calculate_metrics(test_data$EC, test_predictions_log_linear)
#     
#     results_df <- rbind(results_df, data.frame(
#       Iteration = i,
#       Model_Type = "Log-Linear",
#       R2_Train = metrics_train_log_linear$R2,
#       R2_Test = metrics_test_log_linear$R2,
#       RMSE_Train = metrics_train_log_linear$RMSE,
#       RMSE_Test = metrics_test_log_linear$RMSE
#     ))
#   }
#   
#   
# ### 2. OLS Model with all variables ###
#   
#   ols_model <- tryCatch({
#   lm(EC ~ Blue_R + Red_R + Green_R + NIR_R + SWIR1_R + SWIR2_R + 
#        NDVI + NDWI + NDSI1 + NDSI2 + SI1 + SI2 + SI3 + SI4 + SI5 + SAVI + VSSI + 
#        NBR + NBG + NBNIR + NBSWIR1 + NBSWIR2 + NRSWIR1 + NRSWIR2 + NGSWIR1 + NGSWIR2 + NNIRSWIR1 + NNIRSWIR2, data = train_data)
# }, error = function(e) {
#   message("OLS failed on iteration: ", i)
#   return(NULL)
# })
# 
# if (!is.null(ols_model)) {
#   train_predictions_ols <- predict(ols_model, newdata = train_data)
#   test_predictions_ols <- predict(ols_model, newdata = test_data)
#   
#   metrics_train_ols <- calculate_metrics(train_data$EC, train_predictions_ols)
#   metrics_test_ols <- calculate_metrics(test_data$EC, test_predictions_ols)
#   
#   results_df <- rbind(results_df, data.frame(
#     Iteration = i,
#     Model_Type = "OLS",
#     R2_Train = metrics_train_ols$R2,
#     R2_Test = metrics_test_ols$R2,
#     RMSE_Train = metrics_train_ols$RMSE,
#     RMSE_Test = metrics_test_ols$RMSE
#   ))
# }
# 
#   
#   
# ### 3. Polynomial with all variables ###
#   
#   poly_all <- tryCatch({
#     lm(EC ~ poly(Blue_R,2) + poly(Red_R,2) + poly(Green_R,2) + poly(NIR_R,2) + poly(SWIR1_R,2) + poly(SWIR2_R,2) + 
#                    poly(NDVI,2) + poly(NDWI,2) + poly(NDSI1,2) + poly(NDSI2,2) + poly(SI1,2) + poly(SI2,2) + poly(SI3,2) + poly(SI4,2) + poly(SI5,2) + poly(SAVI,2) + poly(VSSI,2) + 
#                    poly(NBR,2) + poly(NBG,2) + poly(NBNIR,2) + poly(NBSWIR1,2) + poly(NBSWIR2,2) + poly(NRSWIR1,2) + poly(NRSWIR2,2) + poly(NGSWIR1,2) + poly(NGSWIR2,2) + poly(NNIRSWIR1,2) + poly(NNIRSWIR2,2) , data = train_data)
#   }, error = function(e) {
#     message("Polynomial OLS failed on iteration: ", i)
#     return(NULL)
#   })
#   if (!is.null(poly_all)) {
#     train_predictions_ols <- predict(poly_all, newdata = train_data)
#     test_predictions_ols <- predict(poly_all, newdata = test_data)
#     
#     metrics_train_ols <- calculate_metrics(train_data$EC, train_predictions_ols)
#     metrics_test_ols <- calculate_metrics(test_data$EC, test_predictions_ols)
#     
#     # Append results for Random Forest
#     results_df <- rbind(results_df, data.frame(
#       Iteration = i,
#       Model_Type = "Quadratic_all",
#       R2_Train = metrics_train_ols$R2,
#       R2_Test = metrics_test_ols$R2,
#       RMSE_Train = metrics_train_ols$RMSE,
#       RMSE_Test = metrics_test_ols$RMSE
#     ))
#   }
#   
# 
#   ### 4. Polynomial with highly correlated variables ###  
#   poly_corr <- tryCatch({
#     lm(EC ~ poly(NIR_R,2) + poly(NDWI,2) + poly(NDSI2,2) + poly(SAVI,2) + poly(NBNIR,2) + poly(NRSWIR1,2), data = train_data)
#       }, error = function(e) {
#     message("Polynomial (correlated) failed on iteration: ", i)
#     return(NULL)
#   })
#   if (!is.null(poly_corr)) {
#     train_predictions_corr <- predict(poly_corr, newdata = train_data)
#     test_predictions_corr <- predict(poly_corr, newdata = test_data)
#     
#     metrics_train_corr <- calculate_metrics(train_data$EC, train_predictions_corr)
#     metrics_test_corr <- calculate_metrics(test_data$EC, test_predictions_corr)
#     
#     # Append results for Random Forest
#     results_df <- rbind(results_df, data.frame(
#       Iteration = i,
#       Model_Type = "Quadratic_correlated",
#       R2_Train = metrics_train_corr$R2,
#       R2_Test = metrics_test_corr$R2,
#       RMSE_Train = metrics_train_corr$RMSE,
#       RMSE_Test = metrics_test_corr$RMSE
#     ))
#   }
#   
#   
#   ### 5. Stepwise regression ###  
#   stepwise_model <- tryCatch({
#     stepAIC(lm(EC ~ Blue_R + Red_R + Green_R + NIR_R + SWIR1_R + SWIR2_R + 
#                  NDVI + NDWI + NDSI1 + NDSI2 + SI1 + SI2 + SI3 + SI4 + SI5 + SAVI + VSSI + 
#                  NBR + NBG + NBNIR + NBSWIR1 + NBSWIR2 + NRSWIR1 + NRSWIR2 + NGSWIR1 + NGSWIR2 + NNIRSWIR1 + NNIRSWIR2, data = train_data), direction = "both", trace = FALSE)
#   }, error = function(e) {
#     message("Stepwise regression failed on iteration: ", i)
#     return(NULL)
#   })
#   
#   if (!is.null(stepwise_model)) {
#     train_predictions_stepwise <- predict(stepwise_model, newdata = train_data)
#     test_predictions_stepwise <- predict(stepwise_model, newdata = test_data)
#     
#     metrics_train_stepwise <- calculate_metrics(train_data$EC, train_predictions_stepwise)
#     metrics_test_stepwise <- calculate_metrics(test_data$EC, test_predictions_stepwise)
#     
#     results_df <- rbind(results_df, data.frame(
#       Iteration = i,
#       Model_Type = "Stepwise",
#       R2_Train = metrics_train_stepwise$R2,
#       R2_Test = metrics_test_stepwise$R2,
#       RMSE_Train = metrics_train_stepwise$RMSE,
#       RMSE_Test = metrics_test_stepwise$RMSE
#     ))
#   }
#   
#   
#   
#   
#   ### 6. Random Forest ###
#   rf_model <- tryCatch({
#     randomForest(EC ~ Blue_R + Red_R + Green_R + NIR_R + SWIR1_R + SWIR2_R + 
#                    NDVI + NDWI + NDSI1 + NDSI2 + SI1 + SI2 + SI3 + SI4 + SI5 + SAVI + VSSI + 
#                    NBR + NBG + NBNIR + NBSWIR1 + NBSWIR2 + NRSWIR1 + NRSWIR2 + NGSWIR1 + NGSWIR2 + NNIRSWIR1 + NNIRSWIR2, data = train_data, ntree = 500)
#   }, error = function(e) {
#     message("Random Forest failed on iteration: ", i)
#     return(NULL)
#   })
#   
#   if (!is.null(rf_model)) {
#     train_predictions_rf <- predict(rf_model, newdata = train_data)
#     test_predictions_rf <- predict(rf_model, newdata = test_data)
#     
#     metrics_train_rf <- calculate_metrics(train_data$EC, train_predictions_rf)
#     metrics_test_rf <- calculate_metrics(test_data$EC, test_predictions_rf)
#     
#     # Append results for Random Forest
#     results_df <- rbind(results_df, data.frame(
#       Iteration = i,
#       Model_Type = "RandomForest",
#       R2_Train = metrics_train_rf$R2,
#       R2_Test = metrics_test_rf$R2,
#       RMSE_Train = metrics_train_rf$RMSE,
#       RMSE_Test = metrics_test_rf$RMSE
#     ))
#   }
#   
#   ### 7. Bagging (Random Forest with Bootstrap Aggregating) ###
#   bagging_model <- tryCatch({
#     randomForest(EC ~ Blue_R + Red_R + Green_R + NIR_R + SWIR1_R + SWIR2_R + 
#                    NDVI + NDWI + NDSI1 + NDSI2 + SI1 + SI2 + SI3 + SI4 + SI5 + SAVI + VSSI + 
#                    NBR + NBG + NBNIR + NBSWIR1 + NBSWIR2 + NRSWIR1 + NRSWIR2 + NGSWIR1 + NGSWIR2 + NNIRSWIR1 + NNIRSWIR2, data = train_data, ntree = 500, mtry = 5)
#   }, error = function(e) {
#     message("Random Forest with Bagging failed on iteration: ", i)
#     return(NULL)
#   })
#   
#   if (!is.null(bagging_model)) {
#     train_predictions_bagging <- predict(bagging_model, newdata = train_data)
#     test_predictions_bagging <- predict(bagging_model, newdata = test_data)
#     
#     metrics_train_bagging <- calculate_metrics(train_data$EC, train_predictions_bagging)
#     metrics_test_bagging <- calculate_metrics(test_data$EC, test_predictions_bagging)
#     
#     results_df <- rbind(results_df, data.frame(
#       Iteration = i,
#       Model_Type = "Bagging",
#       R2_Train = metrics_train_bagging$R2,
#       R2_Test = metrics_test_bagging$R2,
#       RMSE_Train = metrics_train_bagging$RMSE,
#       RMSE_Test = metrics_test_bagging$RMSE
#     ))
#   }
#   
#   ### 8. Artificial Neural Network (ANN) ###
#   
#   # Scale data for ANN
#   train_data_scaled <- scale_numeric(train_data)
#   test_data_scaled <- scale_numeric(test_data)
#   
#   ann_model <- tryCatch({
#     nnet(EC ~ Blue_R + Red_R + Green_R + NIR_R + SWIR1_R + SWIR2_R + 
#            NDVI + NDWI + NDSI1 + NDSI2 + SI1 + SI2 + SI3 + SI4 + SI5 + SAVI + VSSI + 
#            NBR + NBG + NBNIR + NBSWIR1 + NBSWIR2 + NRSWIR1 + NRSWIR2 + NGSWIR1 + NGSWIR2 + NNIRSWIR1 + NNIRSWIR2, data = train_data_scaled, size = 4, linout = TRUE, maxit = 100)
#   }, error = function(e) {
#     message("ANN failed on iteration: ", i)
#     return(NULL)
#   })
#   
#   if (!is.null(ann_model)) {
#     train_predictions_ann <- predict(ann_model, newdata = train_data_scaled)
#     test_predictions_ann <- predict(ann_model, newdata = test_data_scaled)
#     
#     train_predictions_ann_rescaled <- reverse_scale(train_predictions_ann, train_data$EC)
#     test_predictions_ann_rescaled <- reverse_scale(test_predictions_ann, test_data$EC)
#     
#     metrics_train_ann <- calculate_metrics(train_data$EC, train_predictions_ann_rescaled)
#     metrics_test_ann <- calculate_metrics(test_data$EC, test_predictions_ann_rescaled)
#     
#     results_df <- rbind(results_df, data.frame(
#       Iteration = i,
#       Model_Type = "ANN",
#       R2_Train = metrics_train_ann$R2,
#       R2_Test = metrics_test_ann$R2,
#       RMSE_Train = metrics_train_ann$RMSE,
#       RMSE_Test = metrics_test_ann$RMSE
#     ))
#   }
#   
#   # Print current size of results_df for diagnostics
#   print(paste("Results so far: ", nrow(results_df)))
# }
# 
# # Check final result size
# print(paste("Final number of rows in results_df: ", nrow(results_df)))
# 
# # Summarize the results
# summary(results_df)
# 
# 
# # Plot R² for all models
# ggplot(results_df, aes(x = Iteration, y = R2_Train, color = Model_Type)) +
#   geom_line() +
#   labs(title = "Training R² Across Iterations", y = "R²", x = "Iteration") +
#   theme_minimal()
# 
# ggplot(results_df, aes(x = Iteration, y = R2_Test, color = Model_Type)) +
#   geom_line() +
#   labs(title = "Testing R² Across Iterations", y = "R²", x = "Iteration") +
#   theme_minimal() +
#   coord_cartesian(ylim = c(quantile(results_df$R2_Test, 0.3), quantile(results_df$R2_Test, 0.7))) 
# 
# 
# write.csv(results_df, "100_iteration_results_ECall.csv")
# 
# # From these results, Random Forest with Bagging is still yielding the best results, even if the test results are not great! 
# 
# 
# 
# 
# 
# #### Try EC as a binary - High Salinity / Low Salinity at a particular threshold (=1900 when it gets too high for rice/agriculture?)
# #~~~~~
# # Pick an appropriate y metric here to run the following analysis (EC_bin, EC_cat, EC_all)
# soil_data_numeric <- soil_data[, sapply(soil_data, is.numeric)]
# soil_data_numeric$EC <- soil_data_numeric$EC_bin
# # Convert EC to a binary factor
# soil_data_numeric$EC <- as.factor(soil_data_numeric$EC)
# 
# table(soil_data_numeric$EC_Bin)
# 
# 
# #~~~~~
# # Need to apply a few changes to the previous script since this will require logistical analysis instead of linear 
# # Changes:
# # 1.As in Sarkar et al. Change from regression metrics (R², RMSE) to classification metrics (Accuracy, Sensitivity, Specificity, AUC)
# #   Accuracy: Overall correct prediction rate
# #   Sensitivity: True positive rate (correctly identifying 1's)
# #   Specificity: True negative rate (correctly identifying 0's)
# #   AUC: Area under the ROC curve (model's ability to distinguish between classes)
# # 2. Replace the continuous models (OLS, log-linear, polynomial, and ANN) with classification models, such as logistic regression, decision trees, and Random Forest classifiers.
# # 3. i.e. Modify RandomForest to do classification instead of regression
# # 4. Add proper handling of factor variables
# # 5. Add probability predictions where appropriate
# 
# 
# 
# #### Stratified Sampling  #####
# 
# #### With Stratified sampling for training and testing data along with 10-fold###
# # ! Almost all results are too accurate. Possibly overfitting. 
# 
# # Initialize an empty dataframe to store classification results
# results_df <- data.frame(
#   Iteration = integer(),
#   Model_Type = character(),
#   Accuracy_Train = numeric(),
#   Accuracy_Test = numeric(),
#   Precision = numeric(),
#   Recall = numeric(),
#   F1_Score = numeric()
# )
# 
# # Define functions
# calculate_classification_metrics <- function(actual, predicted) {
#   confusion <- table(factor(actual, levels = c(0, 1)), factor(predicted, levels = c(0, 1)))
#   
#   TN <- ifelse(!is.na(confusion[1, 1]), confusion[1, 1], 0) # True Negative
#   FP <- ifelse(!is.na(confusion[1, 2]), confusion[1, 2], 0) # False Positive
#   FN <- ifelse(!is.na(confusion[2, 1]), confusion[2, 1], 0) # False Negative
#   TP <- ifelse(!is.na(confusion[2, 2]), confusion[2, 2], 0) # True Positive
#   
#   accuracy <- (TP + TN) / (TN + FP + FN + TP)
#   precision <- ifelse((TP + FP) > 0, TP / (TP + FP), NA)
#   recall <- ifelse((TP + FN) > 0, TP / (TP + FN), NA)
#   f1_score <- ifelse(!is.na(precision) && !is.na(recall) && (precision + recall) > 0,
#                      2 * (precision * recall) / (precision + recall), NA)
#   
#   return(list(
#     Accuracy = accuracy,
#     Precision = precision,
#     Recall = recall,
#     F1_Score = f1_score
#   ))
# }
# 
# scale_numeric <- function(data) {
#   # Get numeric columns
#   numeric_cols <- sapply(data, is.numeric)
#   
#   # Scale numeric columns
#   scaled_data <- data
#   scaled_data[, numeric_cols] <- scale(data[, numeric_cols])
#   
#   return(scaled_data)
# }
# 
# # Loop to run the models 100 times
# for (i in 1:100) {
#   print(paste("Iteration:", i))
#   
#   # Stratified sampling for train-test split
#   set.seed(i)
#   train_indices <- createDataPartition(soil_data_numeric$EC, p = 0.8, list = FALSE)
#   train_data <- soil_data_numeric[train_indices, ]
#   test_data <- soil_data_numeric[-train_indices, ]
#   
#   # Debug: Check class balance
#   print(paste("Train class distribution (Iteration:", i, ")"))
#   print(table(train_data$EC))
#   print(paste("Test class distribution (Iteration:", i, ")"))
#   print(table(test_data$EC))
#   
#   # Check and remove near-zero variance predictors
#   nzv <- nearZeroVar(train_data, saveMetrics = TRUE)
#   if (any(nzv$nzv)) {
#     print("Near-zero variance predictors found:")
#     print(nzv[nzv$nzv, ])
#     train_data <- train_data[, !nzv$nzv]
#     test_data <- test_data[, !nzv$nzv]
#   }
#   
#   # Define training control for cross-validation
#   train_control <- trainControl(method = "cv", number = 10)
#   
#   ### 1. Logistic Regression Model ###
#   logistic_model <- tryCatch({
#     train(
#       #EC ~ Blue_R + Red_R + Green_R + NIR_R + SWIR1_R + SWIR2_R + 
#       #      NBNIR + NDSI2 + NRSWIR1 + NDWI + SAVI,
#           EC ~ NDWI + Green_R + SWIR2_R + NBR + NBNIR + NBSWIR2 + 
#             NRSWIR2 + NGSWIR1 + NDSI1 + VSSI,
#           data = train_data, method = "glm", family = binomial(link='logit'), trControl = train_control)
#   }, error = function(e) {
#     message("Logistic model failed on iteration: ", i)
#     return(NULL)
#   })
#   
#   if (!is.null(logistic_model)) {
#     train_pred_class <- predict(logistic_model, newdata = train_data)
#     test_pred_class <- predict(logistic_model, newdata = test_data)
#     
#     metrics_train_logistic <- calculate_classification_metrics(train_data$EC, train_pred_class)
#     metrics_test_logistic <- calculate_classification_metrics(test_data$EC, test_pred_class)
#     
#     results_df <- rbind(results_df, data.frame(
#       Iteration = i,
#       Model_Type = "Logistic",
#       Accuracy_Train = metrics_train_logistic$Accuracy,
#       Accuracy_Test = metrics_test_logistic$Accuracy,
#       Precision = metrics_test_logistic$Precision,
#       Recall = metrics_test_logistic$Recall,
#       F1_Score = metrics_test_logistic$F1_Score
#     ))
#   }
#   
#   ### 2. Random Forest Model ###
#   rf_model <- tryCatch({
#     randomForest(
#       #EC ~ Blue_R + Red_R + Green_R + NIR_R + SWIR1_R + SWIR2_R + 
#       #      NBNIR + NDSI2 + NRSWIR1 + NDWI + SAVI,
#       EC ~ NDWI + Green_R + SWIR2_R + NBR + NBNIR + NBSWIR2 + 
#         NRSWIR2 + NGSWIR1 + NDSI1 + VSSI,
#       data = train_data, ntree = 100,
#       mtry = max(5, floor(sqrt(ncol(train_data) - 1))),
#       nodesize = 1, maxnodes = 15
#     )
#   }, error = function(e) {
#     cat("Random Forest failed on iteration:", i, "\n")
#     cat("Random Forest Error Message:", conditionMessage(e), "\n")
#     return(NULL)
#   })
#   
#   if (!is.null(rf_model)) {
#     train_pred_class <- predict(rf_model, newdata = train_data)
#     test_pred_class <- predict(rf_model, newdata = test_data)
#     
#     metrics_train_rf <- calculate_classification_metrics(train_data$EC, train_pred_class)
#     metrics_test_rf <- calculate_classification_metrics(test_data$EC, test_pred_class)
#     
#     results_df <- rbind(results_df, data.frame(
#       Iteration = i,
#       Model_Type = "Random Forest",
#       Accuracy_Train = metrics_train_rf$Accuracy,
#       Accuracy_Test = metrics_test_rf$Accuracy,
#       Precision = metrics_test_rf$Precision,
#       Recall = metrics_test_rf$Recall,
#       F1_Score = metrics_test_rf$F1_Score
#     ))
#   }
#   
#   ### 3. Artificial Neural Network (ANN) ###
#   train_data_scaled <- scale_numeric(train_data)
#   test_data_scaled <- scale_numeric(test_data)
#   
#   ann_model <- tryCatch({
#     nnet(
#       #EC ~ Blue_R + Red_R + Green_R + NIR_R + SWIR1_R + SWIR2_R + 
#       #      NBNIR + NDSI2 + NRSWIR1 + NDWI + SAVI,
#       EC ~ NDWI + Green_R + SWIR2_R + NBR + NBNIR + NBSWIR2 + 
#         NRSWIR2 + NGSWIR1 + NDSI1 + VSSI,
#          data = train_data_scaled, size = 4, linout = FALSE, maxit = 100)
#   }, error = function(e) {
#     message("ANN failed on iteration: ", i)
#     return(NULL)
#   })
#   
#   if (!is.null(ann_model)) {
#     train_pred_class <- predict(ann_model, newdata = train_data_scaled, type = "class")
#     test_pred_class <- predict(ann_model, newdata = test_data_scaled, type = "class")
#     
#     metrics_train_ann <- calculate_classification_metrics(train_data$EC, train_pred_class)
#     metrics_test_ann <- calculate_classification_metrics(test_data$EC, test_pred_class)
#     
#     results_df <- rbind(results_df, data.frame(
#       Iteration = i,
#       Model_Type = "ANN",
#       Accuracy_Train = metrics_train_ann$Accuracy,
#       Accuracy_Test = metrics_test_ann$Accuracy,
#       Precision = metrics_test_ann$Precision,
#       Recall = metrics_test_ann$Recall,
#       F1_Score = metrics_test_ann$F1_Score
#     ))
#   }
#   
#   print(paste("Results so far:", nrow(results_df)))
# }
# 
# # Print summary of results
# print("Final Summary")
# print(summary(results_df))
# 
# # Plot Accuracy for all models
# ggplot(results_df, aes(x = Iteration, y = Accuracy_Train, color = Model_Type)) +
#   geom_line() +
#   labs(title = "Training Accuracy Across Iterations", y = "Accuracy", x = "Iteration") +
#   theme_minimal()
# 
# ggplot(results_df, aes(x = Iteration, y = Accuracy_Test, color = Model_Type)) +
#   geom_line() +
#   labs(title = "Testing Accuracy Across Iterations", y = "Accuracy", x = "Iteration") +
#   theme_minimal()
# 
# # Save results
# write.csv(results_df, "outputs/100_iteration_results_EC_binary_stratified_lasso.csv")
# 
# 
# 
# 
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# #### Random Sampling  #####
# # # Initialize an empty dataframe to store results
# results_df <- data.frame(
#   Iteration = integer(),
#   Model_Type = character(),
#   Accuracy_Train = numeric(),
#   Accuracy_Test = numeric()
# )
# 
# for (i in 1:100) {
#   print(paste("Iteration:", i))
#   
#   # Shuffle and split data
#   set.seed(i)
#   shuffled_indices <- sample(seq_len(nrow(soil_data)))
#   train_size <- floor(0.8 * nrow(soil_data))
#   train_indices <- shuffled_indices[1:train_size]
#   test_indices <- shuffled_indices[(train_size + 1):nrow(soil_data)]
#   
#   train_data <- soil_data_numeric[train_indices, ]
#   test_data <- soil_data_numeric[test_indices, ]
#   
#   # Check and remove near-zero variance predictors
#   nzv <- nearZeroVar(train_data, saveMetrics = TRUE)
#   if (any(nzv$nzv)) {
#     print("Near-zero variance predictors found:")
#     print(nzv[nzv$nzv, ])
#     train_data <- train_data[, !nzv$nzv]
#     test_data <- test_data[, !nzv$nzv]
#   }
#   
#   # Define training control for cross-validation
#   train_control <- trainControl(method = "cv", number = 10)
#   
#   ### 1. Logistic Regression Model ###
#   logistic_model <- tryCatch({
#     train(
#       #EC ~ Blue_R + Red_R + Green_R + NIR_R + SWIR1_R + SWIR2_R + 
#       #      NBNIR + NDSI2 + NRSWIR1 + NDWI + SAVI,
#       EC ~ NDWI + NDSI + Green_R + SWIR2_R + NBR + NBNIR + NBSWIR2 + 
#         NRSWIR2 + NGSWIR1 + NDSI1 + VSSI,
#           data = train_data, method = "glm", family = binomial(link='logit'), trControl = train_control)
#   }, error = function(e) {
#     message("Logistic model failed on iteration: ", i)
#     return(NULL)
#   })
#   
#   if (!is.null(logistic_model)) {
#     train_pred_class <- predict(logistic_model, newdata = train_data)
#     test_pred_class <- predict(logistic_model, newdata = test_data)
#     
#     metrics_train_logistic <- calculate_classification_metrics(train_data$EC, train_pred_class)
#     metrics_test_logistic <- calculate_classification_metrics(test_data$EC, test_pred_class)
#     
#     results_df <- rbind(results_df, data.frame(
#       Iteration = i,
#       Model_Type = "Logistic",
#       Accuracy_Train = metrics_train_logistic$Accuracy,
#       Accuracy_Test = metrics_test_logistic$Accuracy,
#       Precision = metrics_test_logistic$Precision,
#       Recall = metrics_test_logistic$Recall,
#       F1_Score = metrics_test_logistic$F1_Score
#     ))
#   }
#   
#   ### 2. Random Forest Model ###
#   rf_model <- tryCatch({
#     randomForest(
#       #EC ~ Blue_R + Red_R + Green_R + NIR_R + SWIR1_R + SWIR2_R + 
#       #      NBNIR + NDSI2 + NRSWIR1 + NDWI + SAVI,
#       EC ~ NDWI + NDSI + Green_R + SWIR2_R + NBR + NBNIR + NBSWIR2 + 
#         NRSWIR2 + NGSWIR1 + NDSI1 + VSSI,
#       data = train_data, ntree = 100,
#       mtry = max(5, floor(sqrt(ncol(train_data) - 1))),
#       nodesize = 1, maxnodes = 15
#     )
#   }, error = function(e) {
#     cat("Random Forest failed on iteration:", i, "\n")
#     cat("Random Forest Error Message:", conditionMessage(e), "\n")
#     return(NULL)
#   })
#   
#   if (!is.null(rf_model)) {
#     train_pred_class <- predict(rf_model, newdata = train_data)
#     test_pred_class <- predict(rf_model, newdata = test_data)
#     
#     metrics_train_rf <- calculate_classification_metrics(train_data$EC, train_pred_class)
#     metrics_test_rf <- calculate_classification_metrics(test_data$EC, test_pred_class)
#     
#     results_df <- rbind(results_df, data.frame(
#       Iteration = i,
#       Model_Type = "Random Forest",
#       Accuracy_Train = metrics_train_rf$Accuracy,
#       Accuracy_Test = metrics_test_rf$Accuracy,
#       Precision = metrics_test_rf$Precision,
#       Recall = metrics_test_rf$Recall,
#       F1_Score = metrics_test_rf$F1_Score
#     ))
#   }
#   
#   ### 3. Artificial Neural Network (ANN) ###
#   train_data_scaled <- scale_numeric(train_data)
#   test_data_scaled <- scale_numeric(test_data)
#   
#   ann_model <- tryCatch({
#     nnet(
#       #EC ~ Blue_R + Red_R + Green_R + NIR_R + SWIR1_R + SWIR2_R + 
#       #      NBNIR + NDSI2 + NRSWIR1 + NDWI + SAVI,
#       EC ~ NDWI + NDSI + Green_R + SWIR2_R + NBR + NBNIR + NBSWIR2 + 
#         NRSWIR2 + NGSWIR1 + NDSI1 + VSSI,
#       data = train_data_scaled, size = 4, linout = FALSE, maxit = 100)
#   }, error = function(e) {
#     message("ANN failed on iteration: ", i)
#     return(NULL)
#   })
#   
#   if (!is.null(ann_model)) {
#     train_pred_class <- predict(ann_model, newdata = train_data_scaled, type = "class")
#     test_pred_class <- predict(ann_model, newdata = test_data_scaled, type = "class")
#     
#     metrics_train_ann <- calculate_classification_metrics(train_data$EC, train_pred_class)
#     metrics_test_ann <- calculate_classification_metrics(test_data$EC, test_pred_class)
#     
#     results_df <- rbind(results_df, data.frame(
#       Iteration = i,
#       Model_Type = "ANN",
#       Accuracy_Train = metrics_train_ann$Accuracy,
#       Accuracy_Test = metrics_test_ann$Accuracy,
#       Precision = metrics_test_ann$Precision,
#       Recall = metrics_test_ann$Recall,
#       F1_Score = metrics_test_ann$F1_Score
#     ))
#   }
#   
#   print(paste("Results so far:", nrow(results_df)))
# }
# 
# summary(results_df)
# 
# # Plot accuracy for each model
# ggplot(results_df, aes(x = Iteration, y = Accuracy_Train, color = Model_Type)) +
#   geom_line() +
#   labs(title = "Training Accuracy Across Iterations", y = "Accuracy", x = "Iteration") +
#   theme_minimal()
# 
# ggplot(results_df, aes(x = Iteration, y = Accuracy_Test, color = Model_Type)) +
#   geom_line() +
#   labs(title = "Testing Accuracy Across Iterations", y = "Accuracy", x = "Iteration") +
#   theme_minimal()
# 
# # Save results
# write.csv(results_df, "100_iteration_results_EC_binary_random_rf.csv")
# 
# 
# 
# 
#  
# 
# # ================= feature selection with LASSO =====================
# drop.cols <- c('X.1', 'pH', 'Soil_Moisture_Lab', 'Dist_Sea',
#                'RPD_pH', 'RPD_EC', 'RPD_SM',
#                'Aqua_in_GRID', 'X',
#                'EC', 'EC_all', 'EC_bin',
#                'EC_cat',
#                'Blue', 'Green', 'Red', 'NIR','SWIR1', 'SWIR2')
# x <- soil_data_numeric %>% dplyr::select(-one_of(drop.cols))
# # scale x
# for(i in 1:ncol(x)){
#   # print(i)
#   x[, i] <- rescale(x[, i])
# }
# x <- data.matrix(x)
# 
# y <- soil_data_numeric$EC
# table(y)
# 
# cvmodel <- cv.glmnet(x, y, alpha=1, family='binomial') # did not converge
# plot(cvmodel)
# best_lambda <- cvmodel$lambda.min
# best_lambda
# 
# # we can also tweak lambda to see
# bestlasso <- glmnet(x, y, alpha=1, lambda=best_lambda, family='binomial')
# coef(bestlasso)
# # selected predictors
# # if Dist_Sea is kept:
# # Dist_Sea, NDWI, NDSI, Green_R, SWIR2_R, NBNIR, NBSWIR2, NRSWIR1, NGSWIR1, VSSI
# # if Dist_Sea is removed:
# # NDWI, NDSI, Green_R, SWIR2_R, NBR, NBNIR, NBSWIR2, NRSWIR2, NGSWIR1, NDSI1, VSSI
# 
# 
# 
# 
# 
# 
# # Next steps: 
# # 1. Add quadratic and logarithmic values of a select bands in the model too, to improve it further. 
# # 2. Add principal components of the bands instead of the bands directly. 
# # 3. Add land_cover as a control/FE and then conduct the analysis. The same land cover categories can be used in the GEE LULC to apply in the predictive model. 
# # 4. Test with fewer parameters 
# # 5. Test data small 90-10
# # 6. Leave one out cross validation methods 