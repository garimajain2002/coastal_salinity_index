#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~ Salinity Map using Stacked Ensemble Model ~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

library(raster)
library(terra)
library(dplyr)
library(ggplot2)


# ================ 1. Read data ===============
getwd()

source("code/Funcs.R")
source("code/14_Salinity_Index_StackingEnsemble_Multimodel.R")

# Steps: 
# 0. Read relevant landsat multiband and aquaculture classification images
# 1. Prepare landsat multiband image 
# 2. Apply salinity model 
# 3. Map Predictions Back to Raster
# 4. Mask Aquaculture ponds for final image 
# 5. Plot and save final salinity maps


# 0. Read relevant landsat multiband image and aquaculture classification image for masking
# 2024
landsat_image <- stack("data/tifs/2024_JSP_Landsat_Composite_AllBands.tif")
aqua_image <- stack("data/tifs/2024_JSP_Aquaculture.tif") #stack() reads the bands into a multi-layer object


# # For historical maps replace images with respective years and run the same code below
# # 2014
# landsat_image <- stack("data/tifs/2014_JSP_Composite_AllBands.tif")
# aqua_image <- stack("data/tifs/2014_JSP_Aquaculture.tif")
# 
# # 2017
# landsat_image <- stack("data/tifs/2017_JSP_Composite_AllBands.tif")
# aqua_image <- stack("data/tifs/2017_JSP_Aquaculture.tif")
# 
# # 2021
# landsat_image <- stack("data/tifs/2021_JSP_Composite_AllBands.tif")
# aqua_image <- stack("data/tifs/2021_JSP_Aquaculture.tif")
# 


# 1. Prepare landsat multiband image
# Create a raster where each pixel has a unique ID to use when merging predicted values 
landsat_df <- as.data.frame(landsat_image, na.rm = FALSE) # Keep all pixels, including NAs
colnames(landsat_df) <- c("BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2", "NDVI", "NDWI", "NDSI")
landsat_df$ID <- seq_len(nrow(landsat_df))

# Drop masked cells or where value is 0 (surface water cells)
landsat_df <- subset(landsat_df, BLUE != 0)

# Ensure no NA values in landsat_df
landsat_df <- landsat_df[complete.cases(landsat_df), ]


# Convert DN values and handle negative values
landsat_df$Blue_R <- pmax((landsat_df$BLUE * 0.0000275) - 0.2, 0)
landsat_df$Red_R <- pmax((landsat_df$RED * 0.0000275) - 0.2, 0)
landsat_df$Green_R <- pmax((landsat_df$GREEN * 0.0000275) - 0.2, 0)
landsat_df$NIR_R <- pmax((landsat_df$NIR * 0.0000275) - 0.2, 0)
landsat_df$SWIR1_R <- pmax((landsat_df$SWIR1 * 0.0000275) - 0.2, 0)
landsat_df$SWIR2_R <- pmax((landsat_df$SWIR2 * 0.0000275) - 0.2, 0)

# Calculate additional indices 
# Blue and red 
landsat_df$NBR <- (landsat_df$Blue_R-landsat_df$Red_R)/(landsat_df$Blue_R+landsat_df$Red_R)

# Blue and green 
landsat_df$NBG <- (landsat_df$Blue_R-landsat_df$Green_R)/(landsat_df$Blue_R+landsat_df$Green_R)

# Blue and NIR
landsat_df$NBNIR <- (landsat_df$Blue_R-landsat_df$NIR_R)/(landsat_df$Blue_R+landsat_df$NIR_R)

# Blue and SWIR1
landsat_df$NBSWIR1 <- (landsat_df$Blue_R-landsat_df$SWIR1_R)/(landsat_df$Blue_R+landsat_df$SWIR1_R)

# Blue and SWIR2 
landsat_df$NBSWIR2 <- (landsat_df$Blue_R-landsat_df$SWIR2_R)/(landsat_df$Blue_R+landsat_df$SWIR2_R)

# Red and Green (also NDVI) 
landsat_df$NDVI <- (landsat_df$Red_R-landsat_df$Green_R)/(landsat_df$Red_R+landsat_df$Green_R)

# Red and NIR (NDSI2 or Normalised Difference Salinity Index 2 as per as per Khan et al 2001 in Nguyen et al 2020)
landsat_df$NDSI2 <- (landsat_df$Red_R-landsat_df$NIR_R)/(landsat_df$Red_R+landsat_df$NIR_R)

# Red and SWIR1
landsat_df$NRSWIR1 <- (landsat_df$Red_R-landsat_df$SWIR1_R)/(landsat_df$Red_R+landsat_df$SWIR1_R)

# Red and SWIR2
landsat_df$NRSWIR2 <- (landsat_df$Red_R-landsat_df$SWIR2_R)/(landsat_df$Red_R+landsat_df$SWIR2_R)

# Green and NIR (also NDWI) 
landsat_df$NDWI <- (landsat_df$Green_R-landsat_df$NIR_R)/(landsat_df$Green_R+landsat_df$NIR_R)

# Green and SWIR1
landsat_df$NGSWIR1 <- (landsat_df$Green_R-landsat_df$SWIR1_R)/(landsat_df$Green_R+landsat_df$SWIR1_R)

# Green and SWIR2
landsat_df$NGSWIR2 <- (landsat_df$Green_R-landsat_df$SWIR2_R)/(landsat_df$Green_R+landsat_df$SWIR2_R)

# NIR and SWIR1
landsat_df$NNIRSWIR1 <- (landsat_df$NIR_R-landsat_df$SWIR1_R)/(landsat_df$NIR_R+landsat_df$SWIR1_R)

# NIR and SWIR2 
landsat_df$NNIRSWIR2 <- (landsat_df$NIR_R-landsat_df$SWIR2_R)/(landsat_df$NIR_R+landsat_df$SWIR2_R)

# SWIR1 and SWIR2 (also NDSI as per the Index Database: https://www.indexdatabase.de/db/is.php?sensor_id=168 )
landsat_df$NDSI1 <- (landsat_df$SWIR1_R-landsat_df$SWIR2_R)/(landsat_df$SWIR1_R+landsat_df$SWIR2_R)

# Salinity Index 1 = sqrt(green^2+red^2)
landsat_df$SI1 <- sqrt((landsat_df$Green_R)^2 + (landsat_df$Red_R)^2) 

# Salinity Index 2 = sqrt(green x red)
landsat_df$SI2 <- sqrt(landsat_df$Green_R * landsat_df$Red_R)

# Salinity Index 3 = sqrt(blue x red) 
landsat_df$SI3 <- sqrt(landsat_df$Blue_R * landsat_df$Red_R)

# salinity index 4 = red x NIR / green 
landsat_df$SI4 <- (landsat_df$Red_R * landsat_df$NIR_R / landsat_df$Green_R)  

# salinity index 5 = blue/red 
landsat_df$SI5 <- (landsat_df$Blue_R / landsat_df$Red_R)   

# Soil Adjusted Vegetation Index (SAVI) = ((1.5)x NIR) - (red/0.5) + NIR + Red 
landsat_df$SAVI <- (1.5 * landsat_df$NIR_R) - (0.5 * landsat_df$Red_R) + landsat_df$NIR_R + landsat_df$Red_R

# Vegetation Soil Salinity Index (VSSI) = (2 x green) - 5 x (red + NIR) 
landsat_df$VSSI <- (2 * landsat_df$Green_R) - 5 * (landsat_df$Red_R + landsat_df$NIR_R)

landsat_df <- landsat_df[, c("ID", "Blue_R", "Green_R", "Red_R", "NIR_R", "SWIR1_R", "SWIR2_R",
                                     "NDVI", "NDWI", "NDSI1", "NDSI2", "SI1", "SI2", "SI3", "SI4", "SI5", "SAVI", "VSSI",
                                     "NBR", "NBG", "NBNIR", "NBSWIR1", "NBSWIR2", "NRSWIR1", "NRSWIR2", "NGSWIR1", "NGSWIR2",
                                     "NNIRSWIR1", "NNIRSWIR2")]


# Remove NA values if any after processing and before applying the ensemble model
length(landsat_df$ID)
landsat_df <- landsat_df[complete.cases(landsat_df), ]
length(landsat_df$ID)

# Check df for any Infinite, or NaN values
colSums(is.na(landsat_df))   # Check for NA values
sum(is.infinite(as.matrix(landsat_df)))  # Check for Inf values
sum(is.nan(as.matrix(landsat_df)))  # Check for NaN values

# Find columns containing Inf values
inf_cols <- sapply(landsat_df, function(x) any(is.infinite(x)))
names(inf_cols[inf_cols == TRUE])

# If any Red band is 0, SI5 will become Infinite. Omit those rows 
# Convert Inf values to NA first
landsat_df <- as.data.frame(lapply(landsat_df, function(x) {
  if (is.numeric(x)) x[is.infinite(x)] <- NA  # Only apply to numeric columns
  return(x)
}))
# Omit rows with NA values
landsat_df <- na.omit(landsat_df)
# Check if Infinite values are addressed
sum(is.infinite(as.matrix(landsat_df)))  # Should return 0



# 2. Apply salinity model
predictions <- predict(MultiStratEnsemble, newdata = landsat_df)
landsat_df$predicted_EC <- ifelse(predictions[, "X1"] > best_threshold$threshold, 1, 0)


# 3. Map predictions back to raster
predicted_raster <- raster(landsat_image[[1]])
values(predicted_raster)[landsat_df$ID] <- landsat_df$predicted_EC


# Convert raster to data frame
predicted_df <- as.data.frame(predicted_raster, xy = TRUE)
colnames(predicted_df) <- c("x", "y", "predicted_EC")

length(landsat_df$ID)
length(landsat_df$predicted_EC)
ncell(predicted_raster)


# Create ggplot
predicted_EC<- ggplot(predicted_df, aes(x = x, y = y, fill = factor(predicted_EC))) +
  geom_raster() +
  scale_fill_manual(values = c("0" = "black", "1" = "white")) +
  theme_minimal() +
  labs(title = "Predicted Electrical Conductivity", 
       fill = "Salinity Class")

plot(predicted_EC)

# Change year in filename based on original image selection year
writeRaster(predicted_raster, "outputs/predicted_EC_raster_2024.tif", format = "GTiff", overwrite = TRUE)
ggsave("outputs/predicted_EC_map_2024.png", plot = predicted_EC, width = 8, height = 6, dpi = 300)



# 4. Mask Aquaculture ponds for final image 
aqua_df <- as.data.frame(aqua_image, xy = TRUE)

#Note: Aqua == 2, Dry Aqua == 1, and others == 0 
unique(values(aqua_image))

# Check df for any Infinite, or NaN values
colSums(is.na(aqua_df))   # Check for NA values
sum(is.infinite(as.matrix(aqua_df)))  # Check for Inf values
sum(is.nan(as.matrix(aqua_df)))  # Check for NaN values

#plot aquaculture 
predicted_Aqua <- ggplot(aqua_df, aes(x = x, y = y, fill = factor(classification))) +
  geom_raster() +
  scale_fill_manual(values = c("0" = "black", "1" = "black", "2" = "#666666")) +
  theme_minimal() +
  labs(title = "Classified Aquaculture ponds", 
       fill = "Aquaculture ponds")


aqua_mask_raster <- aqua_image == 2  # Create a binary mask of aquaculture areas
# Replace predicted EC values with 2 where aquaculture mask is TRUE
predicted_raster[aqua_mask_raster] <- 2
values(predicted_raster) <- as.numeric(values(predicted_raster))  # Force numeric raster
unique(values(predicted_raster))

# # Change year in filename based on original image selection year
ggsave("outputs/predicted_Aqua_map_2024.png", plot = predicted_Aqua, width = 8, height = 6, dpi = 300)



# 5. Plot and save final combined salinity and aqua maps
#convert final predicted raster to df, then ggplot to check 
final_df <- as.data.frame(predicted_raster, xy = TRUE)
colnames(final_df) <- c("x", "y", "predicted_EC")


predicted_ECAqua <- ggplot(final_df, aes(x = x, y = y, fill = factor(predicted_EC))) +
  geom_raster() +
  scale_fill_manual(values = c("0" = "black", "1" = "orange", "2" = "blue")) + #get other colours from http://www.cookbook-r.com/Graphs/Colors_(ggplot2)/
  theme_minimal() +
  labs(title = "Predicted Electrical Conductivity in Aquaculture context", 
       fill = "Salinity Class")

plot(predicted_ECAqua)

# Change year in filename based on original image selection year
ggsave("outputs/predicted_ECAqua_map_2024.png", plot = predicted_ECAqua, width = 8, height = 6, dpi = 300)
writeRaster(predicted_raster, "outputs/predicted_ECAqua_raster_2024.tif", format = "GTiff", overwrite = TRUE)

