#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  Landsat 5 (final) Salinity Map using Stacked Ensemble Model 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

library(raster)
library(terra)
library(dplyr)
library(ggplot2)
library(sf)

# ================ 1. Read data ===============
getwd()

MultiStratEnsemble <- readRDS("ensemble_2000.rds")
best_threshold = 0.624

# Steps: 
# 0. Read relevant landsat multiband and aquaculture classification images - Landsat 8 and Landsat 5
# 1. Prepare landsat multiband image 
# 2. Apply salinity model 
# 3. Map Predictions Back to Raster
# 4. Mask Aquaculture ponds for final image 
# 5. Plot and save final salinity maps

# shape file for clipping the final output
boundary <- vect("data/JSP_shp/JSP.shp")  

# 0. Read relevant landsat multiband image and aquaculture classification image for masking
# # 2024 for reference
landsat8_image <- stack("data/tifs/geomedian_harmonised_New5classified/OD_4_geomedian_2024_Feb.tif")
landsat8_image <- landsat8_image[[1:6]] # only keep the bands and not the indices


# For historical maps replace images with respective years and run the same code below
# 2001 JSP
landsat_image <- stack("data/tifs/geomedian_harmonised_New5classified/OD_4_geomedian_2001_predicted_predicted.tif")
aqua_image <- stack("data/tifs/geomedian_harmonised_New5classified/OD_4_classified_2001.tif") #stack() reads the bands into a multi-layer object


names(landsat_image)
names(landsat8_image)

standard_band_names <- c("Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2")

names(landsat_image) <- standard_band_names
names(landsat8_image) <- standard_band_names

landsat_image <- landsat_image * 0.0000275 - 0.2


# 1. Prepare landsat multiband images
# Create a raster where each pixel has a unique ID to use when merging predicted values 
landsat8_df <- as.data.frame(landsat8_image, na.rm = FALSE) # Keep all pixels, including NAs
landsat_df <- as.data.frame(landsat_image, na.rm = FALSE) # Keep all pixels, including NAs


landsat8_df$ID <- seq_len(nrow(landsat8_df))
landsat_df$ID <- seq_len(nrow(landsat_df))

# Drop masked cells or where value is 0 (surface water cells)
landsat8_df <- subset(landsat8_df, Blue != 0)
landsat_df <- subset(landsat_df, Blue != 0)

# Ensure no NA values in landsat_df
landsat8_df <- landsat8_df[complete.cases(landsat8_df), ]
landsat_df <- landsat_df[complete.cases(landsat_df), ]

summary(landsat8_df)
summary(landsat_df)


# Make -ve values NA 
sapply(landsat8_df, function(x) sum(x < 0, na.rm = TRUE)) 
# Note: Landsat 5 does not have -ves. Corrected during harmonization 


# Replace negative values in those columns with NA
landsat8_df[standard_band_names] <- lapply(landsat8_df[standard_band_names], function(x) {
  x[x < 0] <- NA
  return(x)
})

summary(landsat8_df)


# Remove outlier band values in Landsat 5 image 
# ------






# Calculate additional indices 
# Blue and red 
landsat_df$NBR <- (landsat_df$Blue-landsat_df$Red)/(landsat_df$Blue+landsat_df$Red)

# Blue and green 
landsat_df$NBG <- (landsat_df$Blue-landsat_df$Green)/(landsat_df$Blue+landsat_df$Green)

# Blue and NIR
landsat_df$NBNIR <- (landsat_df$Blue-landsat_df$NIR)/(landsat_df$Blue+landsat_df$NIR)

# Blue and SWIR1
landsat_df$NBSWIR1 <- (landsat_df$Blue-landsat_df$SWIR1)/(landsat_df$Blue+landsat_df$SWIR1)

# Blue and SWIR2 
landsat_df$NBSWIR2 <- (landsat_df$Blue-landsat_df$SWIR2)/(landsat_df$Blue+landsat_df$SWIR2)

# Red and Green (also NDVI) 
landsat_df$NDVI <- (landsat_df$Red-landsat_df$Green)/(landsat_df$Red+landsat_df$Green)

# Red and NIR (NDSI2 or Normalised Difference Salinity Index 2 as per as per Khan et al 2001 in Nguyen et al 2020)
landsat_df$NDSI2 <- (landsat_df$Red-landsat_df$NIR)/(landsat_df$Red+landsat_df$NIR)

# Red and SWIR1
landsat_df$NRSWIR1 <- (landsat_df$Red-landsat_df$SWIR1)/(landsat_df$Red+landsat_df$SWIR1)

# Red and SWIR2
landsat_df$NRSWIR2 <- (landsat_df$Red-landsat_df$SWIR2)/(landsat_df$Red+landsat_df$SWIR2)

# Green and NIR (also NDWI) 
landsat_df$NDWI <- (landsat_df$Green-landsat_df$NIR)/(landsat_df$Green+landsat_df$NIR)

# Green and SWIR1
landsat_df$NGSWIR1 <- (landsat_df$Green-landsat_df$SWIR1)/(landsat_df$Green+landsat_df$SWIR1)

# Green and SWIR2
landsat_df$NGSWIR2 <- (landsat_df$Green-landsat_df$SWIR2)/(landsat_df$Green+landsat_df$SWIR2)

# NIR and SWIR1
landsat_df$NNIRSWIR1 <- (landsat_df$NIR-landsat_df$SWIR1)/(landsat_df$NIR+landsat_df$SWIR1)

# NIR and SWIR2 
landsat_df$NNIRSWIR2 <- (landsat_df$NIR-landsat_df$SWIR2)/(landsat_df$NIR+landsat_df$SWIR2)

# SWIR1 and SWIR2 (also NDSI as per the Index Database: https://www.indexdatabase.de/db/is.php?sensor_id=168 )
landsat_df$NDSI1 <- (landsat_df$SWIR1-landsat_df$SWIR2)/(landsat_df$SWIR1+landsat_df$SWIR2)

# Salinity Index 1 = sqrt(green^2+red^2)
landsat_df$SI1 <- sqrt((landsat_df$Green)^2 + (landsat_df$Red)^2) 

# Salinity Index 2 = sqrt(green x red)
landsat_df$SI2 <- sqrt(landsat_df$Green * landsat_df$Red)

# Salinity Index 3 = sqrt(blue x red) 
landsat_df$SI3 <- sqrt(landsat_df$Blue * landsat_df$Red)

# salinity index 4 = red x NIR / green 
landsat_df$SI4 <- (landsat_df$Red * landsat_df$NIR / landsat_df$Green)  

# salinity index 5 = blue/red 
landsat_df$SI5 <- (landsat_df$Blue / landsat_df$Red)   

# Soil Adjusted Vegetation Index (SAVI) = ((1.5)x NIR) - (red/0.5) + NIR + Red 
landsat_df$SAVI <- (1.5 * landsat_df$NIR) - (0.5 * landsat_df$Red) + landsat_df$NIR + landsat_df$Red

# Vegetation Soil Salinity Index (VSSI) = (2 x green) - 5 x (red + NIR) 
landsat_df$VSSI <- (2 * landsat_df$Green) - 5 * (landsat_df$Red + landsat_df$NIR)

landsat_df <- landsat_df[, c("ID", "Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2",
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
landsat_df$predicted_EC <- ifelse(predictions[, "X1"] > best_threshold, 1, 0)


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
  labs(title = "Predicted Electrical Conductivity - 2001", 
       fill = "Salinity Class")

plot(predicted_EC)

# Change filename based on original image selection (JSP/SA)
writeRaster(predicted_raster, "outputs/2000EC/OD4_predicted_EC_raster_2001.tif", format = "GTiff", overwrite = TRUE)
ggsave("outputs/2000EC/OD4_predicted_EC_map_2001.png", plot = predicted_EC, width = 8, height = 6, dpi = 300)



# 4. Mask Aquaculture ponds for final image 
aqua_df <- as.data.frame(aqua_image, xy = TRUE)

#Note: Aqua == 2, Dry Aqua == 1, and others == 0 
unique(values(aqua_image))

# Check df for any Infinite, or NaN values
colSums(is.na(aqua_df))   # Check for NA values
sum(is.infinite(as.matrix(aqua_df)))  # Check for Inf values
sum(is.nan(as.matrix(aqua_df)))  # Check for NaN values

colnames(aqua_df) <- c("x", "y", "classification")

#plot aquaculture 
predicted_Aqua <- ggplot(aqua_df, aes(x = x, y = y, fill = factor(classification))) +
  geom_raster() +
  scale_fill_manual(values = c("0" = "black", "1" = "black", "2" = "blue")) +
  theme_minimal() +
  labs(title = "Classified Aquaculture ponds - 2001", 
       fill = "Aquaculture ponds")

plot(predicted_Aqua)

aqua_mask_raster <- aqua_image == 2  # Create a binary mask of aquaculture areas
# Replace predicted EC values with 2 where aquaculture mask is TRUE
predicted_raster[aqua_mask_raster] <- 2
values(predicted_raster) <- as.numeric(values(predicted_raster))  # Force numeric raster
unique(values(predicted_raster))

# Change filename based on original image selection (JSP/SA)
ggsave("outputs/2000EC/OD4_predicted_Aqua_map_2001.png", plot = predicted_Aqua, width = 8, height = 6, dpi = 300)
writeRaster(aqua_mask_raster, "outputs/2000EC/OD4_predicted_Aqua_raster_2001.tif", format = "GTiff", overwrite = TRUE)


# 5. Plot and save final combined salinity and aqua maps
#convert final predicted raster to df, then ggplot to check 
final_df <- as.data.frame(predicted_raster, xy = TRUE)
colnames(final_df) <- c("x", "y", "predicted_EC")

# Remove NA values before plotting
final_df <- final_df[!is.na(final_df$predicted_EC), ]

final_df$predicted_EC <- factor(final_df$predicted_EC, levels = c(0, 1, 2))


# Convert predicted raster to terra object
r <- rast(predicted_raster)

# Ensure CRS matches with the clipping boundary
boundary <- project(boundary, crs(r))

# Mask outside boundary
r_masked <- mask(r, boundary)
plot(r_masked)

# Colour contriol to match Landsat 8 plots 

# Convert to dataframe
df_masked <- as.data.frame(r_masked, xy = TRUE)
colnames(df_masked) <- c("x", "y", "predicted_EC")

# Drop NA
df_masked <- df_masked[!is.na(df_masked$predicted_EC), ]
df_masked$predicted_EC <- factor(df_masked$predicted_EC, levels = c(0, 1, 2))

# Plot
predicted_ECAqua <- ggplot(df_masked, aes(x = x, y = y, fill = predicted_EC)) +
  geom_raster() +
  scale_fill_manual(
    values = c("0" = "black", "1" = "orange", "2" = "blue"),
    labels = c("0: Non-saline", "1: Saline", "2: Aqua"),
    name = "Predicted Class"
  ) +
  coord_equal() +
  theme_minimal() +
  labs(title = "Predicted EC in Aquaculture (Masked) - 2001")


# Change filename as needed
ggsave("outputs/2000EC/OD4_predicted_ECAqua_map_2001.png", plot = predicted_ECAqua, width = 8, height = 6, dpi = 300)
writeRaster(r_masked, "outputs/2000EC/OD4_predicted_ECAqua_raster_2001.tif", filetype = "GTiff", overwrite = TRUE)
write.csv(final_df, "outputs/2000EC/2001_OD4_predicted_ECAqua_df.csv")

