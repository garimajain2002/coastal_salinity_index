#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Comparing Band histogram matching vs band normalization 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
library(raster)
library(terra)
library(dplyr)
library(ggplot2)
library(sf)
library(RStoolbox)
library(tidyr)
library(tidyverse)
library(purrr)

# ================ 1. Read data ===============
setwd("C:/Users/garim/Documents/GitHub_LocalRepository/coastal_salinity_index - private")
getwd()

# Read relevant landsat multiband image and aquaculture classification image for masking
# # 2024 for normalization
landsat8_image <- stack("data/tifs/geomedian_harmonised_New5classified/OD_4_geomedian_2024_Feb.tif")
landsat8_image <- landsat8_image[[1:6]] # only keep the bands and not the indices

# For historical maps replace images with respective years and run the same code below
# 1995 JSP
landsat_image <- stack("data/tifs/geomedian_harmonised_New5classified/OD_4_geomedian_1995_predicted_predicted.tif")

names(landsat_image)
names(landsat8_image)

standard_band_names <- c("Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2")

names(landsat_image) <- standard_band_names
names(landsat8_image) <- standard_band_names

landsat_image <- landsat_image * 0.0000275 - 0.2

# Prepare landsat multiband images
# Create a raster where each pixel has a unique ID to use when merging predicted values 
landsat8_df <- as.data.frame(landsat8_image, na.rm = FALSE) # Keep all pixels, including NAs
landsat_df <- as.data.frame(landsat_image, na.rm = FALSE) # Keep all pixels, including NAs

landsat8_df$ID <- seq_len(nrow(landsat8_df))
landsat_df$ID <- seq_len(nrow(landsat_df))

summary(landsat8_df)
summary(landsat_df)

# Make -ve values in Landsat8 NA 
sapply(landsat8_df, function(x) sum(x < 0, na.rm = TRUE))


# Replace negative values in those columns with NA
landsat8_df[standard_band_names] <- lapply(landsat8_df[standard_band_names], function(x) {
  x[x < 0] <- NA
  return(x)
})

summary(landsat8_df)

landsat8_df$method <- "Reference_L8"
landsat_df$method <- "Original_L5"

# -------
# Option 1 : Histogram matching

landsat_matched <- RStoolbox::histMatch(
  x = landsat_image,
  ref = landsat8_image
)

# -------
# Option 2: Band normalization 

# Compute means and sds from L8 and L5
mean_L8 <- colMeans(landsat8_df[, standard_band_names], na.rm = TRUE)
sd_L8 <- apply(landsat8_df[, standard_band_names], 2, sd, na.rm = TRUE)

mean_L5 <- colMeans(landsat_df[, standard_band_names], na.rm = TRUE)
sd_L5 <- apply(landsat_df[, standard_band_names], 2, sd, na.rm = TRUE)

#Additional check
cbind(mean_L5, mean_L8, sd_L5, sd_L8)

# Normalize L5 to L8 scale
landsat_df_norm <- landsat_df
for (band in standard_band_names) {
  landsat_df_norm[[band]] <- ((landsat_df[[band]] - mean_L5[band]) / sd_L5[band]) * sd_L8[band] + mean_L8[band]
}

summary(landsat8_df)
summary(landsat_df)
summary(landsat_df_norm)


# -------
# Comparing normalization and band histogram matching
# Convert histogram-matched raster to df
landsat_matched_df <- as.data.frame(landsat_matched, na.rm = FALSE)
landsat_matched_df$ID <- seq_len(nrow(landsat_matched_df))
names(landsat_matched_df) <- c(standard_band_names, "ID")

# Ensure normalised image has the same IDs 
landsat_df_norm$ID <- landsat_df$ID

# Add method labels to each df
landsat_df_norm$method <- "Normalized_L5"
landsat_matched_df$method <- "HistMatched_L5"

# Add to comparison dataframe
df_all <- bind_rows(
  landsat8_df[, c(standard_band_names, "method")],
  landsat_df[, c(standard_band_names, "method")],
  landsat_df_norm[, c(standard_band_names, "method")],
  landsat_matched_df[, c(standard_band_names, "method")]
)

# Long format
df_long_all <- df_all %>%
  pivot_longer(cols = all_of(standard_band_names), names_to = "Band", values_to = "Value")

# Plot
ggplot(df_long_all, aes(x = Value, fill = method, color = method)) +
  geom_density(alpha = 0.35) +
  facet_wrap(~Band, scales = "free", ncol = 3) +
  theme_minimal(base_size = 14) +
  labs(title = "Reflectance Distribution Across Methods",
       x = "Reflectance", y = "Density") +
  scale_fill_manual(values = c(
    "Reference_L8" = "black", 
    "Original_L5" = "red", 
    "Normalized_L5" = "blue", 
    "HistMatched_L5" = "darkgreen")) +
  scale_color_manual(values = c(
    "Reference_L8" = "black", 
    "Original_L5" = "red", 
    "Normalized_L5" = "blue", 
    "HistMatched_L5" = "darkgreen")) +
  theme(legend.position = "bottom")

# ggsave("outputs/2000EC/comparingNormVsHistMatch.png", width = 12, height = 10, dpi = 300)

# L5 have some extreme tail end values 
# Original image - In many bands (especially Blue, Green, Red), the original L5 (red) distribution appears quite similar in shape and location to the L8 (black) distribution.
# Band Normalisation appears to shrink the distribution in most bands (e.g., NIR, SWIRs). too-peaked, narrow distributions, suggesting over-correction.Less overlap with L8, especially in NIR and SWIR2.
# Histrogram matching - Also diverges notably in SWIR1, SWIR2, and NIR. Introduces longer tails or skew in some bands (e.g., Blue, SWIR1), and might be slightly off in the center? 

# The bands seem to be most aligned after Histogram matching as comapred to normalisation. 
# However, the original histograms still seem better aligned rather that the two normalisations. 
# As long as some of the extreme values are removed from each band, we could potentially just use the original harmonised images without any further normalisation. 
# Option 3: Winsorize the lower tail (first percentile?) 
# Option 4: Drop values more than 3 SDs from mean
# Option 5: Hard threshold and set extreme values as NA 


# ---------------------------
# Option 3: Winsorization by Percentiles (1st and 99th)
landsat_df_win_pct <- landsat_df
for (band in standard_band_names) {
  lower <- quantile(landsat_df[[band]], 0.01, na.rm = TRUE)
  upper <- quantile(landsat_df[[band]], 0.99, na.rm = TRUE)
  landsat_df_win_pct[[band]] <- pmin(pmax(landsat_df[[band]], lower), upper)
}
landsat_df_win_pct$method <- "Winsorized_L5_Percentile"

# ---------------------------
# Option 4: Winsorization by ±3 SD
landsat_df_win_sd <- landsat_df
for (band in standard_band_names) {
  mu <- mean(landsat_df[[band]], na.rm = TRUE)
  sigma <- sd(landsat_df[[band]], na.rm = TRUE)
  lower <- mu - 3 * sigma
  upper <- mu + 3 * sigma
  landsat_df_win_sd[[band]] <- pmin(pmax(landsat_df[[band]], lower), upper)
}
landsat_df_win_sd$method <- "Winsorized_L5_3SD"

# ---------------------------
# Combine all into single comparison dataframe
df_all <- bind_rows(
  landsat8_df[, c(standard_band_names, "method")],
  landsat_df[, c(standard_band_names, "method")],
  landsat_df_norm[, c(standard_band_names, "method")],
  landsat_matched_df[, c(standard_band_names, "method")],
  landsat_df_win_pct[, c(standard_band_names, "method")],
  landsat_df_win_sd[, c(standard_band_names, "method")]
)

# Convert to long format for ggplot
df_long_all <- df_all %>%
  pivot_longer(cols = all_of(standard_band_names), names_to = "Band", values_to = "Value")

# ---------------------------
# Plot
ggplot(df_long_all, aes(x = Value, fill = method, color = method)) +
  geom_density(alpha = 0.3) +
  facet_wrap(~Band, scales = "free", ncol = 3) +
  theme_minimal(base_size = 14) +
  labs(title = "Reflectance Distribution Across Methods",
       x = "Reflectance", y = "Density") +
  scale_fill_manual(values = c(
    "Reference_L8" = "black", 
    "Original_L5" = "red", 
    "Normalized_L5" = "blue", 
    "HistMatched_L5" = "darkgreen",
    "Winsorized_L5_Percentile" = "orange",
    "Winsorized_L5_3SD" = "purple"
  )) +
  scale_color_manual(values = c(
    "Reference_L8" = "black", 
    "Original_L5" = "red", 
    "Normalized_L5" = "blue", 
    "HistMatched_L5" = "darkgreen",
    "Winsorized_L5_Percentile" = "orange",
    "Winsorized_L5_3SD" = "purple"
  )) +
  theme(legend.position = "bottom")

# Save the plot
# ggsave("outputs/2000EC/comparingWinsorizedVsOthers.png", width = 12, height = 10, dpi = 300)

# -------
# Option 5 : Hard drop
# ---------------------------
 
# Some reflectance bins are overpopulated (likely a harmonization artifact)
# Detect values with very high frequency (e.g., top 1% of bins by pixel count)
# Randomly downsample those values, keeping only ~10% or a fixed number
# Leave all other values untouched

#---- The function is flattening the mode a fair bit, and somehow the final compaisons show no change. 
# Function to trim upper tail and downsample lower spike
# clean_landsat5_band_v2 <- function(vec, lower_q = 0.02, upper_q = 0.995, downsample_rate = 0.25) {
#   n <- length(vec)
#   na_idx <- is.na(vec)
#   vec_no_na <- vec[!na_idx]
#   
#   # 1. Winsorize upper tail at upper_q quantile (cap max values)
#   upper_cap <- quantile(vec_no_na, upper_q, na.rm = TRUE)
#   vec_no_na <- pmin(vec_no_na, upper_cap)
#   
#   # 2. Identify low spike values (<= lower_q quantile)
#   lower_cut <- quantile(vec_no_na, lower_q, na.rm = TRUE)
#   spike_idx <- which(vec_no_na <= lower_cut)
#   other_idx <- which(vec_no_na > lower_cut)
#   
#   # 3. For spike values: downsample and replicate back to original count of spike pixels
#   spike_vals <- vec_no_na[spike_idx]
#   spike_count <- length(spike_vals)
#   
#   # Downsample spike by randomly selecting fraction of spike pixels
#   downsampled_spike_vals <- sample(spike_vals, size = max(1, round(spike_count * downsample_rate)), replace = FALSE)
#   
#   # Replicate downsampled spike values to original spike count
#   replicated_spike_vals <- rep(downsampled_spike_vals, length.out = spike_count)
#   
#   # 4. Combine replicated spike values and other values
#   cleaned_vec_no_na <- numeric(length(vec_no_na))
#   cleaned_vec_no_na[spike_idx] <- replicated_spike_vals
#   cleaned_vec_no_na[other_idx] <- vec_no_na[other_idx]
#   
#   # 5. Reinsert NAs back at original positions
#   cleaned_vec <- numeric(n)
#   cleaned_vec[na_idx] <- NA_real_
#   cleaned_vec[!na_idx] <- cleaned_vec_no_na
#   
#   return(cleaned_vec)
# }
# 
# numeric_landsat_df <- landsat_df %>% select(where(is.numeric))
# 
# cleaned_blue <- clean_landsat5_band_v2(landsat_df$Blue)
# 
# summary(cleaned_blue)
# png("outputs/2000EC/Blue_Cleaned_Histogram.png", width = 800, height = 600)
# hist(cleaned_blue, breaks = 100, col = "skyblue", main = "Blue Band After Cleaning")
# dev.off()
# 
# 
# # Clean and pad
# cleaned_landsat_df <- numeric_landsat_df %>%
#   imap_dfc(~ clean_landsat5_band_v2(.x) %>% setNames(.y)) %>%
#   mutate(ID = landsat_df$ID, method = "TailAndSpikeCleaned") %>%
#   select(ID, everything())
# 
# # Verify sizes
# stopifnot(nrow(cleaned_landsat_df) == nrow(landsat_df))
# 
# summary(landsat_df)
# summary(cleaned_landsat_df)
# sapply(cleaned_landsat_df, function(x) table(x == 0, useNA = "ifany"))
# 
# 
# # Generate histogram ggplots for each band
# df_all <- bind_rows(
#   landsat8_df %>% select(all_of(standard_band_names), method),
#   landsat_df %>% select(all_of(standard_band_names), method),
#   cleaned_landsat_df %>% select(all_of(standard_band_names), method)
# )
# 
# df_long <- df_all %>%
#   pivot_longer(cols = all_of(standard_band_names), names_to = "Band", values_to = "Value")
# 
# df_long$method <- factor(df_long$method, levels = c("Reference_L8", "Original_L5", "TailAndSpikeCleaned"))
# 
# ggplot(df_long, aes(x = Value, fill = method, color = method)) +
#   geom_density(alpha = 0.3) +
#   facet_wrap(~ Band, scales = "free") +
#   theme_minimal() +
#   labs(title = "Landsat Bands: Original vs Cleaned vs L8 Reference")
# 
# # Save
# ggsave("outputs/2000EC/Cleaned_L5_Histograms.png", width = 12, height = 8, dpi = 300)
