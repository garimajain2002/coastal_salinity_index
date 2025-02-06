# Script to store frequently used functions


# Define functions
calculate_classification_metrics <- function(actual, predicted) {
  if(!is.null(levels(actual))){
    confusion <- table(actual, predicted)
  } else{
    confusion <- table(factor(actual, levels = c(0, 1)), 
                       factor(predicted, levels = c(0, 1)))
  }

  TN <- ifelse(!is.na(confusion[1, 1]), confusion[1, 1], 0) # True Negative
  FP <- ifelse(!is.na(confusion[1, 2]), confusion[1, 2], 0) # False Positive
  FN <- ifelse(!is.na(confusion[2, 1]), confusion[2, 1], 0) # False Negative
  TP <- ifelse(!is.na(confusion[2, 2]), confusion[2, 2], 0) # True Positive
  
  accuracy <- (TP + TN) / (TN + FP + FN + TP)
  precision <- ifelse((TP + FP) > 0, TP / (TP + FP), NA)
  recall <- ifelse((TP + FN) > 0, TP / (TP + FN), NA)
  f1_score <- ifelse(!is.na(precision) && !is.na(recall) && (precision + recall) > 0,
                     2 * (precision * recall) / (precision + recall), NA)
  
  return(list(
    Accuracy = accuracy,
    Precision = precision,
    Recall = recall,
    F1_Score = f1_score
  ))
}

scale_numeric <- function(data) {
  # Get numeric columns
  numeric_cols <- sapply(data, is.numeric)
  
  # Scale numeric columns
  scaled_data <- data
  scaled_data[, numeric_cols] <- scale(data[, numeric_cols])
  
  return(scaled_data)
}