#---- Calculate change in aqua and saline areas by year --------------

getwd()

data <- read.csv(unz("yourfile.zip", "filename.csv"))

df_1995 <- read.csv(unz("outputs/1995_JSP_predicted_ECAqua_df.zip","1995_JSP_predicted_ECAqua_df.csv" ))



df_2001 <- read.csv(unz("outputs/1995_JSP_predicted_ECAqua_df.zip","1995_JSP_predicted_ECAqua_df.csv" ))
df_2005 <- read.csv(unz("outputs/2001_JSP_predicted_ECAqua_df.zip","2001_JSP_predicted_ECAqua_df.csv" ))
df_2010 <- read.csv(unz("outputs/2010_JSP_predicted_ECAqua_df.zip","2010_JSP_predicted_ECAqua_df.csv" ))
df_2014 <- read.csv(unz("outputs/2014_JSP_predicted_ECAqua_df.zip","2014_JSP_predicted_ECAqua_df.csv" ))
df_2017 <- read.csv(unz("outputs/2017_JSP_predicted_ECAqua_df.zip","2017_JSP_predicted_ECAqua_df.csv" ))
df_2021 <- read.csv(unz("outputs/2021_JSP_predicted_ECAqua_df.zip","2021_JSP_predicted_ECAqua_df.csv" ))
df_2024 <- read.csv(unz("outputs/2024_JSP_predicted_ECAqua_df.zip","2024_JSP_predicted_ECAqua_df.csv" ))

