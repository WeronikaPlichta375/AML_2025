install.packages("data.table", repos = "http://cran.us.r-project.org")
library(data.table)
install.packages("spls")
install.packages("BiocManager")
BiocManager::install("Biobase")
BiocManager::install("golubEsets")
BiocManager::install("multtest")
library(multtest)
library(spls)
library(Biobase)
#Prostate data
data(prostate)
prostate_data <- as.data.frame(prostate$x)
prostate_data$class_index <- prostate$y

write.csv(prostate_data, file = "prostate_dataF.csv", row.names = FALSE)
##Golub data
data(golub)  # Leukemia gene expression data
dim(golub)   # 3051 genes (features) and 38 samples (observations)

gol.fac <- factor(golub.cl, levels=0:1, labels = c("ALL","AML"))
golub_transposed <- t(golub)
golub_df <- as.data.frame(golub_transposed)
golub_df$target <- gol.fac
golub_df
write.csv(golub_df, file = "golub_data_with_target3.csv", row.names = FALSE)

