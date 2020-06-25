library(synergyfinder)
library(reshape2)


df <- read.csv("dataset_1.csv")

#Set the random number seed for generating noises.

set.seed(1)
dose.response.mat <- ReshapeData(df ,data.type = "viability", impute = TRUE, noise = TRUE, correction = "non")

#PlotDoseResponse(dose.response.mat)
PlotDoseResponse(dose.response.mat, save.file = TRUE)



synergy.score <- CalculateSynergy(data = dose.response.mat,method = "ZIP")

#PlotSynergy(synergy.score, type = "all")
PlotSynergy(synergy.score, type = "all", save.file = TRUE)




synergydf <- melt(synergy.score$scores[1])[,c(1:3)]
colnames(synergydf) <- c("d2","d1","synergy")
write.csv(synergydf, "synergyfinder_output.csv", row.names=FALSE)


