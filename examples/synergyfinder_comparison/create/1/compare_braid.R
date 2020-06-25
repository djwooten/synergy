library(braidrm)

# Parameters are [IDMA, IDMB, na, nb, δ, κ, E0, EfA, EfB, Ef]
# Parameters are [C1, C2, h1, h2, delta, kappa, E0, E1, E2, E3]
parameters <- c('C1', 'C2', 'h1', 'h2', 'delta', 'kappa', 'E0', 'E1', 'E2', 'E3')

kappa_model <- "kappa3" #c(1,2,3,4,  6,7,8,9,10)
kappa_parameters <- c(1,2,3,4,  6,7,8,9,10)

delta_model <- "delta3" #c(1,2,3,4,5,  7,8,9,10)
delta_parameters <- c(1,2,3,4,5,  7,8,9,10)

full_model <- "ebraid"  #c(1,2,3,4,5,6,7,8,9,10)
full_parameters <- c(1,2,3,4,5,6,7,8,9,10)

df <- read.csv("dataset_1.csv")

#Set the random number seed for generating noises.

set.seed(1)




concs <- df[,c('conc_c','conc_r')]
effect <- df$response/100

braidmodel <- braidrm(concs, effect, fixed=kappa_model)
results <- matrix(nrow=length(kappa_parameters), ncol=3)
for (i in 1:length(kappa_parameters)) {
    results[i, 1] <- braidmodel$ciVec[2*i-1]
    results[i, 2] <- braidmodel$coefficients[i]
    results[i, 3] <- braidmodel$ciVec[2*i]
}
results <- as.data.frame(results, row.names=parameters[kappa_parameters])
colnames(results) <- c("lower","best","upper")
write.csv(results,"kappa_results.csv")





braidmodel <- braidrm(concs, effect, fixed=delta_model)
results <- matrix(nrow=length(delta_parameters), ncol=3)
for (i in 1:length(delta_parameters)) {
    results[i, 1] <- braidmodel$ciVec[2*i-1]
    results[i, 2] <- braidmodel$coefficients[i]
    results[i, 3] <- braidmodel$ciVec[2*i]
}
results <- as.data.frame(results, row.names=parameters[delta_parameters])
colnames(results) <- c("lower","best","upper")
write.csv(results,"delta_results.csv")





braidmodel <- braidrm(concs, effect, fixed=full_model)
results <- matrix(nrow=length(full_parameters), ncol=3)
for (i in 1:length(full_parameters)) {
    results[i, 1] <- braidmodel$ciVec[2*i-1]
    results[i, 2] <- braidmodel$coefficients[i]
    results[i, 3] <- braidmodel$ciVec[2*i]
}
results <- as.data.frame(results, row.names=parameters[full_parameters])
colnames(results) <- c("lower","best","upper")
write.csv(results,"full_results.csv")



