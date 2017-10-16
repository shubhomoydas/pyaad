#===================
# Streaming Data
#-------------------
# KddCup99
#-------------------
kddcup = read.csv("/Users/moy/work/datasets/anomaly/kddcup/streaming/kddcup-streaming-final.csv", header=T, sep=",")
kddcup_temp = read.csv("/Users/moy/work/datasets/anomaly/kddcup/streaming/kddcup-streaming-temp.csv", header=F, sep=",")
if (nrow(kddcup) != nrow(kddcup_temp)) {
  stop("Number of rows do not match")
}
labels = kddcup[, 1]
original_labels = kddcup_temp[, 1]

labeldf = data.frame(ground.truth=labels, label=original_labels)
head(labeldf)
write.table(labeldf, file="/Users/moy/work/datasets/anomaly/kddcup/streaming/kddcup-streaming-final_orig_labels.csv", 
            row.names = F, col.names = colnames(labeldf), quote=F, sep=",")

#-------------------
# Covtype
#-------------------
set.seed(42)
# separate out the 40 different soil types
covtype = read.csv("/Users/moy/work/datasets/anomaly/covtype/covtype.data", header=F, sep=",")
d = ncol(covtype)

# to simulate drift, we order the data by: 
#   elevation, aspect, slope, Horizontal_Distance_To_Hydrology, 
#   Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways
drift_idxs = order(covtype[, 6], covtype[, 4], covtype[, 5], covtype[, 1], covtype[, 2], covtype[, 3], covtype[, 10], decreasing=F)
# drift_idxs = order(covtype[, 5], covtype[, 4], covtype[, 6], covtype[, 1], covtype[, 2], covtype[, 3], covtype[, 10], decreasing=F)
covtype_ordered = covtype[drift_idxs, ]
head(covtype_ordered[, 1:5])

filtered_idxs = which(covtype_ordered[, d] %in% c(2, 4))
covtype_filtered = covtype_ordered[filtered_idxs, ]
table(covtype_filtered[, d])

y = ifelse(covtype_filtered[, d] == 4, 1, 0)
hist(which(y==1))

covtypedf = data.frame(ground.truth=ifelse(y==1, "anomaly", "nominal"), covtype_filtered[, 1:(d-1)])
write.table(covtypedf, file="/Users/moy/work/datasets/anomaly/covtype/streaming/covtype_1.csv", sep=",", row.names = F, 
            col.names = colnames(covtypedf), quote = F)

#soiltypes = as.matrix(covtype[, (d-40):(d-1)])
#ncol(soiltypes)
#dim(soiltypes)
#p = 1:40
#n_soil = soiltypes %*% p
#head(n_soil)
