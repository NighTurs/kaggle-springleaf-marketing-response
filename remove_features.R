set.seed(1234)
ind <- sample(1:nrow(data), 40000)

dt <- data[ind,]
dt$VAR_0212 <- NULL

cr <- cor(scale(dt,center=TRUE,scale=TRUE), method="pearson")
cr2 <- cor(scale(dt,center=TRUE,scale=TRUE), method="spearman")


cols <- colnames(cr)

torm <- NULL

for (c in cols) {
    torm <- c(torm, cols[which(cols > c & 
                                   abs(cr[c, ]) > 0.96 & 
                                   abs(cr2[c, ]) > 0.96)])
}

# with sample 4000 0.8 1179
# with sample 4000 0.9 886 
# with sample 4000 0.95 661
# with sample 20000 0.8 1215  
# with sample 20000 0.9 931
# with sample 20000 0.95 706
# with sample 40000 0.8 1221
# with sample 40000 0.9 933
# with sample 40000 0.95 710
# with sample 40000 0.96 551

data[,unique(torm)] <- NULL

cols <- colnames(data)
row <- nrow(data)
torm <- NULL
for (c in cols) {
    d <- table(data[,c,with=F])
    if (any(d / row > 0.99)) {
        torm <- c(torm, c)
    }
}

data[, torm] <- NULL

pass_columns <- colnames(data)
dput(pass_columns, file = "pass_columns")

