imp <- dget(file = "./dump/imp_1343_1330it")

cols <- head(imp$Feature, n = 55)
train <- fread("dump/ttrain.csv", head = T, sep = ',')
target <- dget(file = "./dump/tdata")$target 

for (col in colnames(train)) {
    tmp <- unlist(log(train[, col, with = F] + 1))
    tmp[is.infinite(k)] <- -1
    train[[paste(col, '_log', sep = '')]] <- tmp
}

set.seed(1234)
splitIndexS <- as.vector(createDataPartition(target, p = .70, 
                                            list = FALSE, times = 1))
splitIndex <- as.vector(createDataPartition(splitIndexS, p = 4/7, 
                                            list = FALSE, times = 1))

train_train <- train[splitIndexS[splitIndex],]
train_test <- train[splitIndexS[-splitIndex],]
target_train <- target[splitIndexS[splitIndex]]
target_test <- target[splitIndexS[-splitIndex]] 
rm(train)
gc()

cols_two_times <- c(colnames(train_train), colnames(train_train))
N <- 5
res <- data.frame(colname = 'colname', include = F, result = 0.6, best = 0.6, 
                  decision = T, stringsAsFactors = F)

iter <- function(cols, iters) {
    dtrain <- xgb.DMatrix(data.matrix(train_train[, cols,with = F]), label=target_train)
    dval <- xgb.DMatrix(data.matrix(train_test[, cols, with = F]), label=target_test)
    
    watchlist <- list(eval = dval, train = dtrain)
    
    param <- list(  objective           = "binary:logistic", 
                    eta                 = 0.010,
                    max_depth           = 7,
                    subsample           = 0.7,
                    colsample_bytree    = 0.8,
                    eval_metric         = "auc"
    )
    
    set.seed(1234)
    
    clf <- xgb.cv(params              = param, 
                  data                = dtrain, 
                  nrounds             = iters, # changed from 300 / my best 1300
                  verbose             = T, 
                  nfold               = 3,
                  maximize            = TRUE)
    clf[iters,]$test.auc.mean
}

best <- iter(cols, N)

for (col in cols_two_times) {
    if (!(col %in% cols)) {
        cols <- c(col, cols)
    } else {
        cols <- cols[cols != col]
    }
    cur <- iter(cols, N)
    include <- col %in% cols
    if (cur > best) {
        res <- rbind(res , list(col, include, cur, best, T))
        best <- cur
    } else {
        res <- rbind(res , list(col, include, cur, best, F))
        if (!(col %in% cols)) {
            cols <- c(col, cols)
        } else {
            cols <- cols[cols != col]
        }   
    }
    dput(x = cols, file = "dump/fs_cols")
    write.csv(res, file = "dump/fs.csv", row.names = F, quote = F) 
    gc()
}