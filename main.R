library(data.table)
library(bit64)
library(e1071)
library(kernlab)
library(caret)
library(MASS)
library(pROC)

train <- fread("data/train.csv", head = T, sep = ',')
target <- train$target
train <- subset(train, select=-c(target, ID))
train_len <- nrow(train)
test <- fread("data/test.csv", head = T, sep = ',')
id <- test$ID
test <- subset(test, select=-c(ID))
data <- rbind(train, test)
rm(train)
rm(test)

remove_cols_with_one_value <- function(data) {
    col_ct = sapply(data, function(x) length(unique(x)))
    data[,!names(data) %in% names(col_ct[col_ct==1]), with = F]
}

transform_date_columns <- function(data) {
    date_cols <- c('VAR_0073', 'VAR_0075', 'VAR_0156', 'VAR_0157', 'VAR_0158',
                   'VAR_0159', 'VAR_0166', 'VAR_0167', 'VAR_0168', 'VAR_0169',
                   'VAR_0176', 'VAR_0177', 'VAR_0178', 'VAR_0179', 'VAR_0204',
                   'VAR_0217')
    
    for (col in date_cols) {
        date <- strptime(data[[col]], "%d%B%y:%H:%M:%S")
        data[[paste(col, '_year', sep = '')]] <- date$year
        data[[paste(col, '_mon', sep = '')]] <- date$mon
        data[[paste(col, '_wday', sep = '')]] <- date$wday
        data[[paste(col, '_double', sep = '')]] <- as.double(date)
        if (col == 'VAR_0204') {
            data[[paste(col, '_hour', sep = '')]] <- date$hour
        }
    }
    data[,!names(data) %in% date_cols, with = F]
}

remaining_chars_to_factors <- function(data) {
    char_columns <- c("VAR_0001", "VAR_0005", "VAR_0008", "VAR_0009", "VAR_0010", 
      "VAR_0011", "VAR_0012", "VAR_0043", "VAR_0044", "VAR_0196", "VAR_0200", 
      "VAR_0202", "VAR_0214", "VAR_0216", "VAR_0222", "VAR_0226", "VAR_0229", 
      "VAR_0230", "VAR_0232", "VAR_0236", "VAR_0237", "VAR_0239", "VAR_0274", 
      "VAR_0283", "VAR_0305", "VAR_0325", "VAR_0342", "VAR_0352", "VAR_0353", 
      "VAR_0354", "VAR_0404", "VAR_0466", "VAR_0467", "VAR_0493", "VAR_1934"
    )
    for (col in char_columns) {
        data[[col]] <- as.integer(as.factor(data[[col]]))
    }
    data
}

data <- remove_cols_with_one_value(data)
data <- transform_date_columns(data)
data <- remaining_chars_to_factors(data)
data[is.na(data)] <- -1
pass_columns <- dget(file = "./pass_columns")
data <- data[, pass_columns, with = F]
data$VAR_0212 <- NULL
gc()

train <- data[1:train_len,]
test <- data[(train_len + 1):nrow(data),]
rm(data)

set.seed(1234)
splitIndex <- as.vector(createDataPartition(target, p = .75, 
                                  list = FALSE, times = 1))
train_train <- train[splitIndex,]
train_test <- train[-splitIndex,]
rm(train)


formula <- as.formula(paste('target ~ ', 
                            paste(colnames(train_train), collapse = '+'), 
                            sep = ''))

f <- as.data.frame(train_train)
f['target'] <- target[splitIndex]

objModel <- lda(formula, f)

ff <- as.data.frame(train_test)
x <- predict(objModel, newdata = ff, type = "response")$posterior[,2]

print(auc(as.factor(target[-splitIndex]), x, levels = c(0, 1), algorithm = 2))


objModel1 <- glm(formula,
                data = f,
                family=binomial)

y <- predict(objModel1, newdata = ff, type = "response")

print(auc(as.factor(target[-splitIndex]), y, levels = c(0, 1), algorithm = 2))

z <- predict(objModel1, newdata = test, type = "response")

out <- data.frame(ID = id, target = z)
write.csv(out, file = "out.csv", row.names = F, quote = F) 


# sample data
set.seed(1234)
sm <- sample(nrow(train), 120000)
train <- train[sm]
target <- target[sm]
# sample data

library(xgboost)

dtrain <- xgb.DMatrix(data.matrix(train_train), label=target[splitIndex])
dval <- xgb.DMatrix(data.matrix(train_test), label=target[-splitIndex])

watchlist <- list(eval = dval, train = dtrain)

param <- list(  objective           = "binary:logistic", 
                # booster = "gblinear",
                eta                 = 0.010,
                max_depth           = 7,  # changed from default of 6
                subsample           = 0.7,
                colsample_bytree    = 0.8,
                eval_metric         = "auc"
                # alpha = 0.0001, 
                # lambda = 1
)

set.seed(1234)

clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 7000, # changed from 300 / my best 1300
                    verbose             = 2, 
                    early.stop.round    = 11,
                    watchlist           = watchlist,
                    maximize            = TRUE)

z <- predict(clf, data.matrix(test))
