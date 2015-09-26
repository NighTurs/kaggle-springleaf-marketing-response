library(data.table)
library(bit64)
library(e1071)
library(kernlab)
library(caret)
library(MASS)
library(pROC)

setRefClass("data",
            fields=list(
                data="data.table",
                target="integer",
                train_len="integer",
                id="integer"
            )
)

read_data <- function() {
    print("Read train dataset : started")
    train <- fread("data/train.csv", head = T, sep = ',')
    target <- train$target
    train <- subset(train, select=-c(target, ID))
    train_len <- nrow(train)
    print("Read train dataset : finished")
    print("Read test dataset : started")
    test <- fread("data/test.csv", head = T, sep = ',')
    id <- test$ID
    test <- subset(test, select=-c(ID))
    data <- rbind(train, test)
    print("Read test dataset : finished")
    gc()
    new("data", data = data, target = target, train_len = train_len, id = id)
}

remove_cols_with_one_value <- function(dt) {
    print("Remove cols with one value : started")
    col_ct = sapply(dt$data, function(x) length(unique(x)))
    dt$data <- dt$data[,!names(dt$data) %in% names(col_ct[col_ct==1]), with = F]
    gc()
    print("Remove cols with one value : finished")
}

transform_date_columns <- function(dt) {
    print("Transform date columns : started")
    date_cols <- c('VAR_0073', 'VAR_0075', 'VAR_0156', 'VAR_0157', 'VAR_0158',
                   'VAR_0159', 'VAR_0166', 'VAR_0167', 'VAR_0168', 'VAR_0169',
                   'VAR_0176', 'VAR_0177', 'VAR_0178', 'VAR_0179', 'VAR_0204',
                   'VAR_0217')
    
    for (col in date_cols) {
        date <- strptime(dt$data[[col]], "%d%B%y:%H:%M:%S")
        dt$data[[paste(col, '_year', sep = '')]] <- date$year
        dt$data[[paste(col, '_mon', sep = '')]] <- date$mon
        dt$data[[paste(col, '_wday', sep = '')]] <- date$wday
        dt$data[[paste(col, '_double', sep = '')]] <- as.double(date)
        if (col == 'VAR_0204') {
            dt$data[[paste(col, '_hour', sep = '')]] <- date$hour
        }
    }
    dt$data <- dt$data[,!names(dt$data) %in% date_cols, with = F]
    gc()
    print("Transform date columns : finished")
}

remaining_chars_to_factors <- function(dt) {
    print("Transform remaining char columns to integers : started")
    char_columns <- c("VAR_0001", "VAR_0005", "VAR_0008", "VAR_0009", "VAR_0010", 
      "VAR_0011", "VAR_0012", "VAR_0043", "VAR_0044", "VAR_0196", "VAR_0200", 
      "VAR_0202", "VAR_0214", "VAR_0216", "VAR_0222", "VAR_0226", "VAR_0229", 
      "VAR_0230", "VAR_0232", "VAR_0236", "VAR_0237", "VAR_0239", "VAR_0274", 
      "VAR_0283", "VAR_0305", "VAR_0325", "VAR_0342", "VAR_0352", "VAR_0353", 
      "VAR_0354", "VAR_0404", "VAR_0466", "VAR_0467", "VAR_0493", "VAR_1934"
    )
    for (col in char_columns) {
        dt$data[[col]] <- as.integer(as.factor(dt$data[[col]]))
    }
    gc()
    print("Transform remaining char columns to integers : finished")
}

cleaning_and_transformatins <- function(dt) {
    print("Cleaning and transformations : started")
    remove_cols_with_one_value(dt)
    transform_date_columns(dt)
    remaining_chars_to_factors(dt)
    print("Substitute every NA with -1 : started")
    dt$data[is.na(dt$data)] <- -1
    print("Substitute every NA with -1 : finished")
    print("Filter pre removed features : started")
    pass_columns <- dget(file = "./dump/pass_columns")
    dt$data <- dt$data[, pass_columns, with = F]
    # this variable has HUGE numbers in it, which later cause troubles
    # seems save to remove it cause this is similar to ID's
    dt$data$VAR_0212 <- NULL    
    print("Filter pre removed features : finished")
    gc()
    print("Cleaning and transformations : finished")
}

save_transformed_data <- function(dt) {
    print("Save transformed data : started")
    write.csv(dt$data, file = "dump/tdata.csv", row.names = F, quote = F) 
    dput(x = list(target = dt$target, train_len = dt$train_len,
                  id = dt$id), file = "dump/tdata")
    print("Save transformed data : finished")
}

save_transformed_train <- function(dt) {
    print("Save transformed train : started")
    write.csv(dt$data[1:dt$train_len], file = "dump/ttrain.csv", row.names = F, quote = F) 
    print("Save transformed train : finished")
}

load_transformed_data <- function() {
    print("Load transformed data : started")
    dt <- dget(file = "dump/tdata")
    print("Load transformed data : finished")
    dt
}

read_transform_and_save_data <- function() {
    print("Read, transform save data : started")
    dt <- read_data()
    cleaning_and_transformatins(dt)
    save_transformed_data(dt)
    print("Read, transform save data : finished")
}

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
