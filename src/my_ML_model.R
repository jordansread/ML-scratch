library(readr)
write_csv(train, 'data/train_data_predict_temperature.csv')
write_csv(test, '~/Downloads/test_data_predict_temperature.csv')

library(caret)
library(dplyr)
library(xgboost)
library(readr)

train <- read_csv('data/train_data_predict_temperature.csv')
test <- read_csv('data/test_data_predict_temperature.csv')

# create a lagged predictor for solar radiation (I read that may be important):
sol_lagged <- c(1, head(train$solar_radiation,-1L))

# insert the new variable into the third column
train[, 3] <- sol_lagged

train_mtx <- xgb.DMatrix(as.matrix(train[, c(-1, -2)]))
pred_mtx <- xgb.DMatrix(as.matrix(test[, c(-1, -2)]))
y <- train$water_temperature

model_xgb <- caret::train(
  train_mtx, y = y,
  trControl = trainControl(
    allowParallel = TRUE,
    verboseIter = FALSE,
    returnData = FALSE
  ),
  objective ='reg:squarederror',
  tuneGrid = expand.grid(
    list(
      nrounds = seq(100,200),
      max_depth = c(4,12,16),
    colsample_bytree = 1,
    eta = 0.5,
    gamma = 0,
    min_child_weight = 1,
    subsample = 1)),
  method = "xgbTree",
  nthread = 8
)

preds <-  stats::predict(model_xgb, pred_mtx)
obs <- test$water_temperature

plot(obs, preds, xlab = 'predicted temperature (°C)', ylab = 'observed temperature (°C)')
abline(0,1)
