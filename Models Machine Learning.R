###############################
########NEURAL NETWORKS########
###############################

library(forecast)
library(nnfor)
library(MuMIn)
library(NeuralNetTools)

##Autoregressive Neural Network###

nnar <- nnetar(train.ret, ic = "aicc")
forecast.nnar <- forecast(nn, h = length.forecast)
forecast.nnar <- forecast.nnar$mean

nnar
accuracy(forecast.nnar, test.ret)
computeMASE(forecast.nnar, train.ret, test.ret, 1)

plot.forecast(as.vector(forecast.nnar), length.train)
plot.forecast(forecast.nnar, length.forecast)
plot.forecast.test(forecast.nnar, length.train)
plot.forecast.test(forecast.nnar, length.forecast)

##Multi Layer Perceptron###

mlp <- mlp(as.ts(train.pr))
forecast.mlp <- forecast(mlp, h = (length.forecast+1))

mlp
plot(mlp)

forecast.mlp <- diff(log(forecast.mlp$mean))

accuracy(forecast.mlp, test.ret)
computeMASE(forecast.mlp, train.ret, test.ret, 1)

plot.forecast(forecast.mlp, length.train)
plot.forecast(forecast.mlp, length.forecast)
plot.forecast.test(forecast.mlp, length.train)
plot.forecast.test(forecast.mlp, length.forecast)

library(keras)
library(tensorflow)
library(forecast)

# create a lagged dataset, i.e to be supervised learning
lags <- function(x, k){
  lagged =  c(rep(NA, k), x[1:(length(x)-k)])
  DF = as.data.frame(cbind(lagged, x))
  colnames(DF) <- c( paste0('x-', k), 'x')
  DF[is.na(DF)] <- 0
  return(DF)
}

train.lstm = lags(train.ret, 1)
test.lstm = lags(test.ret, 1)

## scale data
normalize <- function(train, test, feature_range = c(0, 1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_train = ((x - min(x) ) / (max(x) - min(x)  ))
  std_test  = ((test - min(x) ) / (max(x) - min(x)  ))
  
  scaled_train = std_train *(fr_max -fr_min) + fr_min
  scaled_test = std_test *(fr_max -fr_min) + fr_min
  
  return( list(scaled_train = as.vector(scaled_train), scaled_test = as.vector(scaled_test) ,scaler= c(min =min(x), max = max(x))) )
}

## inverse-transform
inverter = function(scaled, scaler, feature_range = c(0, 1)){
  min = scaler[1]
  max = scaler[2]
  n = length(scaled)
  mins = feature_range[1]
  maxs = feature_range[2]
  inverted_dfs = numeric(n)
  
  for( i in 1:n){
    X = (scaled[i]- mins)/(maxs - mins)
    rawValues = X *(max - min) + min
    inverted_dfs[i] <- rawValues
  }
  return(inverted_dfs)
}

Scaled = normalize(train.lstm, test.lstm, c(-1, 1))

y_train = Scaled$scaled_train[, 2]
x_train = Scaled$scaled_train[, 1]

y_test = Scaled$scaled_test[, 2]
x_test = Scaled$scaled_test[, 1]

## fit the model
dim(x_train) <- c(length(x_train), 1, 1)
dim(x_train)

#specify required arguments
X_shape2 = dim(x_train)[2]
X_shape3 = dim(x_train)[3]
batch_size = 1
units = 1

model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)

model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02 , decay = 1e-6 ),  
  metrics = c('accuracy')
)

Epochs = 50

nb_epoch = Epochs   
for(i in 1:nb_epoch ){
  model %>% fit(x_train, y_train, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}

L = length.forecast
#dim(x_test) = c(length(x_test), 1, 1)

scaler = Scaled$scaler

predictions = numeric(L)
for(i in 1:L){
  X = x_test[i]
  dim(X) = c(1,1,1)
  # forecast
  yhat = model %>% predict(X, batch_size=batch_size)
  
  # invert scaling
  yhat = inverter(yhat, scaler,  c(-1, 1))
  
  # save prediction
  predictions[i] <- yhat
}

forecast.lstm <- predictions

summary(model)

accuracy(as.ts(forecast.lstm), test.ret)
computeMASE(forecast.lstm, train.ret, test.ret, 1)

plot.forecast(forecast.lstm, length.train)
plot.forecast(forecast.lstm, length.forecast)
plot.forecast.test(forecast.lstm, length.train)
plot.forecast.test(forecast.lstm, length.forecast)

###LSTM###

library(keras)
library(tensorflow)
library(forecast)

diffed = sp500.ret

# create a lagged dataset, i.e to be supervised learning

lags <- function(x, k){
  lagged =  c(rep(NA, k), x[1:(length(x)-k)])
  DF = as.data.frame(cbind(lagged, x))
  colnames(DF) <- c( paste0('x-', k), 'x')
  DF[is.na(DF)] <- 0
  return(DF)
}
supervised = lags(diffed, 1)

## split into train and test sets
N = nrow(supervised)
n = round(N *0.920840064, digits = 0)
train = supervised[1:n, ]
test  = supervised[(n+1):N,  ]


## scale data
normalize <- function(train, test, feature_range = c(0, 1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_train = ((x - min(x) ) / (max(x) - min(x)  ))
  std_test  = ((test - min(x) ) / (max(x) - min(x)  ))
  
  scaled_train = std_train *(fr_max -fr_min) + fr_min
  scaled_test = std_test *(fr_max -fr_min) + fr_min
  
  return( list(scaled_train = as.vector(scaled_train), scaled_test = as.vector(scaled_test) ,scaler= c(min =min(x), max = max(x))) )
}

## inverse-transform
inverter = function(scaled, scaler, feature_range = c(0, 1)){
  min = scaler[1]
  max = scaler[2]
  n = length(scaled)
  mins = feature_range[1]
  maxs = feature_range[2]
  inverted_dfs = numeric(n)
  
  for( i in 1:n){
    X = (scaled[i]- mins)/(maxs - mins)
    rawValues = X *(max - min) + min
    inverted_dfs[i] <- rawValues
  }
  return(inverted_dfs)
}

Scaled = normalize(train, test, c(-1, 1))

y_train = Scaled$scaled_train[, 2]
x_train = Scaled$scaled_train[, 1]

y_test = Scaled$scaled_test[, 2]
x_test = Scaled$scaled_test[, 1]

## fit the model
dim(x_train) <- c(length(x_train), 1, 1)
dim(x_train)
#specify required arguments
X_shape2 = dim(x_train)[2]
X_shape3 = dim(x_train)[3]
batch_size = 1
units = 1

model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)

model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02 , decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50

nb_epoch = Epochs   
for(i in 1:nb_epoch ){
  model %>% fit(x_train, y_train, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}

L = 251
#dim(x_test) = c(length(x_test), 1, 1)

scaler = Scaled$scaler

predictions = numeric(L)
for(i in 1:L){
  X = x_test[i]
  dim(X) = c(1,1,1)
  # forecast
  yhat = model %>% predict(X, batch_size=batch_size)
  
  # invert scaling
  yhat = inverter(yhat, scaler,  c(-1, 1))
  
  # save prediction
  predictions[i] <- yhat
}

accuracy(as.ts(predictions), sp500.test.ret)
computeMASE(predictions, sp500.train.ret, sp500.test.ret, 1)

prediction.lstm <- predictions

plot.forecast.ret(forecast = predictions, length.fore = 251, length.train.plot = 251)
plot.forecast.ret(forecast = predictions, length.fore = 251, length.train.plot = 1508)
plot.forecast.ret.test(forecast = predictions, length.fore = 251, length.train.plot = 251)
plot.forecast.ret.test(forecast = predictions, length.fore = 251, length.train.plot = 1508)

####################################
########k-NEAREST NEIGHBOURS########
####################################

library(tsfknn)

knn <- knn_forecasting(timeS = as.ts(train.ret), h = length.forecast, msas = "recursive")

knn

forecast.knn <- knn$prediction

accuracy(forecast.knn, test.ret)
computeMASE(forecast.knn, train.ret, test.ret, 1)

plot.forecast(forecast.knn, length.train)
plot.forecast(forecast.knn, length.forecast)
plot.forecast.test(forecast.knn, length.train)
plot.forecast.test(forecast.knn, length.forecast)

########################################
########SUPPORT VECTOR REGRESSOR########
########################################

library(e1071)
library(caret)
library(forecast)

#SP500
ret = sp500.ret[1:(nrow(train.ret)+nrow(test.ret))]
#IBEX35
ret = ibex.ret[1:(nrow(train.ret)+nrow(test.ret))]

##Model Inputs:
##Define matrix of features (each column is a feature)
#Features: lags 1,2,3, 4, 5
feat = merge(na.trim(lag(ret, 1)),
             na.trim(lag(ret, 2)),
             na.trim(lag(ret, 3)),
             na.trim(lag(ret, 4)), 
             na.trim(lag(ret, 5)),
             all=FALSE)

##add TARGET. We want to predict RETURN
dataset = merge(feat, ret, all=TRUE)

colnames(dataset) = c("lag.1", "lag.2", "lag.3","lag.4", "lag.5",
                      "TARGET")

##Divide data into training and testing. Use caret methods
##process class sets as data frames
training = as.data.frame(dataset[6:(nrow(train.ret)),])
rownames(training) = NULL
testing = as.data.frame(dataset[(nrow(train.ret)+1):(nrow(ret)),])
rownames(testing) = NULL

##Train model
#parameters that can be tuned
type="eps-regression" ##regression
u= -2 ## -3,-2,-1,0,1,2,3
gam=10^{u}
w= 2 ##1.5,-1,0.5,2,3,4
cost=10^{w}
##The higher the cost produce less support vectors, increases accuracy
##However we may overfit
svmFit = svm (training[,-ncol(training)], training[,ncol(training)],
              type=type,
              kernel= "radial",
              gamma=gam,
              cost=cost
)
summary(svmFit)

##build SVM predictor
predsvm = predict(svmFit, testing[,-ncol(testing)])
forecast.svr <- predsvm

accuracy(as.ts(forecast.svr), test.ret)
computeMASE(forecast.svr, train.ret, test.ret, 1)

plot.forecast(forecast.svr, length.train)
plot.forecast(forecast.svr, length.forecast)
plot.forecast.test(forecast.svr, length.train)
plot.forecast.test(forecast.svr, length.forecast)