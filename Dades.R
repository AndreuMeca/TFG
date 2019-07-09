library(tidyverse)
library(quantmod)
library(fBasics)
library(zoo)
library(imputeTS)

###Data Import###
#Importem les dades
getSymbols("^GSPC", from = "2007-01-01", to = "2019-05-01", src = "yahoo")
getSymbols("^IBEX", from = "2007-01-01", to = "2019-05-01", src = "yahoo")

#######SP500########
sp500.pr <- GSPC$GSPC.Adjusted
sp500.ret <- diff(log(sp500.pr))
sp500.ret <- sp500.ret[2:3102,]
plot <- sp500.ret
pr <- sp500.pr
ret <- sp500.ret

##2007-2019
##PREUS TRAINING
train.pr <- head(sp500.pr, n = 2851)

##RENDIMENTS TRAINING
train.ret <- head(sp500.ret, n = 2850)
##RENDIMENTS TEST
#ANY
test.ret <- tail(sp500.ret, n = 251)
#MES
test.ret <- head(tail(sp500.ret, n = 251), n = 22)
#DIA
test.ret <- head(tail(sp500.ret, n = 251), n = 1)

##20012 -2018
sp500.pr <- GSPC$GSPC.Adjusted[1261:3020,]
sp500.ret <- diff(log(sp500.pr))
sp500.ret <- sp500.ret[2:1760,]
plot <- sp500.ret
pr <- sp500.pr
ret <- sp500.ret
##PREUS TRAINING
train.pr <- head(sp500.pr, n = 1509)
##RENDIMENTS TRAINING
train.ret <- head(sp500.ret, n = 1508)
##RENDIMENTS TEST
#ANY
test.ret <- tail(sp500.ret, n = 251)
#MES
test.ret <- head(tail(sp500.ret, n = 251), n = 21)
#DIA
test.ret <- head(tail(sp500.ret, n = 251), n = 1)

#########################

###########IBEX35########

##2007-2019
ibex.pr <- IBEX$IBEX.Adjusted
ibex.pr <- na.interpolation(ibex.pr, option = "linear")
ibex.ret <- diff(log(ibex.pr))
ibex.ret <- ibex.ret[2:3147,]
plot <- ibex.ret
pr <- ibex.pr
ret <- ibex.ret
##PREUS TRAINING
train.pr <- head(ibex.pr, n = 2892)
##RENDIMENTS TRAINING
train.ret <- head(ibex.ret, n = 2891)

##20012 -2018
ibex.pr <- IBEX$IBEX.Adjusted[1276:3064,]
ibex.pr <- na.interpolation(ibex.pr, option = "linear")
ibex.ret <- diff(log(ibex.pr))
ibex.ret <- ibex.ret[2:1789,]
##PREUS TRAINING
train.pr <- head(ibex.pr, n = 1534)
##RENDIMENTS TRAINING
train.ret <- head(ibex.ret, n = 1533)

##RENDIMENTS TEST
#ANY
test.ret <- tail(ibex.ret, n = 255)
#MES
test.ret <- head(tail(ibex.ret, n = 255), n = 22)
#DIA
test.ret <- head(tail(ibex.ret, n = 255), n = 1)

################################

#LENGTH FORECAST
length.forecast <- length(test.ret) 
#LENGTH TRAIN PLOT
length.train <- length(train.ret)

################################

###ANÀLISIS EXPLORATORI DADES
#Gràfic preus
autoplot(pr) +
  theme(panel.background = NULL) +
  labs(x = NULL, y = NULL)
#Gràfic rendiments
autoplot(ret) +
  theme(panel.background = NULL) +
  labs(x = NULL, y = NULL)

#FAS i FAP de preus
acf(pr, main = "")
pacf(pr, main = "")

#FAS i FAP de rendiments
acf(ret, main = "")
pacf(ret, main = "")

###Histograma rendiments
hist(ret, breaks = 30, col = "blue", xlab = NULL, ylab = NULL, main = NULL)
###Estadístics bàsics
basicStats(ret)

#Plot train+test
plot.test(test.ret, length.train)

#Plots MASE
mase <- read.csv("Downloads/MASE-DIA.csv", nrows = 11)
colnames(mase)[1] <- "Model"

ggplot(data = mase, aes(x = reorder(Model, +MASE), MASE)) +
  geom_bar(stat = "identity", fill = "grey30",  width = 0.6) +
  theme(panel.background = NULL) +
  labs(x = NULL, y = NULL)

#Criteri d'error MASE
computeMASE <- function(forecast, train, test, period){
  
  forecast <- as.vector(forecast)  #forecast <- forecasted values
  train <- as.vector(train)  #train <- data used for forecasting
  test <- as.vector(test)  #period <- in case of seasonal data, if not, use 1
  
  n <- length(train)
  scalingFactor <- sum(abs(train[(period+1):n]-train[1:(n-period)])) / (n-period)
  
  et <- abs(test - forecast)
  qt <- et/scalingFactor
  meanMASE <- mean(qt)
  return(meanMASE)
  
}

#Create dataframe forecast + plot
#PLOT TRAIN+TEST
plot.test <- function(forecast, length.train.plot) {
  
  date.df <- data.frame(index(plot))
  training.df <- data.frame(index(train.ret), coredata(train.ret))
  forecast.df <- data.frame(index(test.ret), forecast)
  colnames(date.df)[1] <- "date"
  colnames(training.df)[1] <- "date"
  colnames(training.df)[2] <- "train"
  colnames(forecast.df)[1] <- "date"
  colnames(forecast.df)[2] <- "forecast"
  
  df <- merge(date.df, training.df, by = "date", all = T)
  df <- merge(df, forecast.df, by = "date", all = T)
  
  ggplot(df[(nrow(training.df)-length.train.plot):(nrow(training.df)+length.forecast),], aes(date)) +
      geom_line(aes(y = train), color = "black") +
      geom_line(aes(y = forecast), color = "blue") +
      theme(panel.background = NULL) +
      labs(x = NULL, y = NULL)
  
}
#PLOT TRAIN+FORECAST
plot.forecast <- function(forecast, length.train.plot) {
  
  date.df <- data.frame(index(plot))
  training.df <- data.frame(index(train.ret), coredata(train.ret))
  forecast.df <- data.frame(index(test.ret), forecast.mlp)
  colnames(date.df)[1] <- "date"
  colnames(training.df)[1] <- "date"
  colnames(training.df)[2] <- "train"
  colnames(forecast.df)[1] <- "date"
  colnames(forecast.df)[2] <- "forecast"
  
  df <- merge(date.df, training.df, by = "date", all = T)
  df <- merge(df, forecast.df, by = "date", all = T)
  
  ggplot(df[(max(which(!is.na(df$train)))-length.train.plot):(max(which(!is.na(df$train)))+1+length.forecast),], aes(date)) +
    geom_line(aes(y = train), color = "black") +
    geom_line(aes(y = forecast), color = "red") +
    theme(panel.background = NULL) +
    labs(x = NULL, y = NULL)
  
}
#PLOT TRAIN+FORECAST+TEST
plot.forecast.test <- function(forecast, length.train.plot) {
  
  date.df <- data.frame(index(plot))
  training.df <- data.frame(index(train.ret), coredata(train.ret))
  forecast.df <- data.frame(index(test.ret), forecast)
  test.df <- data.frame(index(test.ret), coredata(test.ret))
  colnames(date.df)[1] <- "date"
  colnames(training.df)[1] <- "date"
  colnames(training.df)[2] <- "train"
  colnames(forecast.df)[1] <- "date"
  colnames(forecast.df)[2] <- "forecast"
  colnames(test.df)[1] <- "date"
  colnames(test.df)[2] <- "test"
  
  df <- merge(date.df, training.df, by = "date", all = T)
  df <- merge(df, forecast.df, by = "date", all = T)
  df <- merge(df, test.df, by = "date", all = T)
  
  ggplot(df[(max(which(!is.na(df$train)))-length.train.plot):(max(which(!is.na(df$train)))+1+length.forecast),], aes(date)) +
    geom_line(aes(y = train), color = "black") +
    geom_line(aes(y = test), color = "blue") +
    geom_line(aes(y = forecast), color = "red") +
    theme(panel.background = NULL) +
    labs(x = NULL, y = NULL)
  
}

#Plots MASE
mase <- read.csv("Downloads/MASE-DIA.csv", nrows = 11)
colnames(mase)[1] <- "Model"

ggplot(data = mase, aes(x = reorder(Model, +MASE), MASE)) +
  geom_bar(stat = "identity", fill = "grey30",  width = 0.6) +
  theme(panel.background = NULL) +
  labs(x = NULL, y = NULL)

