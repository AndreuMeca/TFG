library(forecast)
library(rugarch)
library(lmtest)

###ARIMA###

arima <- auto.arima(train.ret)
forecast.arima <- forecast(arima, h = length.forecast)
forecast.arima <- forecast.arima$mean

arima
coeftest(arima)

accuracy(forecast.arima, test.ret)
computeMASE(forecast.arima, train.ret, test.ret, 1)

plot.forecast(forecast.arima, length.train)
plot.forecast(forecast.arima, length.forecast)
plot.forecast.test(forecast.arima, length.train)
plot.forecast.test(forecast.arima, length.forecast)

###GARCH###

final.aic <- Inf
final.order <- c(0, 0)
for(p in 1:2) for(q in 1:2) {
  garch.spec <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(p, q)), 
                           mean.model = list(armaOrder = c(0, 0), include.mean = FALSE), 
                           distribution.model = "sged")
  garch.fit <- ugarchfit(spec = garch.spec, data = train.ret, solver = "hybrid") 
  current.aic <- infocriteria(garch.fit)[1]
  if(current.aic < final.aic){
    final.aic <- current.aic
    final.order <- c(p, q)
    final.fit <- garch.fit
  }
}

garch.spec <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(p, q)), 
                         mean.model = list(armaOrder = c(0, 0),  include.mean = FALSE))
garch.fit <- ugarchfit(spec = garch.spec, data = train.ret, solver = "hybrid")
garch.forecast <- ugarchforecast(garch.fit, n.ahead = length.forecast)
forecast.garch <- garch.forecast@forecast$seriesFor

garch.fit
coeftest(garch.spec)

accuracy(as.ts(forecast.garch), test.ret)
computeMASE(forecast.garch, train.ret, test.ret, 1)

plot.forecast(forecast.garch, length.train)
plot.forecast(forecast.garch, length.forecast)
plot.forecast.test(forecast.garch, length.train)
plot.forecast.test(forecast.garch, length.forecast)

###ARIMA-GARCH###

final.aic <- Inf
final.order <- c(0, 0)
for(p in 0:3) for(q in 0:3) for(r in 1:2) for(s in 1:2) {
  garch.spec <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(r, s)), 
                           mean.model = list(armaOrder = c(p, q), include.mean = TRUE), 
                           distribution.model = "sged")
  garch.fit <- ugarchfit(spec = garch.spec, data = train.ret, solver = "hybrid") 
  current.aic <- infocriteria(garch.fit)[1]
  if(current.aic < final.aic){
    final.aic <- current.aic
    final.order.arma <- c(p, q)
    final.order.garch <- c(r, s)
    final.fit <- garch.fit
  }
}

armagarch.spec <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = final.order.garch), 
                         mean.model = list(armaOrder = final.order.arma))
armagarch.fit <- ugarchfit(spec = armagarch.spec, data = train.ret, solver = "hybrid")
forecast.armagarch <- ugarchforecast(armagarch.fit, n.ahead = length.forecast)
forecast.armagarch <- forecast.armagarch@forecast$seriesFor

armagarch.fit

accuracy(as.ts(forecast.armagarch), test.ret)
computeMASE(forecast.armagarch, train.ret, test.ret, 1)

plot.forecast(forecast.armagarch, length.train)
plot.forecast(forecast.armagarch, length.forecast)
plot.forecast.test(forecast.armagarch, length.train)
plot.forecast.test(forecast.armagarch, length.forecast) 

##################################################
########LINEAR GAUSSIAN STATE SPACE MODELS########
##################################################

library(dlm)
library(quantmod)
library(PerformanceAnalytics)
library(forecast)
library(MuMIn)


###EXPONENTIAL SMOOTHING###

ets <- ets(train.ret, ic = "aicc")
forecast.ets <- forecast(ets, h = length.forecast)
forecast.ets <- forecast.ets$mean

ets

accuracy(forecast.ets, test.ret)
computeMASE(forecast.ets, train.ret, test.ret, 1)

plot.forecast(forecast.ets, length.train)
plot.forecast(forecast.ets, length.forecast)
plot.forecast.test(forecast.ets, length.train)
plot.forecast.test(forecast.ets, length.forecast)

###ARIMA##

final.aic <- Inf
final.order <- c(0, 0, 0)
for(p in 0:5) for(d in 0:1) for(q in 0:5) {
  current.aic <- AIC(Arima(train.ret, order = c(p, d, q)))
  if(current.aic < final.aic){
    final.aic <- current.aic
    final.order <- c(p, d, q)
    final.arima <- Arima(train.ret, order = final.order)
  }
}
forecast.arima.ss <- forecast(final.arima, h = length.forecast)
forecast.arima.ss <- forecast.arima.ss$mean

final.arima

accuracy(forecast.arima.ss, test.ret)
computeMASE(forecast.arima.ss, train.ret, test.ret, 1)

plot.forecast(forecast.arima.ss, length.train)
plot.forecast(forecast.arima.ss, length.forecast)
plot.forecast.test(forecast.arima.ss, length.train)
plot.forecast.test(forecast.arima.ss, length.forecast)

###Stochastic Volatility###

#SP500
getSymbols("^GSPC", from = "2007-01-01", to = "2018-04-30", src = "yahoo")
getSymbols("^GSPC", from = "2012-01-01", to = "2017-12-31", src = "yahoo")
GSPC = GSPC[, "GSPC.Adjusted", drop = FALSE[]]
GSPC.ret = CalculateReturns(GSPC, method = "compound")
GSPC.ret = GSPC.ret[-1, ]*100
colnames(GSPC.ret) = "GSPC"
lnabs.ret = log(abs(GSPC.ret[GSPC.ret != 0])) #compute log of returns
lnabsadj.ret = lnabs.ret + 0.63518

#IBEX35
getSymbols("^IBEX", from = "2007-01-01", to = "2018-04-30", src = "yahoo")
getSymbols("^IBEX", from = "2012-01-01", to = "2017-12-31", src = "yahoo")
IBEX = IBEX[, "IBEX.Adjusted", drop = FALSE[]]
IBEX.ret = CalculateReturns(IBEX, method = "compound")
IBEX.ret = IBEX.ret[-1, ]*100
colnames(IBEX.ret) = "GSPC"
lnabs.ret = log(abs(IBEX.ret[IBEX.ret != 0])) #compute log of returns
lnabsadj.ret = lnabs.ret + 0.63518

buildSV = function(parm){
  parm[3] = exp(parm[3])
  F.mat = matrix(c(1, 0, 1), 1, 3)
  V.val = pi^2/8
  G.mat = matrix(c(1,0,0,0,1,0,0,1,parm[1]), 3, 3, byrow = TRUE)
  W.mat = diag(0, 3)
  W.mat[3, 3] = parm[3]
  m0.vec = c(-0.63518, parm[2], parm[2]/(1-parm[1]))
  C0.mat = diag(1, 3)*1e7
  C0.mat[1, 1] = 1e-7
  C0.mat[2, 2] = 1e-7
  C0.mat[3, 3] = parm[3]/(1-parm[1]^2)
  SV.dlm = dlm(FF=F.mat, V=V.val, GG=G.mat, W=W.mat,
               m0=m0.vec, C0=C0.mat)
  return(SV.dlm)
}

phi.start = 0.9
omega.start = (1-phi.start)*(mean(lnabs.ret))
lnsig2n.start = log(0.1)
start.vals = c(phi.start, omega.start, lnsig2n.start)
SV.mle <- dlmMLE(y=lnabs.ret, parm=start.vals, build=buildSV, hessian = T,
                 lower=c(0, -Inf, -Inf), upper = c(0.999, Inf, Inf))
SV.dlm = buildSV(SV.mle$par)
SV.f <- dlmFilter(lnabs.ret, SV.dlm)
SV.f

unlist(SV.dlm)

SV.s <- dlmSmooth(SV.f)
logvol.s = xts(SV.s$s[-1, 3, drop = F], as.Date(rownames(SV.s$s[-1,])))
colnames(logvol.s) = "Volatility"
#SP500
plot.zoo(cbind(abs(GSPC.ret), exp(logvol.s)), 
         main = "", col = c(4,1), lwd = 1:2, xlab=NULL, ylab = c("S&P500", "Volatilitat"))
#IBEX 35
plot.zoo(cbind(abs(IBEX.ret), exp(logvol.s)), 
         main = "", col = c(4,1), lwd = 1:2, xlab=NULL, ylab = c("IBEX35", "Volatilitat"))

SV.fcst = dlmForecast(SV.f, nAhead = length.forecast)
forecast.sv <- (exp(SV.fcst$f))/100

accuracy(as.ts(forecast.sv), test.ret)
computeMASE(forecast.sv, train.ret, test.ret, 1)

plot.forecast(forecast.sv, length.train)
plot.forecast(forecast.sv, length.forecast)
plot.forecast.test(forecast.sv, length.train)
plot.forecast.test(forecast.sv, length.forecast)
