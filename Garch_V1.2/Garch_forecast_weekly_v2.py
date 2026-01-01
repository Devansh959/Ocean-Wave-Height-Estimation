# example of ARCH model
from random import gauss
from random import seed
from matplotlib import pyplot as plt
from arch import arch_model
from arch.__future__ import reindexing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from arch.univariate import MIDASHyperbolic
from scipy import signal
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, q_stat
from numpy import log
import numpy as np
import math
from math import sqrt
from scipy.stats import probplot as probplot
import pmdarima as pm
import pandas
import statsmodels.api as sm
import moment
from sklearn.metrics import mean_squared_error
import datetime as dt

ct=dt.datetime.now()
start_time=ct.timestamp()
print(start_time)

path = '/Users/HP/Downloads/Garch_V1.2/TimeSeriesData/'
# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
	# prepare training dataset
	train_size = int(len(X) * 0.66)
	train, test = X[0:train_size], X[train_size:]
	history = [x for x in train]
	# make predictions
	predictions = list()
	for t in range(len(test)):
		model = ARIMA(history, order=arima_order)
		model_fit = model.fit()
		yhat = model_fit.forecast()[0]
		predictions.append(yhat)
		history.append(test[t])
	# calculate out of sample error
	rmse = math.sqrt(mean_squared_error(test, predictions))
	return rmse

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        print(p)
        for d in d_values:
            print(d)
            for q in q_values:
                print(q)
                order = (p,d,q)
                try:
                    rmse = evaluate_arima_model(dataset, order)
                    if rmse < best_score:
                        best_score, best_cfg = rmse, order
                    print('ARIMA%s RMSE=%.3f' % (order,rmse))
                except:
                    continue
    print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))

def hurst(ts):
    """Returns the Hurst Exponent of the time series vector ts"""
    # Create the range of lag values
    lags = range(2, 100)

    # Calculate the array of the variances of the lagged differences
    tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]

    # Use a linear fit to estimate the Hurst Exponent
    poly = np.polyfit(np.log(lags), np.log(tau), 1)

    # Return the Hurst exponent from the polyfit output
    return poly[0]*2.0

def plot_correlogram(x, lags=None, title=None):    
    lags = min(10, int(len(x)/5)) if lags is None else lags
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    x.plot(ax=axes[0][0])
    q_p = np.max(q_stat(plot_acf(x, nlags=lags), len(x))[1])
    stats = f'Q-Stat: {np.max(q_p):>8.2f}\nADF: {adfuller(x)[1]:>11.2f} \nHurst: {round(hurst(x.values),2)}'
    axes[0][0].text(x=.02, y=.85, s=stats, transform=axes[0][0].transAxes)
    probplot(x, plot=axes[0][1])
    mean, var, skew, kurtosis = moment(x, moment=[1, 2, 3, 4])
    s = f'Mean: {mean:>12.2f}\nSD: {np.sqrt(var):>16.2f}\nSkew: {skew:12.2f}\nKurtosis:{kurtosis:9.2f}'
    axes[0][1].text(x=.02, y=.75, s=s, transform=axes[0][1].transAxes)
    plot_acf(x=x, lags=lags, zero=False, ax=axes[1][0])
    plot_pacf(x, lags=lags, zero=False, ax=axes[1][1])
    axes[1][0].set_xlabel('Lag')
    axes[1][1].set_xlabel('Lag')
    fig.suptitle(title, fontsize=20)
    fig.tight_layout()
    fig.subplots_adjust(top=.9)

numhours = 50
outputfilename = 'GARCH_HARX_var_Weekly3.txt'

meanwindow = 2;
estimatedvar = [];

curvar1 = 60
curvar2 = 60
curvar = 60

radardata = [];
for i in range(1,numhours):
   print('filename_iteration',i)
   filename = path + 'TimeSeriesData' + str(i) + '.txt'
   inputfile = open(filename)
   for line in inputfile:
      radardata = radardata + [float(x)/1000 for x in line.split()];
   inputfile.close()

for i in range(0*5,numhours-1,5):
    print('data_iteration',i)
    radardatawindow = radardata[i*8192:(i+4)*8192];

    realradardata = radardatawindow[::2]
    imagradardata = radardatawindow[1::2]
#print(realradardata)

    realradardata = np.asarray(realradardata)
    imagradardata = np.asarray(imagradardata)

    print('RealDataSize',realradardata.size)
    print('ImagDataSize',imagradardata.size)

# seed pseudorandom number generator
    seed(1)
# create dataset
    data00 = realradardata#np.sqrt(np.multiply(realradardata,realradardata)+np.multiply(imagradardata,imagradardata))
    data01 = imagradardata#np.sqrt(np.multiply(realradardata,realradardata)+np.multiply(imagradardata,imagradardata))
    data0 = data00
    data1 = data01
    print(np.sqrt(np.var(data00)))
    print(data00.size)

    #windowsize = 8 
    #data0=np.convolve(data00, np.ones(windowsize), 'valid')/windowsize
    #data0 = data0[::4]
    #data = np.sqrt(np.multiply(data0,data0)+np.multiply(data1,data1))
    #result = adfuller(data0)
    #print('ADF Statistic: %f' % result[0])
    #print('p-value: %f' % result[1])

    #p_values = [0, 1, 2, 4, 6, 8, 10]
    #d_values = range(0, 3)
    #q_values = range(0, 3)
    #warnings.filterwarnings("ignore")
    #evaluate_models(data0, p_values, d_values, q_values)
    #data1 = pandas.DataFrame(data0)
    #data = data1.copy()
    #data['Log_Return'] = np.log(data1).diff().mul(100)
    #data['Return'] = 100 * (data1.pct_change())

    #print(hurst(data))

    #data['Log_Return'][0] = data['Log_Return'][1] 
    #print(data['Log_Return'])
#    model = pm.auto_arima(data['Log_Return'],
#
#    d=0, # non-seasonal difference order
#    start_p=1, # initial guess for p
#    start_q=1, # initial guess for q
#    max_p=20, # max value of p to test
#    max_q=20, # max value of q to test
#
#    seasonal=False, # is the time series seasonal
#
#    information_criterion='bic', # used to select best model
#    trace=True, # print results whilst training
#    error_action='ignore', # ignore orders that don't work
#    stepwise=True, # apply intelligent order search
#
#    )
#    print(model.summary())
    #data = np.convolve(data, np.ones(meanwindow)/meanwindow, mode='valid')
    #data = data[::meanwindow]
    #data0 - np.mean(data0)
    squared_data = [x*x for x in data1]
    squared_data = np.asarray(squared_data)

    # check correlations of squared observations
    # create acf plot
    #plot_pacf(squared_data,lags=np.arange(500))
    #pyplot.show()
    #plot_acf(squared_data,lags=np.arange(500))
    #pyplot.show()
    #data0 = data0 - mean(data0)
    #n_test = 3*2048
    #n_test = 2*4096
    segmentlength = 4096
    n_test = 2*segmentlength
    #train, test = data['Log_Return'][:-n_test], data['Log_Return'][-n_test:]
    #train, test = data0[:-n_test], data0[-n_test:-segmentlength]
    train, test = data0[:-n_test], data0[-n_test:-segmentlength]
    trainb, testb = data1[:-n_test], data1[-n_test:-segmentlength]

    train2 = data0[segmentlength+1:-1]
    train2b = data1[segmentlength+1:-1]
    #train2, test2 = data0[segmentlength+1:-segmentlength], data0[-segmentlength:-1]
    #model = arch_model(train, mean='HARX', vol='GARCH', p=60, q=90)
    #model = ARIMA(train, order=(15,0,5))
    pval1 = 70#int(np.round(12*np.sqrt(np.mean(curvar1))))
    pval2 = 70#int(np.round(12*np.sqrt(np.mean(curvar2))))
    if (np.isnan(np.median(curvar1)) or (np.isnan(np.median(curvar2)))):
        qval1 = 96 
    else:
        qval1 = int(12*np.round(np.sqrt(0.5*(np.mean(curvar1)+np.mean(curvar2)))))
    if (qval1 > 425):
        qval1 = 425
    if (np.isnan(np.median(curvar1)) or (np.isnan(np.median(curvar2)))):
        qval2 = 96 
    else:
        qval2 = int(12*np.round(np.sqrt(0.5*(np.mean(curvar1)+np.mean(curvar2)))))
    if (qval2 > 425):
        qval2 = 425
    print('pval1',pval1)
    print('qval1',qval1)
    print('qval2',qval2)

    model = arch_model(train, mean='HARX', vol='EGARCH', p=pval1, o=pval1, q=qval1) 
    #model2 = arch_model(train, mean='HARX', vol='FIGARCH', p=1, q=1, power=1.0)

    model2 = arch_model(train2, mean='HARX', vol='EGARCH', p=pval2, o=pval2, q=qval2)

    modelb = arch_model(trainb, mean='HARX', vol='EGARCH', p=pval1, o=pval1, q=qval1)
    #modelb = arch_model(trainb, mean='HARX', vol='FIGARCH', p=1, q=1, power=1.0)
    model2b = arch_model(train2b, mean='HARX', vol='EGARCH', p=pval2, o=pval2, q=qval2)
    #model = MIDASHyperbolic(m=60, asym=True)#power=1.0, rescale=True)
    #modelb = MIDASHyperbolic(m=60, asym=True)#power=1.0, rescale=True)
    #arma_model = sm. tsa.SARIMAX(endog=data['Log_Return'],order=(8,0,8))
    #model_fit = arma_model.fit(maxiter=250)
    #print(arma_model)

    # fit model
    model_fit = model.fit(options={'maxiter': 5000})
    model_fit2 = model2.fit(options={'maxiter': 5000})
    model_fitb = modelb.fit(options={'maxiter': 5000})
    model_fit2b = model2b.fit(options={'maxiter': 5000})
    print('Model_fit', model_fit)
    print('Model_fit2', model_fit2)
    print('Model_fitb', model_fitb)
    print('Model_fit2b', model_fit2b)
    ######print(model_fit.summary())

    scale = model_fit.scale
    scale2 = model_fit2.scale
    scaleb = model_fitb.scale
    scale2b = model_fit2b.scale
    # forecast the test set
    yhat = model_fit.forecast(horizon=segmentlength, method="simulation")
    print('yhat',yhat)
    yhat2 = model_fit2.forecast(horizon=segmentlength, method="simulation")
    print('yhat2',yhat2)
    yhatb = model_fitb.forecast(horizon=segmentlength, method="simulation")
    print('yhatb',yhatb)
    yhat2b = model_fit2b.forecast(horizon=segmentlength, method="simulation")
    print('yhat2b',yhat2b)
    #ratio = np.sqrt((np.mean(yhat2.variance.values/scale2 + np.mean(yhat2b.variance.values/scale2b))))/np.sqrt(np.mean(yhat.variance.values/scale + np.mean(yhatb.variance.values/scaleb)))
    #print(ratio)
    yhat_plot= plt.plot(yhat.mean, label='Mean', color='r')
    yhat_plot=plt.plot(yhat2.residual_variance, label='Residual Variance2', color='b')
    yhat_plot=plt.plot(yhatb.residual_variance, label='Residual Varianceb', color='g')

    curvar1 = yhat.variance.values/scale
    curvarb = yhatb.variance.values/scaleb
    #curvar2 = 0.5*(yhat2.variance.values/scale2+yhat2b.variance.values/scale2b)
    curvar2 = yhat2.variance.values/scale2
    curvar2b = yhat2b.variance.values/scale2
    #curvar = np.concatenate([curvar1, curvarb])
    curvar1 = np.hstack((curvar1[None,:],curvarb[None,:]))
    curvar2 = np.hstack((curvar2[None,:],curvar2b[None,:]))
    #curvar = np.hstack((curvar[None,:],curvar2[None,:]))
    #curvar = curvar1
    print('curvar1_size',len(curvar1))
    print('curvar1_median',np.median(curvar1))
    print('curvar2_size',len(curvar2))
    print('curvar2_median',np.median(curvar2))
    #Curvar1_plot= plt.plot(curvar1, color= 'r')
    #curvar2_plot=plt.plot(curvar2,color='b')
#data = imagradardata 
#data = np.convolve(data, np.ones(meanwindow)/meanwindow, mode='valid')
#data = data[::meanwindow]
#n_test = 3*8192
#train, test = data[:-n_test], data[-n_test:]
#model = arch_model(train, mean='HARX', lags=130, vol='GARCH', p=130, q=130, power=1.0, rescale=True)
#model_fit = model.fit(options={'maxiter': 250})
#scale = model_fit.scale
# forecast the test set
#yhat = model_fit.forecast(horizon=n_test, method="simulation", reindex=False)
# fit model
#    model_fit = model.fit()
# forecast the test set
# plot the actual variance
# var = [i*0.01 for i in range(0,100)]
# pyplot.plot(var[-n_test:])
# plot forecast variance
# pyplot.plot(yhat.variance.values[-1, :])
# pyplot.show()
#curvar2 = yhat.variance.values/scale
#print(curvar2)
#meanwindow2 = 2048;
#meanwindow2 = 4096;
#data1 = np.convolve(curvar1[0], np.ones(meanwindow2)/meanwindow2, mode='valid')
#data1 = data1[::meanwindow2]
#data2 = np.convolve(curvar2[0], np.ones(meanwindow2)/meanwindow2, mode='valid')
#data2 = data2[::meanwindow2]
#data1 = curvar1[0]
#data2 = curvar2[0]
#print(curvar[-20:])
#print(data1)
#print(data2)
#averagedata = 0.5*(data1+data2)

    averagedata1 = curvar1
    averagedatalist1 = averagedata1.tolist()
    print('average_data_curvar1',averagedata1)
    of = open(outputfilename, "a")
    for v in averagedatalist1:
    	of.write(f"{v}\n")
    averagedata2 = curvar2
    averagedatalist2 = averagedata2.tolist()
    print('Average_data_curvar2',averagedata2)
    of = open(outputfilename, "a")
    for v in averagedatalist2:
    	of.write(f"{v}\n")
    of.close()
	
	#print(estimatedvar)

et=dt.datetime.now()
End_time=et.timestamp()


exec_time = End_time - start_time
print('Execution Time', exec_time)