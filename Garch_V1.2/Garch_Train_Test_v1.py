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
    
def GARCH_model(data, vol_type,p_val,q_val, o_val):
    
    model_var = arch_model(data, mean='HARX', vol=vol_type, p=p_val, o=o_val, q=q_val)
    
    return model_var
    

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

#meanwindow = 2;
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

for i in range(0,numhours-1,5):
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
#    seed(1)
# create dataset
    data00 = realradardata#np.sqrt(np.multiply(realradardata,realradardata)+np.multiply(imagradardata,imagradardata))
    data01 = imagradardata#np.sqrt(np.multiply(realradardata,realradardata)+np.multiply(imagradardata,imagradardata))
    data0 = data00
    data1 = data01
    print(np.sqrt(np.var(data00)))
    print(data00.size)
    
    squared_data = [x*x for x in data1]
    squared_data = np.asarray(squared_data)
    segmentlength = 4096
    train_length = segmentlength*3/4
    test_length = segmentlength/4
    
    n_test = 2*segmentlength
    train, test = data0[:-n_test], data0[-n_test:-segmentlength]
    trainb, testb = data1[:-n_test], data1[-n_test:-segmentlength]
    test2 = data0[segmentlength+1:-1]
    test2b = data1[segmentlength+1:-1]
    #train2, test2 = data0[segmentlength+1:-segmentlength], data0[-segmentlength:-1]
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
    
    # plot the actual variance
    
    
    sum1=0    # To store sum of stream 
    sumsq=0  # To store sum of square of stream 
    n=0      # To store count of numbers 
    i=0
    actual_variance=[]
    print(len(train))
    while(i<len(train)/2): 
        x=train[i]
        i+=1
        sum1+=x 
        sumsq+=x**2 
        #Variance 
        var = (sumsq/i) - (sum1/i ** 2) 
        actual_variance.append(var)
    #ctual_variance=np.var(train)
    # print('Actual_Variance', actual_variance)
    plt.plot(actual_variance[:])
    
    #Train Model
    train_modele = GARCH_model(train, 'EGARCH', pval1, qval1, pval1)
    train_modelg1 = GARCH_model(train, 'GARCH', pval1, qval1, pval1)
    
    train_modeleb = GARCH_model(trainb, 'EGARCH', pval1, qval1, pval1)
    train_modelg1b = GARCH_model(trainb, 'GARCH', pval1, qval1, pval1)
    
    
    # fit model
    train_modele_fit = train_modele.fit(options={'maxiter': 250})
    train_modelg_fit1= train_modelg1.fit(options={'maxiter':250})
    train_modeleb_fitb = train_modeleb.fit(options={'maxiter': 250})
    train_modelgb_fit1b = train_modelg1b.fit(options={'maxiter': 250})
    
    # mod_plot=train_modele_fit.plot()
    
    print('Model_fit', train_modele_fit)
    print('Model_fit1', train_modelg_fit1)
    print('Model_fitb', train_modeleb_fitb)
    print('Model_fit1b', train_modelgb_fit1b)
    
    scale_e = train_modele_fit.scale
    print('Scale', scale_e)
    scale1_g = train_modelg_fit1.scale
    scaleb_e = train_modeleb_fitb.scale
    scale1b_g = train_modelgb_fit1b.scale
    
    # forecast the test set
    yhat = train_modele_fit.forecast(horizon=segmentlength, start=train_length, method="simulation", reindex=False)
    print('yhat',yhat)
    yhat1 = train_modelg_fit1.forecast(horizon=segmentlength, start=train_length, method="simulation", reindex=False)
    print('yhat',yhat1)
    yhatb = train_modeleb_fitb.forecast(horizon=segmentlength, start=train_length, method="simulation")
    print('yhat',yhatb)
    yhat1b = train_modelgb_fit1b.forecast(horizon=segmentlength, start=train_length, method="simulation")
    print('yhat',yhat1b)
    # yhat_plot=plt.plot(yhat.variance, label='Variance EGARCH', color='b')
    # yhat_plot=plt.plot(yhat1.variance, label='Variance GARCH', color='g')
    # yhat_plot=yhat.variance.plot()
    ys=yhat.variance.values[::5]
    xs= [x for x in range(len(ys))]
    print('Variance', yhat.variance)
    print('Variance_Values', yhat.variance.values)
    print('Mean_Values', yhat.mean.values)
    print('Mean', yhat.mean)
    actual_variance=np.var(train)
    print('Actual_Variance', actual_variance)
    # plt.plot(xs, ys, color='r')

    curvar1 = yhat.variance.values/scale_e
    print('Curvar1', curvar1)
    curvar1g = yhat1.variance.values/scale1_g
    curvarb = yhatb.variance.values/scaleb_e
    curvar1bg = yhat1b.variance.values/scale1b_g
    
    plt.plot(yhat.variance.values[-1,:], color='b')
    plt.plot(yhat1.variance.values[-1,:], color='r')
    plt.plot(actual_variance, color='g')
    #plt.plot(curvar1, numhours, label='Variance EGARCH', color='b')
    # curvar_plot=plt.plot(curvar1g, label='Variance GARCH', color='r')
    curvar1 = np.hstack((curvar1[None,:],curvarb[None,:]))
    curvar1g = np.hstack((curvar1g[None,:],curvar1bg[None,:]))
    
    # curvar_plot=plt.plot(curvar1, label='Variance EGARCH', color='b')
    # curvar_plot=plt.plot(curvar1g, label='Variance GARCH', color='r')
    
    
    # print('curvar1_size',len(curvar1))
    # print('curvar1_median',np.median(curvar1))
    # print('curvar1g_size',len(curvar1g))
    # print('curvar1g_median',np.median(curvar1g))
    
    
    # #Test set
    # #test_data=data01
    # #test = test_data[-n_test:]
    # n_test=3*segmentlength
    # print('Test2', len(test2))
    # print('Test2b', len(test2b))
    #Test model
    # test_modelg = arch_model(test2, mean='HARX', lags=130, vol='GARCH', p=130, q=130, power=1.0, rescale=True)
    # test_modele = arch_model(test2, mean='HARX', lags=130, vol='EGARCH', p=130, q=130, power=1.0, rescale=True)
    # test_modelgb = arch_model(test2b, mean='HARX', lags=130, vol='GARCH', p=130, q=130, power=1.0, rescale=True)
    # test_modeleb = arch_model(test2b, mean='HARX', lags=130, vol='EGARCH', p=130, q=130, power=1.0, rescale=True)
    # print('Garch_Test',test_modelg)
    # print('EGarch test', test_modele)
    
    # #Fitting the Test Model
    # test_modelg_fit = test_modelg.fit(options={'maxiter': 250})
    # test_modele_fit = test_modele.fit(options={'maxiter': 250})
    # test_modelgb_fit = test_modelgb.fit(options={'maxiter': 250})
    # test_modeleb_fit = test_modeleb.fit(options={'maxiter': 250})
    
    # #Test Scale
    # test_gscale = test_modelg_fit.scale
    # print('Garch_Scale', test_gscale)
    # test_escale = test_modele_fit.scale
    # print('EGARCH_Scale', test_escale)
    # test_gscaleb = test_modelgb_fit.scale
    # test_escaleb = test_modeleb_fit.scale
    # forecast the test set
    # yhatg = test_modelg_fit.forecast(horizon=test_length, start=train_length, method="simulation", reindex=False)
    # yhate = test_modele_fit.forecast(horizon=test_length, start=train_length, method="simulation", reindex=False)
    # yhatgb = test_modelg_fit.forecast(horizon=test_length, start=train_length, method="simulation", reindex=False)
    # yhateb = test_modele_fit.forecast(horizon=test_length, start=train_length, method="simulation", reindex=False)
    
    
    # yhatg = train_modelg_fit1.forecast(horizon=test_length, start=train_length, method="simulation", reindex=False)
    # yhate = train_modele_fit.forecast(horizon=test_length, start=train_length, method="simulation", reindex=False)
    # yhatgb = train_modelgb_fit1b.forecast(horizon=test_length, start=train_length, method="simulation", reindex=False)
    # yhateb = train_modeleb_fitb.forecast(horizon=test_length, start=train_length, method="simulation", reindex=False)


    
    # curvar2 = yhatg.variance.values/scale_e
    # #curvar1g = yhat1.variance.values/scale1_g
    
    
    # curvar = np.hstack((curvar[None,:],curvar2[None,:]))
    # print('curvar',curvar)
    # print(yhatg)
    # print(yhate)
    
    averagedata1 = curvar1
    averagedatalist1 = averagedata1.tolist()
    print('average_data_curvar1',averagedata1)
    of = open(outputfilename, "a")
    for v in averagedatalist1:
    	of.write(f"{v}\n")
    averagedata1g = curvar1g
    averagedatalist1g = averagedata1g.tolist()
    print('average_data_curvar1g',averagedata1g)
    of = open(outputfilename, "a")
    for v in averagedatalist1g:
    	of.write(f"{v}\n")
    # averagedata2 = curvar2
    # averagedatalist2 = averagedata2.tolist()
    # print('Average_data_curvar2',averagedata2)
    # of = open(outputfilename, "a")
    # for v in averagedatalist2:
    # 	of.write(f"{v}\n")
    of.close()
	
	#print(estimatedvar)

et=dt.datetime.now()
End_time=et.timestamp()


exec_time = End_time - start_time
print('Execution Time', exec_time)
    


