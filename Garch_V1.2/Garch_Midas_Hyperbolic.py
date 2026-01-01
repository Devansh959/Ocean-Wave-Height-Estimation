#example of ARCH model
from random import gauss
from random import seed
from matplotlib import pyplot
from arch import arch_model
from arch.univariate.mean import HARX, ZeroMean
from arch.univariate.volatility import GARCH, FixedVariance
from arch.__future__ import reindexing
from arch.univariate import MIDASHyperbolic
import math
import numpy as np
import datetime as dt

ct=dt.datetime.now()
start_time=ct.timestamp()
print(start_time)

path = '/Users/HP/Downloads/Garch_V1.2/TimeSeriesData/'
numhours = 10
estimatedvar = np.zeros(shape=(numhours,1))
outputfilename = 'GARCH_var_Zigzag_Updated.txt'
of = open(outputfilename, "w")

meanwindow = 3

for i in range(1,numhours):
   print("Numhours",i)
   radardata = [];
   filename = path + 'TimeSeriesData' + str(i) + '.txt'
   inputfile = open(filename)
   for line in inputfile:
      radardata.append([float(x)/10000 for x in line.split()])
   inputfile.close()

   radardata = np.array(radardata[0])
   realradardata = 10*radardata[::2]
   imagradardata = 10*radardata[1::2]
   #segmentlength = 4096
   segmentlength=2048
   
   print('Realdata', realradardata)
   print('imagdata', imagradardata)

   realradardata = np.asarray(realradardata, dtype=float)
   imagradardata = np.asarray(imagradardata, dtype=float)

   # seed pseudorandom number generator
   seed(1)
   # create dataset
   data = realradardata 
   data = data.astype(float)
   train = np.array([],dtype=float)
   test = np.array([],dtype=float)
   curvar1 = np.zeros((1,data.size//5))
   variance = np.zeros((1,data.size))
   variance.fill(1.0)
   lag1 = 1
   lag2 = 10
   lag3 = 24
   
   '''for startindex in range(1,data.size,5):
      trainsub1 = data[startindex:startindex+3*segmentlength+1]
      variance0 = np.empty_like(train)
      variance0.fill(1.0)
  
      mod1 = HARX(trainsub1);#lags=[1, 12, 24])
      res1 = mod1.fit()
      variance0 = np.empty([3195, int((data.size-1)/5)])
      variance0.fill(1)
      trainsubcon=[]
      #x=0

      #zig zag
      
      for k in range(10):
         # if x>9:
         #     x=0
         # else:
            trainsub1 = data[startindex:startindex+3*segmentlength+1]
            print("trainlen",len(trainsub1))
            print("trainsub1",trainsub1)
            print("LoopIteration",k)
            #vol_mod = ZeroMean(res1.resid[~np.isnan(res1.resid)], volatility=MIDASHyperbolic(m=512,asym=True))
            vol_mod = ZeroMean(res1.resid[~np.isnan(res1.resid)], volatility=GARCH(p=1,o=1,q=1))
            vol_res = vol_mod.fit(disp="on")
            variance0 = vol_res.conditional_volatility ** 2.0
            print('variance0',len(variance0))
            fv = FixedVariance(variance0, unit_scale=True)
            print('fv',fv)
            mod1 = HARX(trainsub1, volatility=fv)#volatility=fv)
            print('mod1', mod1)
            #res1 = mod1.fit(disp="off")
            res1 = mod1.fit()
            print("res1",res1)
            xt=res1.resid
            #xt1=res1.forecast()
            #print('xt1',xt1)
            #yhat_temp = res1.forecast(start=k , reindex=True, x=xt)
            
            #yhat_plot=res1.plot()
            
            xt_plot=pyplot.plot(res1.conditional_volatility, label= 'Conditional Volatility', color='r')
            xt2_plot=pyplot.plot(variance0, label='Variance', color='b')
            #xt3_plot=pyplot.plot(sum(trainsub1)/len(trainsub1),color='g')
            #yhat_plot.show()
            #xt_plot.show()
            #xt2_plot.show()
            #yhat_temp = res1.forecast(horizon=segmentlength, method='simulation', simulations=1000, reindex=True)
            #yhat_temp = res1.forecast(horizon=segmentlength, start=k-x, reindex=True)
            #print("Yhat_temp",yhat_temp)
     #x+=1
             #print(res1)
      # fit model
      model_fit1 = res1
      print("model_fit",model_fit1)
      print('Segment_Length', segmentlength)
      print('res1.resid', len(res1.resid))
      # forecast the test set
      yhat1 = model_fit1.forecast(horizon=segmentlength, start=0, reindex=True, align='origin')
      print('YHAT1', yhat1)
      
      #yhat1 = model_fit1.forecast(horizon=1, reindex=True)
      # plot forecast variance
      pyplot.plot(yhat1.variance.values[-1, :])
      pyplot.show()
      curvar1 = variance0'''
      #print(variance)
   data = imagradardata 
   data = data.astype(float)
   train = np.array([],dtype=float)
   print("trainSize",train.size)
   test = np.array([],dtype=float)
   #curvar2 = np.zeros((1,4*(test-1)))
   variance = np.zeros((1,data.size))
   variance.fill(1.0)
   print('data',len(data))
   #test_segmentlength = (segmentlength/2+1).astype(int)
   for startindex in range(1,data.size,5):
      trainsub2 = data[startindex+segmentlength:-28]
      #trainsub2 = data[startindex+segmentlength:startindex+4*segmentlength+1]
      variance0 = np.empty_like(train)
      variance0.fill(1.0)
      #print('trainsub1', trainsub1)
      #print(len(trainsub1))
      print('trainsub2',trainsub2)
      print(len(trainsub2))
      print('variance2',variance0)
  
      mod2 = HARX(trainsub2.astype('float'),lags=[1,4,24])#, lags=[1, 12, 24])
      print('mod2', mod2)
      res2 = mod2.fit()
      print('res2',res2)
      variance2 = np.empty_like(trainsub2)
      variance2.fill(1)

      #zig zag
      for k in range(10):
         vol_mod = ZeroMean(res2.resid[~np.isnan(res2.resid)], volatility=GARCH(p=1,o=1,q=1), rescale=True)
         vol_res = vol_mod.fit(disp="off")
         #variance2[24:] = vol_res.conditional_volatility ** 2.0
         variance2[24:] = vol_res.conditional_volatility ** 2.0
         fv = FixedVariance(variance2, unit_scale=True)
         mod2 = HARX(trainsub2, volatility=fv, lags=[1, 4, 24])#, volatility=fv)
         res2 = mod2.fit(disp="off")
      # fit model
      model_fit2 = mod2.fit(disp="off")
      print(model_fit2)
      # forecast the test set
      yhat2 = model_fit2.forecast(horizon=segmentlength, start=24, reindex=True)
      # plot forecast variance
      pyplot.plot(yhat2.variance.values[-1, :])
      pyplot.show()
      curvar2 = variance2
      #print(variance)
   estimatedvar[i] = 0.5*(np.mean(curvar1)+np.mean(curvar2))
   print(estimatedvar[i][0])
   of.write(str(estimatedvar[i][0]) + "\n")

of.close()

#print(estimatedvar)
et=dt.datetime.now()
End_time=et.timestamp()


exec_time = End_time - start_time
print('Execution Time', exec_time)
