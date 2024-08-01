import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from critical_difference_diagram import draw_diagram

# Accuracy metrics
def forecast_accuracy(model,forecast, actual):
   bias = np.sum(forecast-actual)              # BIAS
   mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
   me   = np.mean(forecast - actual)           # ME (bias)
   mae  = np.mean(np.abs(forecast - actual))   # MAE
   mpe  = np.mean((forecast - actual)/actual)  # MPE
   rmse = np.mean((forecast - actual)**2)**.5  # RMSE
   corr = np.corrcoef(forecast, actual)[0,1]   # correlation coeff
   acf1 = acf(forecast-actual)[1]              # ACF1
   return({'model':model, 'bias':bias, 'mape':mape, 'me':me, 'mae': mae, 'mpe': mpe, 'rmse':rmse,
           'acf1':acf1, 'corr':corr})

def go_analysis():
   dataset  = "res_AR_175"
   fileName = f"../boost_forecast/{dataset}.csv"
   df = pd.read_csv(fileName)
   trueval = df.loc[:,'true']
   results = []
   for colname in df.columns[3:]:
      modelval = df.loc[:,colname]
      results.append( forecast_accuracy(colname,modelval.values, trueval.values) )
   df_results = pd.DataFrame(results)
   df_results.to_csv(f"../results/{dataset}_analysis.csv", index=False)

   # critical difference diagram
   idObjFunc = 2
   if(idObjFunc == 0):   ofName = 'MAE'
   elif(idObjFunc == 1): ofName = 'MSE'
   elif(idObjFunc == 2): ofName = 'BIAS'
   df_perf = pd.DataFrame(columns=['algorithm','instance',f'{ofName}'])
   for j in [3,4,7,8,9,10,11,12,13,14]:
      colname = df.columns[j]
      for i in range(df.shape[0]):
         if (idObjFunc == 0):   ofVal = abs(df.iloc[i,j]-df.iloc[i,2])
         elif (idObjFunc == 1): ofVal = (df.iloc[i,j]-df.iloc[i,2])**2
         elif (idObjFunc == 2): ofVal = (df.iloc[i,j]-df.iloc[i,2])
         df_perf.loc[len(df_perf.index)] = [colname,f"boost{i}",ofVal]

   draw_diagram(df_perf=df_perf, dfname=dataset)
   return

if "__main__" == __name__:
   go_analysis()