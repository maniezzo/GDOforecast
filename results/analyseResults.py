import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
import statsmodels.api as sm
from critical_difference_diagram import draw_diagram
import json

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

# check acceptability linear model
def isLinear():
   from statsmodels.formula.api import ols

   name = "dataframe_nocovid_full"
   df = pd.read_csv(f"../data/{name}.csv")

   numrows = 45
   months = df.iloc[0:45,0]
   for i in np.arange(1,53):
      ds = df.iloc[0:numrows,i]

      x = pd.Series(np.arange(len(ds)))
      # Add a column to the independent variable (x) for the intercept
      X = sm.add_constant(x)  # This adds an intercept term (a column of 1s)

      # Fit a linear regression model
      linear_model = sm.OLS(ds, X).fit()

      # Fit a quadratic model
      ds2 = ds ** 2
      quadratic_model = sm.OLS(ds2,X).fit()

      # Perform an ANOVA (Analysis of Variance) F-test between the models
      anova_results = sm.stats.anova_lm(linear_model, quadratic_model)
      #print(anova_results)

      # Check if the p-value is significant
      p_value = anova_results['Pr(>F)'][1]
      if p_value < 0.05:
          print(f"Series {i}: p={p_value} Significant lack of fit (no linear model).")
      else:
          print(f"Series {i}: p={p_value} No significant lack of fit (linear model is ok).")

def go_analysis():
   with open('config.json') as jconf:
      conf = json.load(jconf)
   print(conf)
   distrib = conf['distrib']
   model = conf['model']  # AR, YW
   nboost= conf['nboost']
   dataset  = f"res_{model}_{distrib}_{nboost}"

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
   for idObjFunc in range(3):
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
   isLinear()
   go_analysis()