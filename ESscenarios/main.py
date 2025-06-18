import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing as ETSStateSpace
import numpy as np

# genera i valori previsti delle serie boosted con modello con distribuzione (da confint holt winters)

np.random.seed(995)
# Load data
data = pd.read_csv('../serie_nocovid_new.csv')
series2plot = 2
numBoost = 175
boostedSeries = []
for i in range(data['idserie'].max()+1):
   series = data[data['idserie'] == i].reset_index(drop=True)
   
   # Fit state space ETS model
   model = ETSStateSpace(
       endog=series['val'].iloc[:-3],
       trend='add',
       seasonal=12
   )
   fit = model.fit()
   
   # Forecast
   forecast_steps = 3
   forecast = fit.get_forecast(steps=forecast_steps)
   
   # Get confidence intervals
   conf_int = forecast.conf_int()
   predicted_mean = forecast.predicted_mean
   
   # Assume 95% confidence interval
   ci = forecast.conf_int(alpha=0.05)  # alpha=0.05 -> 95% CI
   ci_width = ci.iloc[:, 1] - ci.iloc[:, 0]
   z_score = 1.96  # z-score for 95% CI
   
   # Calculate standard deviation of forecast error
   std_dev = ci_width / (2 * z_score)
   
   print(std_dev.head())
   if i==series2plot:
      plt.figure(figsize=(10, 6))
      plt.plot(series['val'], label='Observed')
      plt.plot(predicted_mean, label='Forecast', color='green')
      plt.fill_between(ci.index,
                       ci.iloc[:, 0],
                       ci.iloc[:, 1], color='pink', alpha=0.3,
                       label='95% Confidence Interval')
      
      # Add ±1 std dev lines
      plt.plot(predicted_mean - std_dev, linestyle='--', color='gray', label='-1 Std Dev')
      plt.plot(predicted_mean + std_dev, linestyle='--', color='gray', label='+1 Std Dev')
      
      plt.legend()
      plt.title(f"Forecast series {i}")
      plt.show()
   
   
   # Generate numBoost normally distributed values
   mean = predicted_mean.iloc[-1]
   stdev = std_dev.iloc[-1]
   vals = np.random.normal(loc=mean, scale=stdev, size=numBoost)
   boostedSeries.append(np.round(vals,0).astype(int))
   
   # Plot histogram
   if i==series2plot:
      plt.hist(vals, bins=10, edgecolor='black', alpha=0.7)
      plt.title("Simulated values")
      plt.xlabel("vals")
      plt.ylabel("Frequency")
      plt.show()
      
   np.savetxt(f"ETSboosts{numBoost}.csv",np.transpose(boostedSeries), fmt='%d', delimiter=',')
print("finito")