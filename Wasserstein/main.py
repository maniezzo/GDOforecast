# Wasserstein DRO for 2-echelon problem
import pandas as pd, numpy as np
import scenarios
import wasserstein as wass

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
   costs = pd.read_csv("costs.csv").values
   cap   = pd.read_csv("capacities.csv",header=None).values.flatten()
   df, scenarios = scenarios.go_scenarios() # genera gli scenari
   req = df.iloc[-1,:]
   D, scale = wass.wasserstein_scenario_distance_matrix(scenarios) # normalizza i valori degli scenari
   D0 = wass.characteristic_distance(D) # experimental radius of wasserstein ball
   delta_max = wass.wasserstein_radius_diagnostics(D,D0)
   x = wass.go_wasserstein(scenarios,D,D0,costs,cap,delta_max) # esperimento
   wass.checksol(x,cap,req)
   print("FINE")
