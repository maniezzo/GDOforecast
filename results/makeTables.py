import numpy as np, pandas as pd

def run_table():
   df = pd.read_excel("../stochastic/results_det.xlsx")
   dfres = df.groupby(['numcli','numser','nmult','timestep'], as_index=False).mean()
   return
