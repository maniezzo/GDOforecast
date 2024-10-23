import numpy as np, pandas as pd

def run_table():
   # tabella q deterministici
   df = pd.read_csv("../stochastic/results_qdet.csv")
   dfup = df[df['direction'] == 'up']
   dfdown = df[df['direction'] == 'down']
   # tabella stocastici
   df = pd.read_excel("../stochastic/results_det.xlsx")
   dfres = df.groupby(['numcli','numser','nmult','timestep'], as_index=False).mean()
   return
