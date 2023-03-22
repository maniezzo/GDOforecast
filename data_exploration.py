import numpy as np, pandas as pd

if __name__ == "__main__":
   df = pd.read_csv("dfresults.csv")

   for header in df.columns:
      if('error' in header):
         print(f"algo {header} mean = {np.mean(df[header])} median = {np.median(df[header])} stdev = {np.std(df[header])}")

   pass
