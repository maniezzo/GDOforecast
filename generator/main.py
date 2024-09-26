import numpy as np
import pandas as pd

if __name__ == "__main__":
   # row: server, col: client
   df = pd.read_csv('seedMatrix.csv',header=0)
   dforg = df.iloc[0:52, 0:4].T
   arr = dforg.to_numpy()
   dforg.to_csv("cost1.csv", header=False, index=False)

   # Randomly select a subset of rows and columns
   num_rows    = 4  # Number of rows to sample
   num_columns = 2  # Number of columns to sample

   # Randomly select rows
   instance_rows = df.sample(n=num_rows, random_state=42)  # Set random_state for reproducibility
   # Randomly select columns
   instance_columns = instance_rows.sample(n=num_columns, axis=1, random_state=42)

   cost = instance_columns.T

   print("Fine")