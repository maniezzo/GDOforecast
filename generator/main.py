import numpy as np
import pandas as pd

if __name__ == "__main__":
   df = pd.read_csv('seedMatrix.csv',header=0)
   df = df.iloc[:, 1:]
   # Iterate over upper triangle of the matrix
   for i in range(df.shape[0]):
      for j in range(i+1,df.shape[1]):
         if (df.iloc[i,j]==999999 and i!=j): df.iloc[i,j] = df.iloc[j,i]
         if abs(df.iloc[i,j] - df.iloc[j,i]) > 150:
            print(f"i {i} j {j} {df.iloc[j, i]} => {df.iloc[i, j]}")
            df.iloc[j,i]=df.iloc[i,j]
   df.to_csv('seedMatrix.csv')
   dforg = df.iloc[0:4, 305:]
   dforg.to_csv("cost1.csv")
   print("Fine")