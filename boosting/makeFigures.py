import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# boxplot for increasing nboost
def boxplotb0(df):
   col75  = df[df['num.boost'] == 75]["objval"].reset_index(drop=True)
   col125 = df[df['num.boost'] == 125]["objval"].reset_index(drop=True)
   col175 = df[df['num.boost'] == 175]["objval"].reset_index(drop=True)
   dfData = pd.DataFrame({'boost75': col75, 'boost125': col125, 'boost175':col175})

   fig, ax1 = plt.subplots(figsize=(10, 6))
   labels = ['boost75','boost125','boost175']
   box = ax1.boxplot([dfData["boost75"].dropna(),dfData["boost125"].dropna(),dfData["boost175"]],
                       patch_artist=True, widths=0.25,tick_labels=labels)
   # Customize the box (rectangle fill)
   for patch in box['boxes']:
      patch.set_facecolor('lightblue')  # Set fill color
      patch.set_edgecolor('black')  # Set box edge color
      patch.set_linewidth(2)  # Set the edge line width

   # Customize the median line
   for median in box['medians']:
      median.set_color('black')  # Set median line color
      median.set_linewidth(2.5)  # Set median line thickness
      median.set_linestyle('--')  # Set median line style

   # Calculate quartiles and median
   quartiles75 = dfData['boost75'].quantile([0.25, 0.5, 0.75])
   q1_75 = int(quartiles75[0.25])
   median75 = int(quartiles75[0.5])
   q3_75 = int(quartiles75[0.75])
   min_val75 = int(dfData['boost75'].min())
   max_val75 = int(dfData['boost75'].max())

   quartiles125 = dfData['boost125'].quantile([0.25, 0.5, 0.75])
   q1_125     = int(quartiles125[0.25])
   median125  = int(quartiles125[0.5])
   q3_125     = int(quartiles125[0.75])
   min_val125 = int(dfData['boost125'].min())
   max_val125 = int(dfData['boost125'].max())

   quartiles175 = dfData['boost175'].quantile([0.25, 0.5, 0.75])
   q1_175     = int(quartiles175[0.25])
   median175  = int(quartiles175[0.5])
   q3_175     = int(quartiles175[0.75])
   min_val175 = int(dfData['boost175'].min())
   max_val175 = int(dfData['boost175'].max())

   ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.3)

   # Annotating the quartiles and median
   font_size = 12
   plt.text(1.15, min_val75,f'{min_val75}', verticalalignment='center', fontsize=font_size)
   plt.text(1.15, q1_75,    f'{q1_75}', verticalalignment='center', fontsize=font_size)
   plt.text(1.15, median75, f'{median75}', verticalalignment='center', fontsize=font_size)
   plt.text(1.15, q3_75,   f'{q3_75}', verticalalignment='center', fontsize=font_size)
   plt.text(1.15, max_val75,f'{max_val75}', verticalalignment='center', fontsize=font_size)

   plt.text(2.15, min_val125, f'{min_val125}', verticalalignment='center', fontsize=font_size)
   plt.text(2.15, q1_125, f'{q1_125}', verticalalignment='center', fontsize=font_size)
   plt.text(2.15, median125, f'{median125}', verticalalignment='center', fontsize=font_size)
   plt.text(2.15, q3_125, f'{q3_125}', verticalalignment='center', fontsize=font_size)
   plt.text(2.15, max_val125, f'{max_val125}', verticalalignment='center', fontsize=font_size)

   plt.text(3.15, min_val175, f'{min_val175}', verticalalignment='center', fontsize=font_size)
   plt.text(3.15, q1_175, f'{q1_175}', verticalalignment='center', fontsize=font_size)
   plt.text(3.15, median175, f'{median175}', verticalalignment='center', fontsize=font_size)
   plt.text(3.15, q3_175, f'{q3_175}', verticalalignment='center', fontsize=font_size)
   plt.text(3.15, max_val175, f'{max_val175}', verticalalignment='center', fontsize=font_size)

   # Add labels and title
   plt.title('Objective function values, enlarging boostset', fontsize=font_size)
   plt.ylabel('Values', fontsize=font_size)

   # Show the plot
   plt.show()
   return

# boxplot for increasing b values
def boxplotBmult(df):
   col0 = df[df['nmult'] == 0]["objval"].reset_index(drop=True)
   col4 = df[df['nmult'] == 4]["objval"].reset_index(drop=True)
   col8 = df[df['nmult'] == 8]["objval"].reset_index(drop=True)
   col12 = df[df['nmult'] == 12]["objval"].reset_index(drop=True)
   dfData = pd.DataFrame({'nb0': col0, 'nb4': col4, 'nb8':col8, 'nb12':col12})

   fig, ax1 = plt.subplots(figsize=(10, 6))
   labels = ['nb0','nb4','nb8','nb12']
   box = ax1.boxplot([dfData["nb0"].dropna(),dfData["nb4"].dropna(),dfData["nb8"],dfData["nb12"]],
                       patch_artist=True, widths=0.25,tick_labels=labels)
   # Customize the box (rectangle fill)
   for patch in box['boxes']:
      patch.set_facecolor('lightblue')  # Set fill color
      patch.set_edgecolor('black')  # Set box edge color
      patch.set_linewidth(2)  # Set the edge line width

   # Customize the median line
   for median in box['medians']:
      median.set_color('black')  # Set median line color
      median.set_linewidth(2.5)  # Set median line thickness
      median.set_linestyle('--')  # Set median line style

   # Calculate quartiles and median
   quartiles0 = dfData['nb0'].quantile([0.25, 0.5, 0.75])
   q1_0     = int(quartiles0[0.25])
   median0  = int(quartiles0[0.5])
   q3_0     = int(quartiles0[0.75])
   min_val0 = int(dfData['nb0'].min())
   max_val0 = int(dfData['nb0'].max())

   quartiles4 = dfData['nb4'].quantile([0.25, 0.5, 0.75])
   q1_4     = int(quartiles4[0.25])
   median4  = int(quartiles4[0.5])
   q3_4     = int(quartiles4[0.75])
   min_val4 = int(dfData['nb4'].min())
   max_val4 = int(dfData['nb4'].max())

   quartiles8 = dfData['nb8'].quantile([0.25, 0.5, 0.75])
   q1_8     = int(quartiles8[0.25])
   median8  = int(quartiles8[0.5])
   q3_8     = int(quartiles8[0.75])
   min_val8 = int(dfData['nb8'].min())
   max_val8 = int(dfData['nb8'].max())

   quartiles12 = dfData['nb12'].quantile([0.25, 0.5, 0.75])
   q1_12     = int(quartiles12[0.25])
   median12  = int(quartiles12[0.5])
   q3_12     = int(quartiles12[0.75])
   min_val12 = int(dfData['nb12'].min())
   max_val12 = int(dfData['nb12'].max())

   ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.3)

   # Annotating the quartiles and median
   font_size = 12
   plt.text(1.15, min_val0,f'{min_val0}', verticalalignment='center', fontsize=font_size)
   plt.text(1.15, q1_0,    f'{q1_0}',     verticalalignment='center', fontsize=font_size)
   plt.text(1.15, median0, f'{median0}',  verticalalignment='center', fontsize=font_size)
   plt.text(1.15, q3_0,    f'{q3_0}',     verticalalignment='center', fontsize=font_size)
   plt.text(1.15, max_val0,f'{max_val0}', verticalalignment='center', fontsize=font_size)

   plt.text(2.15, min_val4,f'{min_val4}', verticalalignment='center', fontsize=font_size)
   plt.text(2.15, q1_4,    f'{q1_4}',     verticalalignment='center', fontsize=font_size)
   plt.text(2.15, median4, f'{median4}',  verticalalignment='center', fontsize=font_size)
   plt.text(2.15, q3_4,    f'{q3_4}',     verticalalignment='center', fontsize=font_size)
   plt.text(2.15, max_val4,f'{max_val4}', verticalalignment='center', fontsize=font_size)

   plt.text(3.15, min_val8,f'{min_val8}', verticalalignment='center', fontsize=font_size)
   plt.text(3.15, q1_8,    f'{q1_8}',     verticalalignment='center', fontsize=font_size)
   plt.text(3.15, median8, f'{median8}',  verticalalignment='center', fontsize=font_size)
   plt.text(3.15, q3_8,    f'{q3_8}',     verticalalignment='center', fontsize=font_size)
   plt.text(3.15, max_val8,f'{max_val8}', verticalalignment='center', fontsize=font_size)

   plt.text(4.15, min_val12,f'{min_val12}', verticalalignment='center', fontsize=font_size)
   plt.text(4.15, q1_12,    f'{q1_12}',     verticalalignment='center', fontsize=font_size)
   plt.text(4.15, median12, f'{median12}',  verticalalignment='center', fontsize=font_size)
   plt.text(4.15, q3_12,    f'{q3_12}',     verticalalignment='center', fontsize=font_size)
   plt.text(4.15, max_val12,f'{max_val12}', verticalalignment='center', fontsize=font_size)

   # Add labels and title
   plt.title('Objective function values, increasing b values', fontsize=font_size)
   plt.ylabel('Values', fontsize=font_size)

   # Show the plot
   plt.show()
   return

if __name__ == "__main__":
   df = pd.read_csv("../results/res_nboost_b0.csv")
   boxplotb0(df)
   df = pd.read_csv("../results/res_nboost_bmult.csv")
   boxplotBmult(df)
   pass