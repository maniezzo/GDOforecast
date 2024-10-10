import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
   df = pd.read_csv("../results/res_nboost_b0.csv")

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
   plt.title('Objective function values', fontsize=font_size)
   plt.ylabel('Values', fontsize=font_size)

   # Show the plot
   plt.show()
   pass