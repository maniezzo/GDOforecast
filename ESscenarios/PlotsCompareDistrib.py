# genera plot comparativi distribuzioni beso e ETS
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

# Sample data for ds1 and ds2
df1 = pd.read_csv('BESOboosts_75.csv')
df2 = pd.read_csv('ETSboosts_75.csv')

# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ds1 = df1.iloc[15,:]
ds2 = df2.iloc[15,:]

# Plot histogram for ds1 on the left subplot
ax1.hist(ds1, bins=30, color='skyblue', edgecolor='black')
ax1.set_title('BESO boostest')
ax1.set_xlabel('Value')
ax1.set_ylabel('Frequency')

# Plot histogram for ds2 on the right subplot
ax2.hist(ds2, bins=30, color='salmon', edgecolor='black')
ax2.set_title('ETS boostest')
ax2.set_xlabel('Value')
ax2.set_ylabel('Frequency')

# Display the plot
plt.tight_layout()
plt.savefig('compDistr.eps', format='eps')
plt.show()
