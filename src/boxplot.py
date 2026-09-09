from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)

# Features + target as a single DataFrame
df = housing.frame

# Quick check
print(df.head())
print(df.shape)

import matplotlib.pyplot as plt

# Make boxplot
df["MedHouseVal"].plot(kind="box")

# Save boxplot
plt.savefig("figs/boxplot.png")
plt.show()