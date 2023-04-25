import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

fname = "gridplot_uni.csv"
B=0

df = pd.read_csv(fname)
# df_valid = df.loc[(df["mean"]-B*df["stdev"])>0]
df_valid = df.loc[(df["mean"])>0]
print(df_valid)

data_valid = df_valid.values

ax = plt.axes(projection="3d")
ax.scatter3D(
    df_valid["x1"],
    df_valid["x2"],
    df_valid["alpha"]
)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("alpha")
plt.show()