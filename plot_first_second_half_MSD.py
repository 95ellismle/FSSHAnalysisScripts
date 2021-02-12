import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os


folder = "../MSD_Data"
if not os.path.isdir(folder): raise SystemExit("Can't find the folder you wanted.")

all_files = [f"{folder}/{i}" for i in os.listdir(folder) if '.csv' in i]


# Get some stats on the traj data
halfPoint = int(len(all_files) / 2)
firstHalf = all_files[:halfPoint]
secondHalf = all_files[halfPoint:]
print(f"Num half traj = {halfPoint}")

# Get average data over first half of traj
df1 = pd.read_csv(firstHalf[0])
for f in firstHalf[1:]:
    df1 += pd.read_csv(f)
sumdf1 = df1.dropna()
df1 = sumdf1 / len(firstHalf)

# Get average data over second half of traj
df2 = pd.read_csv(secondHalf[0])
for f in secondHalf[1:]:
    df2 += pd.read_csv(f)
sumdf2 = df2.dropna()
df2 = sumdf2 / len(secondHalf)

# Get total average
dfavg = (sumdf1 + sumdf2) / len(all_files)

def calc_mobility(df, start_time, end_time, alpha=1):
    chopped_df = df.loc[(df['time'] > start_time) & (df['time'] < end_time), :]
    D =  np.zeros((3,3))
    mobilities =  np.zeros((3,3))
    colors = {'x': 'tab:blue', 'y': 'tab:orange', 'z': 'tab:green'}
    ls = {'x': '-', 'y': '--', 'z': ':'}
    for i, dim1 in enumerate('xyz'):
        for j, dim2 in enumerate('xyz'):
            if (j > i): continue
            dir_ = f"{dim1}{dim2}"

            fit = np.polyfit(chopped_df['time'], chopped_df[dir_], 1)
            fit, pcov = curve_fit(lambda x, m, c: m*x + c, chopped_df['time'],
                                  chopped_df[dir_], p0=fit)
            D[i][j] = 0.5 * fit[0]
            D[j][i] = 0.5 * fit[0]
            mobilities[i][j] = D[i][j] * 0.1 / (0.0000861728 * 300)
            mobilities[j][i] = mobilities[i][j]

            if alpha == 1:
                plt.plot(df['time'], df[dir_], ls=ls[dim2], color=colors[dim1],
                         label=dir_, alpha=alpha)
            else:
                plt.plot(df['time'], df[dir_], ls=ls[dim2], color=colors[dim1],
                         alpha=alpha)

    plt.legend()
    return mobilities

u1 = calc_mobility(df1, 50, 300, alpha=0.5)
u2 = calc_mobility(df2, 50, 300, alpha=0.5)
u = calc_mobility(dfavg, 50, 300)

print(np.sort(np.linalg.eig(u1)[0]))
print(np.sort(np.linalg.eig(u2)[0]))
print(np.sort(np.linalg.eig(u)[0]))

plt.show()



