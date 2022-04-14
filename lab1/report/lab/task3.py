import numpy as np
import matplotlib.pyplot as plt
import distrs

import os

def task3():
    repeats = 1000
    sizes = [20, 100]
    rows = []
    if not os.path.isdir("task3_data"):
        os.makedirs("task3_data")
    for name in distrs.distr_names:
        arr_20 = distrs.get_distr(name, sizes[0])
        arr_100 = distrs.get_distr(name, sizes[1])
        for a in arr_20:
            if a <= -50 or a >= 50:
                arr_20 = np.delete(arr_20, list(arr_20).index(a))
        for a in arr_100:
            if a <= -50 or a >= 50:
                arr_100 = np.delete(arr_100, list(arr_100).index(a))
        line_props = dict(color = "black", alpha = 0.3, linestyle = "dashdot")
        bbox_props = dict(color = "b", alpha = 0.9)
        flier_props = dict(marker = "o", markersize = 4)
        plt.boxplot((arr_20, arr_100), whiskerprops = line_props, boxprops = bbox_props, flierprops = flier_props, labels = ["n = 20", "n = 100"], vert = False)
        plt.ylabel("n")
        plt.xlabel("X")
        plt.title(name)
        plt.savefig("task3_data/" + name + ".png")
        plt.figure()

        count = 0
        for j in range(len(sizes)):
            for i in range(repeats):
                arr = distrs.get_distr(name, sizes[j])
                X = []
                X.append(np.quantile(arr, 0.25) - 1.5 * (np.quantile(arr, 0.75) - np.quantile(arr, 0.25)))
                X.append(np.quantile(arr, 0.75) + 1.5 * (np.quantile(arr, 0.75) - np.quantile(arr, 0.25)))
                for k in range(0, sizes[j]):
                    if arr[k] > X[1] or arr[k] < X[0]:
                        count += 1
            count /= repeats
            rows.append(name + " n = $" + str(sizes[j]) + "$ & $" + str(np.around(count / sizes[j], decimals=3)) + "$")
    with open("task3_data/task3.tex", "w") as f:
        f.write("\\begin{tabular}{|c|c|}\n")
        f.write("\\hline\n")
        f.write("Sample & Share of emissions \\\\\n")
        f.write("\\hline\n")
        for row in rows:
            f.write(row + "\\\\\n")
            f.write("\\hline\n")
        f.write("\\end{tabular}")

if __name__ == "__main__":
    task3()