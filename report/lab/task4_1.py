import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
import distrs

import os

def task4_1():
    sizes = [20, 60, 100]
    if not os.path.isdir("task4_data"):
        os.makedirs("task4_data")
    for name in distrs.distr_names:
        if name == "Poisson":
            interval = np.arange(6, 15, 1)
        else:
            interval = np.arange(-4, 4, 0.01)
        fig, ax = plt.subplots(1, 3, figsize = (12, 4))
        plt.subplots_adjust(wspace = 0.5)
        fig.suptitle(name)
        for j in range(len(sizes)):
            arr = distrs.get_distr(name, sizes[j])
            for a in arr:
                if name == "Poisson" and (a < 6 or a > 14):
                    arr = np.delete(arr, list(arr).index(a))
                elif name != "Poisson" and (a < -4 or a > 4):
                    arr = np.delete(arr, list(arr).index(a))
            ax[j].set_title("n = " + str(sizes[j]))
            if name == "Poisson":
                ax[j].step(interval, [distrs.get_distr_func(name, x) for x in interval], color = 'blue', linewidth = 0.8)
            else:
                ax[j].plot(interval, [distrs.get_distr_func(name, x) for x in interval], color = 'blue', linewidth = 0.8)
            if name == "Poisson":
                arr_ex = np.linspace(6, 14)
            else:
                arr_ex = np.linspace(-4, 4)
            ecdf = ECDF(arr)
            y = ecdf(arr_ex)
            ax[j].step(arr_ex, y, color = 'black')
            plt.savefig("task4_data/" + name + "_emperic_f.png")

if __name__ == "__main__":
    task4_1()