import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns
import distrs

import os

def task4_2():
    sizes = [20, 60, 100]
    if not os.path.isdir("task4_data"):
        os.makedirs("task4_data")
    for name in distrs.distr_names:
        if name == "Poisson":
            interval = np.arange(6, 15, 1)
        else:
            interval = np.arange(-4, 4, 0.01)
        for j in range(len(sizes)):
            arr = distrs.get_distr(name, sizes[j])
            for a in arr:
                if name == "Poisson" and (a < 6 or a > 14):
                    arr = np.delete(arr, list(arr).index(a))
                elif name != "Poisson" and (a < -4 or a > 4):
                    arr = np.delete(arr, list(arr).index(a))

            title = ["h = 1/2 * h_n", "h = h_n", "h = 2 * h_n"]
            bw = [0.5, 1, 2]
            fig, ax = plt.subplots(1, 3, figsize = (12, 4))
            plt.subplots_adjust(wspace = 0.5)
            for k in range(len(bw)):
                kde = gaussian_kde(arr, bw_method = 'silverman')
                h_n = kde.factor
                fig.suptitle(name + ", n = " + str(sizes[j]))
                ax[k].plot(interval, distrs.get_density_func(name, interval), color='blue', alpha=0.5, label='density')
                ax[k].set_title(title[k])
                sns.kdeplot(arr, ax = ax[k], bw = h_n * bw[k], label = 'kde', color = 'black')
                ax[k].set_xlabel('x')
                ax[k].set_ylabel('f(x)')
                ax[k].set_ylim([0, 1])
                if name == 'Poisson':
                    ax[k].set_xlim([6, 14])
                else:
                    ax[k].set_xlim([-4, 4])
                ax[k].legend()
            plt.savefig("task4_data/" + name + "_kernel_n" + str(sizes[j]) + ".png")

if __name__ == "__main__":
    task4_2()