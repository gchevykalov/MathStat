import matplotlib.pyplot as plt
import distrs

import os

num_bins = 20

def task1():
    sizes = [10, 50, 1000]
    if not os.path.isdir("task1_images"):
        os.makedirs("task1_images")
    for name in distrs.distr_names:
        fig, ax = plt.subplots(1, 3, figsize = (12, 4))
        plt.subplots_adjust(wspace = 0.5)
        fig.suptitle(name)
        for i in range(len(sizes)):
            array = distrs.get_distr(name, sizes[i])
            n, bins, patches = ax[i].hist(array, num_bins, density = 1, edgecolor = 'black', alpha = 0.2)
            ax[i].plot(bins, distrs.get_density_func(name, bins), color = 'r', linewidth = 1)
            ax[i].set_title("n = " + str(sizes[i]))
        plt.savefig("task1_images/" + name + ".png")

if __name__ == "__main__":
    task1()