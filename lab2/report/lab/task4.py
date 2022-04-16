import numpy as np
from scipy.stats import chi2, t, norm, moment
import os
import matplotlib.pyplot as plt

gamma = 0.95
alpha = 1 - gamma

def get_student_mo(samples, alpha):
    med = np.mean(samples)
    n = len(samples)
    s = np.std(samples)
    t_a = t.ppf(1 - alpha / 2, n - 1)
    q_1 = med - s * t_a / np.sqrt(n - 1)
    q_2 = med + s * t_a / np.sqrt(n - 1)
    return q_1, q_2

def get_chi_sigma(samples, alpha):
    n = len(samples)
    s = np.std(samples)
    q_1 =  s * np.sqrt(n) / np.sqrt(chi2.ppf(1 - alpha / 2, n - 1))
    q_2 = s * np.sqrt(n) / np.sqrt(chi2.ppf(alpha / 2, n - 1))
    return q_1, q_2

def get_as_mo(samples, alpha):
    med = np.mean(samples)
    n = len(samples)
    s = np.std(samples)
    u = norm.ppf(1 - alpha / 2)
    q_1 = med - s * u / np.sqrt(n)
    q_2 = med + s * u / np.sqrt(n)
    return q_1, q_2

def get_as_sigma(samples, alpha):
    n = len(samples)
    s = np.std(samples)
    u = norm.ppf(1 - alpha / 2)
    m4 = moment(samples, 4)
    e = m4 / (s * s * s * s)
    U = u * np.sqrt((e + 2) / n)
    q_1 = s / np.sqrt(1 + U)
    q_2 = s / np.sqrt(1 - U)
    return q_1, q_2

def task4():
    if not os.path.isdir("task4_data"):
        os.makedirs("task4_data")

    samples20 = np.random.normal(0, 1, size=20)
    samples100 = np.random.normal(0, 1, size=100)

    fig, ax = plt.subplots(1, 2)
    plt.subplots_adjust(wspace = 0.5)
    ax[0].set_ylim([0,1])
    ax[0].hist(samples20, 10, density = 1, edgecolor = 'black')
    ax[0].set_title('N(0,1) hist, n = 20')
    ax[1].set_ylim([0,1])
    ax[1].hist(samples100, 10, density = 1, edgecolor = 'black')
    ax[1].set_title('N(0,1) hist, n = 100')
    plt.savefig('task4_data/hist.png')

    fig, ax = plt.subplots(2, 2)
    plt.subplots_adjust(wspace = 0.5, hspace = 1)

    student_20 = get_student_mo(samples20, alpha)
    student_100 = get_student_mo(samples100, alpha)

    chi_20 = get_chi_sigma(samples20, alpha)
    chi_100 = get_chi_sigma(samples100, alpha)

    ax[0][0].plot([student_20[0], student_20[1]], [0.3, 0.3], color='r', marker = '.', linewidth = 1, label = 'm interval, n = 20')
    ax[0][0].plot([student_100[0], student_100[1]], [0.6, 0.6], color='blue', marker = '.', linewidth = 1, label = 'm interval, n = 100')
    ax[0][0].set_ylim([0,1])
    ax[0][0].set_title('Classic approach')
    ax[0][0].legend()

    ax[0][1].plot([chi_20[0], chi_20[1]], [0.3, 0.3], color='r', marker = '.', linewidth = 1, label = 'sigma interval, n = 20')
    ax[0][1].plot([chi_100[0], chi_100[1]], [0.6, 0.6], color='blue', marker = '.', linewidth = 1, label = 'sigma interval, n = 100')
    ax[0][1].set_ylim([0,1])
    ax[0][1].set_title('Classic approach')
    ax[0][1].legend()

    print(f"Классический подход:\n"
        f"n = 20 \n"
        f"\t m: " + str(student_20) + " \t sigma: " + str(chi_20) + "\n"
        f"n = 100 \n"
        f"\t m: " + str(student_100) + " \t sigma: " + str(chi_100) + "\n")

    as_mo_20 = get_as_mo(samples20, alpha)
    as_mo_100 = get_as_mo(samples100, alpha)

    as_d_20 = get_as_sigma(samples20, alpha)
    as_d_100 = get_as_sigma(samples100, alpha)

    ax[1][0].plot([as_mo_20[0], as_mo_20[1]], [0.3, 0.3], color='r', marker = '.', linewidth = 1, label = 'm interval, n = 20')
    ax[1][0].plot([as_mo_100[0], as_mo_100[1]], [0.6, 0.6], color='blue', marker = '.', linewidth = 1, label = 'm interval, n = 100')
    ax[1][0].set_ylim([0,1])
    ax[1][0].set_title('Asymptotic approach')
    ax[1][0].legend()

    ax[1][1].plot([as_d_20[0], as_d_20[1]], [0.3, 0.3], color='r', marker = '.', linewidth = 1, label = 'sigma interval, n = 20')
    ax[1][1].plot([as_d_100[0], as_d_100[1]], [0.6, 0.6], color='blue', marker = '.', linewidth = 1, label = 'sigma interval, n = 100')
    ax[1][1].set_ylim([0,1])
    ax[1][1].set_title('Asymptotic approach')
    ax[1][1].legend()

    plt.savefig('task4_data/intervals.png')

    print(f"Асимптотический подход:\n"
        f"n = 20 \n"
        f"\t m: " + str(as_mo_20) + " \t sigma: " + str(as_d_20) + "\n"
        f"n = 100 \n"
        f"\t m: " + str(as_mo_100) + " \t sigma: " + str(as_d_100) + "\n")

if __name__ == "__main__":
    task4()