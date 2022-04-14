import numpy as np
import math
import os
import scipy.stats as stats

def get_cdf(x, mu, sigma):
    return stats.norm.cdf(x, mu, sigma)

def normal():
    sz = 100
    samples = np.random.normal(0, 1, size=sz)
    mu_c = np.mean(samples)
    sigma_c = np.std(samples)
    k = 7
    borders = np.linspace(mu_c - 3, mu_c + 3, num=(k - 1))

    p_arr = np.array([get_cdf(borders[0], mu_c, sigma_c)])
    for i in range(len(borders) - 1):
        val = get_cdf(borders[i + 1], mu_c, sigma_c) - get_cdf(borders[i], mu_c, sigma_c)
        p_arr = np.append(p_arr, val)
    p_arr = np.append(p_arr, 1 - get_cdf(borders[-1], mu_c, sigma_c))

    print(f"mu: " + str(np.around(mu_c, decimals=2)) + ", sigma: " + str(np.around(sigma_c, decimals=2)))

    n_arr = np.array([len(samples[samples <= borders[0]])])
    for i in range(len(borders) - 1):
        n_arr = np.append(n_arr, len(samples[(samples <= borders[i + 1]) & (samples >= borders[i])]))
    n_arr = np.append(n_arr, len(samples[samples >= borders[-1]]))

    res_arr = np.divide(np.multiply((n_arr - p_arr * 100), (n_arr - p_arr * 100)), p_arr * 100)

    intervals = [('$-\inf$', str(np.around(borders[0], decimals=2)))]
    for i in range(len(borders) - 1):
        intervals.append((str(np.around(borders[i], decimals=2)), str(np.around(borders[i + 1], decimals=2))))
    intervals.append((str(np.around(borders[-1], decimals=2)), '$+\inf$'))
    rows = [[i + 1, (intervals[i][0] + ', ' + intervals[i][1]),
                   "%.2f" % n_arr[i],
                   "%.4f" % p_arr[i],
                   "%.2f" % (sz * p_arr[i]),
                   "%.2f" % (n_arr[i] - sz * p_arr[i]),
                   "%.2f" % ((n_arr[i] - sz * p_arr[i]) ** 2 / (sz * p_arr[i]))] for i in range(k)]
    rows.insert(0, ['\hline i', 'Границы $\Delta_i$', '$n_i$', '$p_i$', '$np_i$', '$n_i - np_i$', '$\\frac{(n_i - np_i)^2}{np_i}$'])
    rows.append(['$\sum$', '$-$', "%.2f" % np.sum(n_arr), "%.4f" % np.sum(p_arr), "%.2f" % (sz * np.sum(p_arr)),
        "%.2f" % (np.sum(n_arr) - sz * np.sum(p_arr)), "%.2f" % np.sum(res_arr)])
    with open("task3_data/task3_normal.tex", "w") as f:
        f.write("\\begin{tabular}{|c|c|c|c|c|c|c|}\n")
        f.write("\\hline\n")
        for row in rows:
            f.write(" & ".join([str(i) for i in row]) + "\\\\\n")
            f.write("\\hline\n")
        f.write("\\end{tabular}")

def laplace():
    sz = 20
    samples = np.random.laplace(0, 1 / math.sqrt(2), size=sz)
    mu_c = np.mean(samples)
    sigma_c = np.std(samples)
    k = 7
    borders = np.linspace(mu_c - 3, mu_c + 3, num=(k - 1))

    p_arr = np.array([get_cdf(borders[0], mu_c, sigma_c)])
    for i in range(len(borders) - 1):
        val = get_cdf(borders[i + 1], mu_c, sigma_c) - get_cdf(borders[i], mu_c, sigma_c)
        p_arr = np.append(p_arr, val)
    p_arr = np.append(p_arr, 1 - get_cdf(borders[-1], mu_c, sigma_c))

    n_arr = np.array([len(samples[samples <= borders[0]])])
    for i in range(len(borders) - 1):
        n_arr = np.append(n_arr, len(samples[(samples <= borders[i + 1]) & (samples >= borders[i])]))
    n_arr = np.append(n_arr, len(samples[samples >= borders[-1]]))

    res_arr = np.divide(np.multiply((n_arr - p_arr * sz), (n_arr - p_arr * sz)), p_arr * sz)

    intervals = [('$-\inf$', str(np.around(borders[0], decimals=2)))]
    for i in range(len(borders) - 1):
        intervals.append((str(np.around(borders[i], decimals=2)), str(np.around(borders[i + 1], decimals=2))))
    intervals.append((str(np.around(borders[-1], decimals=2)), '$+\inf$'))
    rows = [[i + 1, (intervals[i][0] + ', ' + intervals[i][1]),
                   "%.2f" % n_arr[i],
                   "%.4f" % p_arr[i],
                   "%.2f" % (sz * p_arr[i]),
                   "%.2f" % (n_arr[i] - sz * p_arr[i]),
                   "%.2f" % ((n_arr[i] - sz * p_arr[i]) ** 2 / (sz * p_arr[i]))] for i in range(k)]
    rows.insert(0, ['\hline i', 'Границы $\Delta_i$', '$n_i$', '$p_i$', '$np_i$', '$n_i - np_i$', '$\\frac{(n_i - np_i)^2}{np_i}$'])
    rows.append(['$\sum$', '$-$', "%.2f" % np.sum(n_arr), "%.4f" % np.sum(p_arr), "%.2f" % (sz * np.sum(p_arr)),
        "%.2f" % (np.sum(n_arr) - sz * np.sum(p_arr)), "%.2f" % np.sum(res_arr)])
    with open("task3_data/task3_laplace.tex", "w") as f:
        f.write("\\begin{tabular}{|c|c|c|c|c|c|c|}\n")
        f.write("\\hline\n")
        for row in rows:
            f.write(" & ".join([str(i) for i in row]) + "\\\\\n")
            f.write("\\hline\n")
        f.write("\\end{tabular}")

def task3():
    if not os.path.exists("task3_data"):
        os.mkdir("task3_data")
    normal()
    laplace()

if __name__ == "__main__":
    task3()