import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

from gp_support import *

from data_plotter import *

from scipy.sparse import lil_matrix
import scipy

"""
pythonw pyalad/test_plots.py
"""


def plot_beta_explore_exploit(a=1., b=1., budget=60):

    pdfpath_contours = "./temp/beta_plots.pdf"

    # mean, var, skew, kurt = beta.stats(a, b, moments='mvsk')

    xx, yy = np.meshgrid(np.linspace(0, budget, budget+1), np.linspace(0, budget, budget+1))

    beta_means = np.zeros(shape=(budget+1, budget+1), dtype=float)
    for s in range(budget):
        for f in range(budget - s):
            beta_means[s, f] = beta.stats(a + s, (b + f) ** (1. - (s+f)*1./budget), moments='m')

    dp = DataPlotter(pdfpath=pdfpath_contours, rows=1, cols=1)
    pl = dp.get_next_plot()
    CS = pl.contourf(xx, yy, beta_means, 20)
    plt.xlabel('# failures')
    plt.ylabel('# successes')
    cbar = plt.colorbar(CS)
    cbar.ax.set_ylabel('explore probability')
    dp.close()
    #x = np.linspace(beta.ppf(0.01, a, b), beta.ppf(0.99, a, b), 100)
    #ax.plot(x, beta.pdf(x, a, b), 'r-', lw=5, alpha=0.6, label='beta pdf')


def test_bernoulli_explore_exploit():
    a = 1.
    b = 1.
    s = 10
    f = 2
    budget = 60

    t = 0.
    mean = 0.
    for i in np.arange(1000):
        r, mean = bernoulli_explore_exploit_sample(a=a, b=b, s=s, f=f, budget=budget, mean=True)
        t += r
    print "Sampled mean: %f, true mean: %f" % (t/1000., mean)


def test_thompson():
    a = 1.
    b = 1.

    reward_history = np.zeros(shape=(2, 2), dtype=float)
    reward_history[0, :] = [ 0, 0]
    reward_history[1, :] = [ 0, 0]

    t = 0.
    np.random.seed(43)
    for i in np.arange(20):
        action, samples, mean = thompson_sample(a=a, b=b, reward_history=reward_history, mean=True)
        print "action: %d" % action
        explore = True if action == 1 else False
        rewarded = i % 3 == 0
        if rewarded:
            r = np.array([1., 0.], dtype=float)  # increment success counts
        else:
            r = np.array([0., 1.], dtype=float)  # increment failure counts
        reward_history[1 if explore else 0, :] += r
        print("rewarded: %s, explore: %s, samples:\n%s\nupdated reward matrix:\n%s" %
                     (rewarded, explore, str(samples), str(reward_history)))


def test_get_gp_train_test():
    all_indexes = np.array([1,2,3,4,5,6,7])
    queried_indexes = np.array([3,2,4])
    train, test = get_gp_train_test(all_indexes, queried_indexes, n_train=2, n_test=4)
    print("train:\n%s\ntest:\n%s" % (str(list(train)), str(list(test))))


def test_sparse_variance():
    x_tmp = lil_matrix((2, 5), dtype=float)
    x_tmp[0, 2] = 1
    x_tmp[0, 3] = 2
    x_tmp[1, 4] = 1
    x = x_tmp.tocsr()
    print("x:\n%s" % str(x))

    w = scipy.stats.uniform.rvs(0, 1, 5)
    print("w:\n%s" % str(w))

    print "row x: %s, w: %s" % (str(x[0, :].shape), str(w.shape))

    indxs = x[0, :].nonzero()[1]
    xw = np.array(x[0, indxs].multiply(w[indxs]))
    print("xw:\n%s\nshape: %s" % (str(xw), str(xw.shape)))
    xw_mean = xw.mean(axis=1)
    print("xw_mean:%s" % str(xw_mean))
    xw_sq = xw ** 2
    print("xw_sq:\n%s" % str(xw_sq))

    xw_var = xw_sq.mean(axis=1)[0] - xw_mean[0] ** 2

    print("xw_var:\n%f" % xw_var)


if __name__ == '__main__':
    # plot_beta_explore_exploit(a=1., b=1., budget=60)
    test_thompson()
    # test_get_gp_train_test()
    # test_sparse_variance()
