import os
import numpy as np
import logging
from scipy.stats import beta
from scipy.stats import binom
from scipy.stats import bernoulli
from app_globals import *
from r_support import matrix, cbind

"""
Support for Gaussian Processes
"""


# Define the kernel
def kernel(a, b, length_scale=1.):
    """ GP squared exponential kernel """
    n = a.shape[0]
    m = b.shape[0]
    K = np.zeros(shape=(n, m), dtype=float)
    for i in np.arange(n):
        for j in np.arange(m):
            dif = a[i, :] - b[j, :]
            sqdist = dif * dif.T
            if sqdist.shape[0] != 1:
                raise ArithmeticError("dist is not scalar. has shape %s" % str(sqdist.shape))
            K[i, j] = np.exp(-.5 * (1./length_scale) * sqdist[0, 0])
    return K


class SetList(list):
    """ A list class with support for rudimentary set operations
    This is a convenient class when set operations are required while
    preserving data ordering
    """
    def __init__(self, args):
        super(SetList, self).__init__(args)
    def __sub__(self, other):
        return self.__class__([item for item in self if item not in other])


def get_gp_train_test(all_indexes, queried_indexes, n_train, n_test):
    tmp = np.array(SetList(all_indexes) - SetList(queried_indexes))
    np.random.shuffle(tmp)
    n_reg_train = max(0, n_train - len(queried_indexes))
    if n_reg_train > 0:
        train = tmp[np.arange(n_reg_train)]
    else:
        train = np.array([], dtype=int)
    n_queried = n_train - n_reg_train
    if n_queried > 0:
        # qi = np.array(queried_indexes)
        # np.random.shuffle(qi)
        # train = append(train, qi[np.arange(n_queried)])
        train = append(train, queried_indexes)  # all queried points must be included
    test = tmp[np.arange(n_reg_train, n_reg_train+n_test)]
    return train, test


def get_closest_indexes(inst, test_set, num=1, dest_set=None):
    n = test_set.shape[0]
    dists = np.zeros(n)
    for i in np.arange(n):
        ts = test_set[i, :]
        if ts.shape[0] > 1:
            # dense matrix
            ts = matrix(ts, nrow=1)
            diff = inst - ts
            dist = np.sum(diff**2)
        else:
            # sparse matrix
            diff = inst - ts
            tmp = diff * diff.T
            if tmp.shape[0] != 1:
                raise ValueError("dot product is %s" % str(tmp.shape))
            dist = tmp[0, 0]
        dists[i] = dist
    ordered = np.argsort(dists)[np.arange(num)]
    if False:
        logger.debug("last ts:\n%s" % str(ts))
        logger.debug("last diff:\n%s" % str(diff))
        logger.debug("ordered indexes: %s" % str(list(ordered)))
        logger.debug("dists: %s" % str(list(dists[ordered])))
        # logger.debug("dists: %s" % str(list(dists)))
        logger.debug("inst:\n%s" % str(inst))
        logger.debug("points:\n%s" % str(test_set[ordered, :]))
        ts = test_set[ordered[1], :]
        ts = matrix(ts, nrow=1)
        logger.debug("dist 2:\n%s" % str(np.sum((inst - ts)**2)))
    if dest_set is not None:
        for indx in ordered:
            dest_set.add(indx)
    return ordered


def get_linear_score_variance(x, w):
    indxs = x.nonzero()[1]  # column indexes
    x_ = x[0, indxs].todense()
    xw = x_.reshape(-1, 1) * w[indxs]
    # logger.debug("xw:\n%s" % str(list(xw)))
    #xw = np.array(x[0, indxs].multiply(w[indxs]))
    #xw_mean = xw.mean(axis=1)[0]
    #xw_sq = xw ** 2
    #var = xw_sq.mean(axis=1)[0] - xw_mean ** 2
    var = np.var(xw)
    return var


def get_score_variances(x, w, ordered_indexes, queried_indexes, n_test,
                        test_indexes=None,
                        eval_set=None, n_closest=9):
    if test_indexes is None:
        top_ranked_indexes = ordered_indexes[np.arange(len(queried_indexes) + n_test)]
        tmp = np.array(SetList(top_ranked_indexes) - SetList(queried_indexes))
        test = tmp[np.arange(n_test)]
        # logger.debug("test:\n%s" % str(list(test)))
    else:
        test = test_indexes
        n_test = len(test)

    tm = Timer()
    vars = np.zeros(len(test))
    for i, idx in enumerate(test):
        vars[i] = get_linear_score_variance(x[idx, :], w)
    logger.debug(tm.message("Time for score variance computation on test set:"))

    v_eval = None
    if eval_set is not None:
        tm = Timer()
        v_eval = np.zeros(eval_set.shape[0], dtype=float)
        closest_indexes = set()  # all indexes from test_set that are closest to any unlabeled instances
        for i in range(n_test):
            test_index = test[i]
            get_closest_indexes(x[test_index, :], eval_set, num=n_closest, dest_set=closest_indexes)
        logger.debug("# Closest: %d" % len(closest_indexes))
        for i, idx in enumerate(closest_indexes):
            v_eval[idx] = get_linear_score_variance(eval_set[idx, :], w)
        logger.debug(tm.message("Time for score variance computation on eval set:"))

    return vars, test, v_eval


def get_gp_predictions(x, y, ordered_indexes,
                       queried_indexes=None,
                       n_train=100, n_test=20, length_scale=20,
                       orig_x=None,
                       eval_set=None, orig_eval_set=None, n_closest=9):
    s = 0.005  # noise variance.

    top_ranked_indexes = ordered_indexes[np.arange(max(n_train, len(queried_indexes)) + n_test)]

    train, test = get_gp_train_test(top_ranked_indexes, queried_indexes, n_train, n_test)

    n_train = len(train)  # this value might be different from input
    n_test = len(test)

    # logger.debug("train indexes:\n%s\ntest indexes:\n%s" % (str(list(train)), str(list(test))))

    y_pred = None  # np.zeros(len(pred_indexes))
    v_pred = np.ones(len(test)) * 0.5

    x_train = x[train, :]
    # y_train = y[train_indexes] + s*np.random.randn(len(train_indexes))

    K = kernel(x_train, x_train, length_scale=length_scale)
    L = np.linalg.cholesky(K + s * np.eye(K.shape[0]))

    # logger.debug("K:\n%s" % str(K))

    tm = Timer()
    for i, idx in enumerate(test):

        x_test = x[idx, :]

        # compute the mean at our test points.
        Lk = np.linalg.solve(L, kernel(x_train, x_test, length_scale=length_scale))
        # y_pred[i] = np.dot(Lk.T, np.linalg.solve(L, y_train))

        # compute the variance at our test points.
        K_ = kernel(x_test, x_test, length_scale=length_scale)
        if (i + 1) % 200 == 0:
            logger.debug("Test Kernel (%d, %d)" % (K_.shape[0], K_.shape[1]))
        s2 = np.diag(K_) - np.sum(Lk ** 2, axis=0)
        v_pred[i] = np.sqrt(s2)

    tm.end()
    logger.debug(tm.message("Time for GP computation on test set:"))

    if False:
        if y_pred is not None:
            logger.debug("predicted means:\n%s" % str(list(y_pred)))
        logger.debug("predicted variances:\n%s" % str(list(v_pred)))

    v_eval = None
    if eval_set is not None:
        tm = Timer()
        closest_indexes = set()  # all indexes from test_set that are closest to any unlabeled instances
        for i in range(n_test):
            test_index = test[i]
            if orig_x is not None and orig_eval_set is not None:
                get_closest_indexes(matrix(orig_x[test_index, :], nrow=1), orig_eval_set,
                                    num=n_closest,
                                    dest_set=closest_indexes)
            else:
                get_closest_indexes(x[test_index, :], eval_set, num=n_closest, dest_set=closest_indexes)

        v_eval = np.ones(eval_set.shape[0], dtype=float) * 0.5
        for i, idx in enumerate(closest_indexes):
            x_test = eval_set[idx, :]

            # compute the mean at our test points.
            Lk = np.linalg.solve(L, kernel(x_train, x_test, length_scale=length_scale))
            # y_pred[i] = np.dot(Lk.T, np.linalg.solve(L, y_train))

            # compute the variance at our test points.
            K_ = kernel(x_test, x_test, length_scale=length_scale)
            if (i + 1) % 200 == 0:
                logger.debug("Test Kernel (%d, %d)" % (K_.shape[0], K_.shape[1]))
            s2 = np.diag(K_) - np.sum(Lk ** 2, axis=0)
            v_eval[idx] = np.sqrt(s2)
        logger.debug(tm.message("Time for GP compuation on eval set:"))

    return y_pred, v_pred, train, test, v_eval


def get_gp_predictions_ext(x, y, ranked_indexes,
                           orig_train=None, orig_test=None,
                           queried_indexes=None,
                           test_set=None, n_train=100, n_test=20, n_closest=9):
    s = 0.005  # noise variance.

    n_train = min(n_train, x.shape[0])
    train_indexes_all = SetList(ranked_indexes[np.arange(n_train)])
    if False: logger.debug("all train indexes:\n%s" % str(list(train_indexes_all)))

    # if a separate test set has *not* been provided, then it is the
    # leave-one-out case where we compute variance for each test instance
    # by first training on other instances and then computing the mean
    # nd variance for the left-out test instance.
    leave_one_out = test_set is None

    test_indexes_all = np.array(train_indexes_all)
    if queried_indexes is not None:
        # test instances can only be unlabeled instances
        test_indexes_all = np.array(SetList(train_indexes_all) - SetList(queried_indexes))

    L = None
    if leave_one_out:
        pred_indexes = np.arange(n_test)
        test_indexes_all = test_indexes_all[np.arange(n_test)]
        if False: logger.debug("all test indexes:%d\n%s" % (n_test, str(list(test_indexes_all))))
        y_pred = None  # np.zeros(len(pred_indexes))
        v_pred = np.ones(len(pred_indexes)) * 0.5
    else:
        closest_indexes = set()  # all indexes from test_set that are closest to any unlabeled instances
        for i in range(n_test):
            test_index = test_indexes_all[i]
            if orig_train is not None and orig_test is not None:
                get_closest_indexes(matrix(orig_train[test_index, :], nrow=1), orig_test, num=n_closest, dest_set=closest_indexes)
            else:
                get_closest_indexes(x[test_index, :], test_set, num=n_closest, dest_set=closest_indexes)
        pred_indexes = np.array(list(closest_indexes))
        if False: logger.debug("pred indexes:\n%s" % str(list(pred_indexes)))
        y_pred = None  # np.zeros(test_set.shape[0])
        v_pred = np.ones(test_set.shape[0]) * 0.5

    n_pred = len(pred_indexes)
    logger.debug("Leave-one-out: %s, n_pred: %d" % (str(leave_one_out), n_pred))

    tm = Timer()
    for cnt, i in enumerate(pred_indexes):

        # pick one test instance
        if leave_one_out:
            # this is the leave-out-out case
            test_index = test_indexes_all[i]
            x_test = x[test_index, :]
            # exclude the test instance from the training
            train_indexes = np.array(train_indexes_all - SetList([test_index]))
        else:
            x_test = test_set[i, :]
            train_indexes = train_indexes_all

        x_train = x[train_indexes, :]
        # y_train = y[train_indexes] + s*np.random.randn(len(train_indexes))

        if leave_one_out or cnt == 0:
            # the matrix needs to be recomputed in each iteration
            # for the leave-one-out case, else it should be
            # computed only once.
            K = kernel(x_train, x_train)
            L = np.linalg.cholesky(K + s * np.eye(K.shape[0]))

        logger.debug("K:\n%s" % str(K))

        # compute the mean at our test points.
        Lk = np.linalg.solve(L, kernel(x_train, x_test))
        # y_pred[i] = np.dot(Lk.T, np.linalg.solve(L, y_train))

        # compute the variance at our test points.
        K_ = kernel(x_test, x_test)
        if (cnt + 1) % 200 == 0:
            logger.debug("Test Kernel (%d, %d)" % (K_.shape[0], K_.shape[1]))
        s2 = np.diag(K_) - np.sum(Lk ** 2, axis=0)
        v_pred[i] = np.sqrt(s2)

    tm.end()
    logger.debug(tm.message("Time for GP compuation:"))

    if False:
        if y_pred is not None:
            logger.debug("predicted means:\n%s" % str(list(y_pred)))
        logger.debug("predicted variances:\n%s" % str(list(v_pred)))

    return y_pred, v_pred, train_indexes_all, test_indexes_all[np.arange(n_test)]


def bernoulli_explore_exploit_sample(a, b, s, f, budget, mean=False):
    bmean = np.nan
    if mean:
        bmean = beta.stats(a + s, (b + f) ** (1. - (s + f) * 1. / budget), moments='m')
    p = beta.rvs(a + s, (b + f) ** (1. - (s + f) * 1. / budget), size=1)
    r = bernoulli.rvs(p, size=1)
    return r, bmean


def thompson_sample(a, b, reward_history, mean=False):
    n = reward_history.shape[0]
    samples = np.zeros(n, dtype=float)
    bmean = None
    if mean:
        bmean = np.zeros(n, dtype=float)
    for i in range(n):
        s = reward_history[i, 0]  # success counts
        f = reward_history[i, 1]  # failure counts
        samples[i] = beta.rvs(a + s, (b + f), size=1)
        if mean:
            bmean[i] = beta.stats(a + s, (b + f), moments='m')
    action = np.argmax(samples)
    return action, samples, bmean
