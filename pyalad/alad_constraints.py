from loda_support import *


def to_dense_mat(x):
    if isinstance(x, csr_matrix):
        return x.toarray()
    else:
        return x


def setup_constraints_scipy(xi, yi, ha, hn, x_tau=None, constraint_type=AAD_CONSTRAINT_PAIRWISE):
    """Constraint setup for Scipy optimization library

    Needs slack variables to be setup explicitly.

    :param xi:
    :param yi:
    :param ha:
    :param hn:
    :param x_tau:
    :param constraint_type:
    :return:
    """
    m = ncol(xi)
    nha = len(ha)
    nhn = len(hn)

    if constraint_type == AAD_CONSTRAINT_WEIGHTS_POSITIVE_SUM_1:
        # no pairwise constraints
        npairs = 0
    elif constraint_type == AAD_CONSTRAINT_TAU_INSTANCE:
        # All constraints will be relative to the tau-th instance
        npairs = len(ha) + len(hn)
    else:
        # All anomalies will be scored higher than all nominals
        npairs = len(ha) * len(hn)

    if constraint_type == AAD_CONSTRAINT_TAU_INSTANCE and x_tau is None:
        raise ValueError("AAD_CONSTRAINT_TAU_INSTANCE constraint requires a valid instance")

    theta = rep(0, m + npairs)  # w and slacks
    ui = None
    ci = None
    a = None
    b = None
    if (constraint_type == AAD_CONSTRAINT_PAIRWISE_WEIGHTS_POSITIVE_SUM_1
            or constraint_type == AAD_CONSTRAINT_WEIGHTS_POSITIVE_SUM_1):
        bounds = [(0, None) for _ in range(m)]  # weights are positive
        # weights sum to 1
        a = np.zeros(shape=(1, m + npairs), dtype=float)
        a[0, 0:m] = np.ones(m, dtype=float)
        b = np.ones(1, dtype=float)
    elif constraint_type == AAD_CONSTRAINT_PAIRWISE \
            or constraint_type == AAD_CONSTRAINT_TAU_INSTANCE:
        bounds = [(None, None) for _ in range(m)]  # No restriction on weights
    else:
        raise ValueError("Incorrect constraint type: %d" % (constraint_type,))

    if npairs > 0:
        ui = np.zeros(shape=(npairs, m + npairs), dtype=float)
        ci = rep(0., npairs)
        if constraint_type == AAD_CONSTRAINT_TAU_INSTANCE:
            ij = 0
            for i in range(nha):
                ui[ij, 0:m] = to_dense_mat(xi[ha[i], :] - x_tau[0, :])
                ui[ij, m + ij] = 1  # add slack variables
                ij += 1
            for j in range(nhn):
                ui[ij, 0:m] = to_dense_mat(x_tau[0, :] - xi[hn[j], :])
                ui[ij, m + ij] = 1  # add slack variables
                ij += 1
            # logger.debug(ui)
        else:
            for i in range(nha):
                for j in range(nhn):
                    ij = i * nhn + j
                    ui[ij, 0:m] = to_dense_mat(xi[ha[i], :] - xi[hn[j], :])
                    ui[ij, m + ij] = 1  # add slack variables
            # logger.debug(ui)

    # Below we set initial value of slack variables
    # to a high value such that optimization can start
    # in a feasible region
    theta[m:len(theta)] = 1.
    for _ in range(npairs):
        bounds.append((0, None))  # slack variables >= 0

    return theta, bounds, ui, ci, a, b


def setup_constraints_cvxopt(xi, yi, ha, hn, x_tau=None, constraint_type=AAD_CONSTRAINT_PAIRWISE):
    """Constraint setup to use with CVXOPT

    CVXOPT computes the slack variables automatically and hence we
    do not need to set them up separately.

    :param xi:
    :param yi:
    :param ha:
    :param hn:
    :param x_tau:
    :param constraint_type:
    :param optimlib:
    :return:
    """
    m = ncol(xi)
    nha = len(ha)
    nhn = len(hn)

    if constraint_type == AAD_CONSTRAINT_WEIGHTS_POSITIVE_SUM_1:
        # no pairwise constraints
        npairs = 0
    elif constraint_type == AAD_CONSTRAINT_TAU_INSTANCE:
        # All constraints will be relative to the tau-th instance
        npairs = len(ha) + len(hn)
    else:
        npairs = len(ha) * len(hn)

    # logger.debug("nha: %d, nhn: %d, npairs: %d" % (nha, nhn, npairs))

    # theta = rep(1/np.sqrt(m), m + npairs)  # w and slacks
    theta = rep(0, m + npairs)  # w and slacks
    ui = None
    ci = None
    a = None
    b = None
    if (constraint_type == AAD_CONSTRAINT_PAIRWISE_WEIGHTS_POSITIVE_SUM_1
            or constraint_type == AAD_CONSTRAINT_WEIGHTS_POSITIVE_SUM_1):
        bounds = [(0, None) for _ in range(m)]  # weights are positive
        # weights sum to 1
        a = np.ones(shape=(1, m + npairs), dtype=float)
        b = np.ones(1, dtype=float)
    elif constraint_type == AAD_CONSTRAINT_PAIRWISE \
            or constraint_type == AAD_CONSTRAINT_TAU_INSTANCE:
        bounds = [(None, None) for _ in range(m)]  # No restriction on weights
        a = np.empty(shape=(0, m + npairs), dtype=float)
        b = np.empty(0, dtype=float)
    else:
        raise ValueError("Incorrect constraint type: %d" % (constraint_type,))

    if npairs > 0:
        ui = np.zeros(shape=(npairs, m + npairs), dtype=float)
        ci = rep(0., npairs)
        if constraint_type == AAD_CONSTRAINT_TAU_INSTANCE:
            ij = 0
            for i in range(nha):
                ui[ij, 0:m] = to_dense_mat(xi[ha[i], :] - x_tau[0, :])
                ui[ij, m + ij] = 1  # add slack variables
                ij += 1
            for j in range(nhn):
                ui[ij, 0:m] = to_dense_mat(x_tau[0, :] - xi[hn[j], :])
                ui[ij, m + ij] = 1  # add slack variables
                ij += 1
            # logger.debug(ui)
        else:
            for i in range(nha):
                for j in range(nhn):
                    ij = i * nhn + j
                    ui[ij, 0:m] = to_dense_mat(xi[ha[i], :] - xi[hn[j], :])
                    ui[ij, m + ij] = 1  # add slack variables
            # logger.debug(ui)

    theta[m:len(theta)] = 0.1
    for _ in range(npairs):
        bounds.append((0, None))  # slack variables >= 0

    return theta, bounds, ui, ci, a, b


def setup_constraints(xi, yi, ha, hn, x_tau=None, constraint_type=AAD_CONSTRAINT_PAIRWISE, optimlib=OPTIMLIB_SCIPY):
    if optimlib == OPTIMLIB_SCIPY:
        return setup_constraints_scipy(xi, yi, ha, hn, x_tau=x_tau, constraint_type=constraint_type)
    else:
        return setup_constraints_cvxopt(xi, yi, ha, hn, x_tau=x_tau, constraint_type=constraint_type)


