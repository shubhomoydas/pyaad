from loda_support import *
from alad_loss_functions import *


def weight_update_aatp_pairwise_constrained_orig(x, y, hf, w, qval,
                                                 Ca=1.0, Cn=1.0, Cx=1.0,
                                                 withprior=False, w_prior=None,
                                                 w_old=None, sigma2=1.0,
                                                 pseudoanomrank=0,
                                                 nu=1.0,
                                                 pseudoanomrank_always=False,
                                                 optimlib=OPTIMLIB_SCIPY):
    nf = len(hf)
    if nf == 0:
        return w, None

    m = ncol(x)
    xi_orig = matrix(x[hf, :], nrow=nf, ncol=m)
    yi_orig = y[hf]
    xi = xi_orig.copy()
    yi = yi_orig.copy()
    ha = np.where(yi == 1)[0]
    if ((len(ha) == 0 and pseudoanomrank > 0) or
            (pseudoanomrank_always and pseudoanomrank > 0)):
        # get the pseudo anomaly instance
        s = x.dot(w)
        ps = order(s, decreasing=True)[pseudoanomrank]
        xi = rbind(xi, matrix(x[ps, :], nrow=1))
        yi = append(yi, 1)
        ha = append(ha, len(hf))

    hn = np.where(yi == 0)[0]
    nha = len(ha)
    nhn = len(hn)
    npairs = len(ha)*len(hn)

    if False:
        logger.debug("xi.shape: %s" % (str(xi.shape),))
        logger.debug(yi)
        logger.debug("npairs: %d" % (npairs,))
        logger.debug(ha)
        logger.debug(hn)

    theta = rep(0, m+npairs) # w and slacks
    if npairs == 0:
        # turn off constraints
        ui = None
        ci = None
    else:
        # ncol(ui) = dim of weights + no. of slack vars
        # nrow(ui) = no. of pair-wise constraints + no. of slack variables
        # Note: no. of pair-wise constraints = no. of slack variables,
        # hence, 2*npairs
        ui = np.zeros(shape=(2*npairs, m+npairs), dtype=float)
        ci = rep(0, 2*npairs)
        for i in range(nha):
            for j in range(nhn):
                ij = i * nhn + j
                ui[2*ij    ,  0:m] = xi[ha[i], :] - xi[hn[j], :]
                ui[2*ij    , m+ij] = 1  # add slack variables
                ui[2*ij + 1, m+ij] = 1  # slack variables must be positive
        # Below we set initial value of slack variables
        # to a high value such that optimization can start
        # in a feasible region
        theta[m:len(theta)] = 100
    if False:
        logger.debug("ui:")
        logger.debug(ui)
        logger.debug("ui:")
        logger.debug(ci)
    # In the below call we send the xi_orig and yi_orig which
    # *do not* contain the pseudo anomaly. Pseudo anomaly is
    # only used to create the constraint matrices

    f, grad, hess = prepare_aatp_optim_functions(theta, xi=xi_orig, yi=yi_orig, qval=qval,
                                                 Ca=Ca, Cn=Cn, Cx=Cx,
                                                 withprior=withprior, w_prior=w_prior, w_old=w_old, sigma2=sigma2,
                                                 nu=nu, optimlib=optimlib)

    soln, success = aatp_constriant_optim(theta=theta, fn=f, grad=grad, hessian=hess,
                                          ui=ui, ci=ci, optimlib=optimlib)
    w_new = soln[0:m]
    slack = None
    if len(theta) > m:
        slack = soln[range(m, len(theta))]
    if True:
        logger.debug("Success %s; |w|=%f" % (str(success), np.sum(w_new)))
        if False and not success:
            logger.debug(w_new)
        if False and slack is not None:
            logger.debug("Slack variables:")
            logger.debug(np.round(slack, 4))
    w_new = w_new/np.sqrt(w_new.dot(w_new))
    return w_new, slack


