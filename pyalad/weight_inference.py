from loda_support import *
from alad_loss_functions import *
from alad_constraints import *


def get_aatp_quantile(x, w, topK):
    s = x.dot(w)
    return quantile(s, (1.0 - (topK*1.0/float(nrow(x))))*100.0)


def prepare_aatp_optim_functions(theta, xi, yi, qval, Ca=1.0, Cn=1.0, Cx=1.0,
                                 withprior=False, w_prior=None, w_old=None, sigma2=1.0, nu=1.0,
                                 optimlib=OPTIMLIB_SCIPY):

    square_slack = True
    # since CVXOPT works well with a positive definite Hessian, we use square slack in loss.
    # square_slack = optimlib == OPTIMLIB_CVXOPT
    if False and square_slack:
        logger.debug("Square slack-loss being used instead of absolute slack loss...")

    optimcontext = OptimizationContext(20000, 0)

    def f(z):
        # logger.debug("f(z)")
        # logger.debug(z)
        return aatp_slack_loss(z, xi=xi, yi=yi, qval=qval,
                               Ca=Ca, Cn=Cn, Cx=Cx,
                               withprior=withprior, w_prior=w_prior,
                               w_old=w_old, sigma2=sigma2, nu=nu, square_slack=square_slack)

    def grad(z):
        # logger.debug("grad(z)")
        # logger.debug(z)
        return aatp_slack_loss_gradient(z, xi=xi, yi=yi, qval=qval,
                                        Ca=Ca, Cn=Cn, Cx=Cx,
                                        withprior=withprior, w_prior=w_prior,
                                        w_old=w_old, sigma2=sigma2, nu=nu, square_slack=square_slack, optimcontext=optimcontext)

    hess = None
    if optimlib == OPTIMLIB_CVXOPT:
        def hess(z):
            # logger.debug("hess(z)")
            # logger.debug(z)
            h = aatp_slack_loss_hessian(z, xi=xi, yi=yi, qval=qval,
                                        Ca=Ca, Cn=Cn, Cx=Cx,
                                        withprior=withprior, w_prior=w_prior,
                                        w_old=w_old, sigma2=sigma2, nu=nu, square_slack=square_slack)
            # logger.debug("diag(h): %s" % str(np.diag(h)))
            return h

    return f, grad, hess


def weight_update_aatp_pairwise_constrained_inner(x, y, hf, w, qval,
                                                  Ca=1.0, Cn=1.0, Cx=1.0,
                                                  withprior=False, w_prior=None,
                                                  w_old=None, sigma2=1.0,
                                                  pseudoanomrank=0,
                                                  nu=1.0,
                                                  pseudoanomrank_always=False,
                                                  constraint_type=AAD_CONSTRAINT_PAIRWISE,
                                                  max_anomalies_in_constraint_set=1000,
                                                  max_nominals_in_constraint_set=1000,
                                                  optimlib=OPTIMLIB_SCIPY):
    """
    Uses optimizer bounds instead of constraints for slack_vars >= 0

    :param x:
    :param y:
    :param hf:
    :param w:
    :param qval:
    :param Ca:
    :param Cn:
    :param Cx:
    :param withprior:
    :param w_prior:
    :param w_old:
    :param sigma2:
    :param pseudoanomrank:
    :param nu:
    :param pseudoanomrank_always:
    :param constraint_type:
    :param max_anomalies_in_constraint_set:
    :param max_nominals_in_constraint_set:
    :param optimlib:
    :return:
    """
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

    # select only a subset of anomaly datapoints for constraints
    if len(ha) > max_anomalies_in_constraint_set:
        constr_ha = ha[sample(range(len(ha)), max_anomalies_in_constraint_set)]
    else:
        constr_ha = ha

    # select only a subset of nominal datapoints for constraints
    if len(hn) > max_nominals_in_constraint_set:
        constr_hn = hn[sample(range(len(hn)), max_nominals_in_constraint_set)]
    else:
        constr_hn = hn

    if False:
        logger.debug("max ha: %d, max hn: %d" %
                     (max_anomalies_in_constraint_set, max_nominals_in_constraint_set))
        # logger.debug("xi.shape: %s" % (str(xi.shape),))
        # logger.debug(yi)
        # logger.debug(ha)
        # logger.debug(hn)
        # npairs = len(ha) * len(constr_hn)
        # logger.debug("npairs: %d" % (npairs,))

    # theta, bounds, ui, ci, a, b = (None, None, None, None, None, None)
    theta, bounds, ui, ci, a, b = setup_constraints(xi, yi, constr_ha, constr_hn,
                                                    constraint_type=constraint_type, optimlib=optimlib)

    if False:
        # logger.debug("ui:")
        # logger.debug(ui)
        # logger.debug("ci:")
        # logger.debug(ci)
        if len(theta) > m:
            logger.debug("ui slack:\n%s" % str(ui[:, m:len(theta)]))
    # In the below call we send the xi_orig and yi_orig which
    # *do not* contain the pseudo anomaly. Pseudo anomaly is
    # only used to create the constraint matrices

    f, grad, hess = prepare_aatp_optim_functions(theta, xi=xi_orig, yi=yi_orig, qval=qval,
                                                 Ca=Ca, Cn=Cn, Cx=Cx,
                                                 withprior=withprior, w_prior=w_prior, w_old=w_old, sigma2=sigma2,
                                                 nu=nu, optimlib=optimlib)

    kktsolver = None  # get_kktsolver_no_equality_constraints(ui=ui, fn=f, grad=grad, hessian=hess)
    soln, success = aatp_constriant_optim(theta=theta, fn=f, grad=grad, hessian=hess,
                                          ui=ui, ci=ci, a=a, b=b, bounds=bounds,
                                          kktsolver=kktsolver,
                                          optimlib=optimlib)
    w_new = soln[0:m]
    slack = None
    if len(theta) > m:
        slack = soln[range(m, len(theta))]
    if False:
        logger.debug("Success %s; |w|=%f" % (str(success), np.sum(w_new)))
        if False and not success:
            logger.debug("w_new: \n%s" % str(w_new))
            # logger.debug(w_new)
        if True and slack is not None:
            logger.debug("Slack variables: \n%s" % (str(slack)))
            # logger.debug(np.round(slack, 4))

    # normalize w_new
    l2_w_new = np.sqrt(w_new.dot(w_new))
    if l2_w_new == 0:
        w_new = rep(1/np.sqrt(m), m)
    else:
        w_new = w_new/l2_w_new

    return w_new, slack


class AATPSolution(object):
    def __init__(self, w, slack, success, tries, Cx):
        self.w = w
        self.slack = slack
        self.success = success
        self.tries = tries
        self.Cx = Cx


def weight_update_aatp_slack_pairwise_constrained(x, y, hf, w, qval,
                                                  Ca=1.0, Cn=1.0, Cx=1.0,
                                                  withprior=False, w_prior=None,
                                                  w_old=None, sigma2=1.0,
                                                  pseudoanomrank=0,
                                                  nu=1.0, pseudoanomrank_always=False,
                                                  constraint_type=AAD_CONSTRAINT_PAIRWISE,
                                                  max_anomalies_in_constraint_set=1000,
                                                  max_nominals_in_constraint_set=1000,
                                                  optimlib=OPTIMLIB_SCIPY):
    # In this method we try multiple times if needed with
    # different values of Cx. It seems that when Cx is too
    # high, then the constraints are very hard to satisfy
    # and constrOptim returns an error: initial value in 'vmmin' is not finite
    opt_success = False
    w_soln = None
    try:
        w_soln = weight_update_aatp_pairwise_constrained_inner(
            x, y,
            hf=hf,
            w=w, qval=qval,
            Ca=Ca, Cn=Cn, Cx=Cx,
            withprior=withprior, w_prior=w_prior,
            w_old=w_old,
            sigma2=sigma2,
            pseudoanomrank=pseudoanomrank,
            nu=nu, pseudoanomrank_always=pseudoanomrank_always,
            constraint_type=constraint_type,
            max_anomalies_in_constraint_set=max_anomalies_in_constraint_set,
            max_nominals_in_constraint_set=max_nominals_in_constraint_set,
            optimlib=optimlib)

        if False:
            # TODO: remove the below after debug
            w_solx = weight_update_aatp_pairwise_constrained_inner(
                x, y,
                hf=hf,
                w=w, qval=qval,
                Ca=Ca, Cn=Cn, Cx=Cx,
                withprior=withprior, w_prior=w_prior,
                w_old=w_old,
                sigma2=sigma2,
                pseudoanomrank=pseudoanomrank,
                nu=nu, pseudoanomrank_always=pseudoanomrank_always,
                constraint_type=constraint_type,
                max_anomalies_in_constraint_set=max_anomalies_in_constraint_set,
                max_nominals_in_constraint_set=max_nominals_in_constraint_set,
                optimlib=OPTIMLIB_CVXOPT)

            logger.debug(w_soln)
            logger.debug(w_solx)
        opt_success = True
    except BaseException, e:
        # logger.debug(exception_to_string(sys.exc_info()))
        logger.warning("Optimization Err '%s'; continuing with previous parameters." % (str(e),))
    if opt_success:
        soln = AATPSolution(w=w_soln[0], slack=w_soln[1], success=True, tries=1, Cx=Cx)
    else:
        soln = AATPSolution(w=w, slack=None, success=False, tries=1, Cx=Cx)
    return soln


# ==================================================
# AATP inference
# ==================================================

def aatp_opt_args_wrapper(**kwargs):
    return (kwargs.get('xi', None),
            kwargs.get('yi', None),
            kwargs.get('qval', None),
            kwargs.get('Ca', None),
            kwargs.get('Cn', None),
            kwargs.get('Cx', None),
            kwargs.get('withprior', False),
            kwargs.get('w_prior', None),
            kwargs.get('w_old', None),
            kwargs.get('sigma2', 1.0),
            kwargs.get('nu', 1.0))


def aatp_constriant_optim(theta, fn, grad=None, hessian=None,
                          ui=None, ci=None, a=None, b=None, bounds=None,
                          kktsolver = None,
                          optimlib=OPTIMLIB_SCIPY):
    # logger.debug("theta shape: %s" % (str(theta.shape),))
    if optimlib == OPTIMLIB_SCIPY:
        # method L-BFGS-B of scipy.minimize cannot handle constraints
        # method COBYLA of scipy.minimize cannot handle bounds
        # method SLSQP of scipy.minimize seems to require minimal programming overhead...
        res = constr_optim(theta=theta, f=fn, grad=grad, hessian=hessian, ui=ui, ci=ci, a=a, b=b, bounds=bounds,
                           method="SLSQP",
                           )
    elif optimlib == OPTIMLIB_CVXOPT:
        res = cvx_optim(theta=theta, f=fn, grad=grad, hessian=hessian, ui=ui, ci=ci, a=a, b=b, bounds=bounds,
                        kktsolver=kktsolver
                        )
    else:
        raise ValueError("Invalid optimization library: %s" % (optimlib,))
    return res

