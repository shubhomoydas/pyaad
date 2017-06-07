from loda_support import *
from alad_loss_functions import *
from alad_constraints import *


def get_aatp_quantile(x, w, topK):
    s = x.dot(w)
    return quantile(s, (1.0 - (topK*1.0/float(nrow(x))))*100.0)


def get_score_ranges(x, w):
    s = x.dot(w)
    qvals = list()
    qvals.append(np.min(s))
    for i in range(1, 10):
        qvals.append(quantile(s, (i * 10.0)))
    qvals.append(np.max(s))
    return qvals


def order_by_diff_from_tau(x, w, ha, hn, x_tau):
    """
    Orders labeled anomalies and nominal instances (in ha, hn resp.)
    by the difference between their scores and the tau-th instance score.

    For an anomaly x \in ha, sort ha in increasing order by:
        score(x) - score(x_tau)

    For a nominal x \in hn, sort hn in increasing order by:
        score(x_tau) - score(x)

    :param x:
    :param w:
    :param ha:
    :param hn:
    :return:
    """
    s = x.dot(w)
    qtau = x_tau.dot(w)
    if len(ha) > 0:
        diff_ha = s[ha] - qtau
        # logger.debug("diff_ha: %s" % str(list(diff_ha)))
        new_ha = ha[order(diff_ha)]
    else:
        new_ha = ha.copy()  # return copy of empty array
    if len(hn) > 0:
        diff_hn = qtau - s[hn]
        # logger.debug("diff_hn: %s" % str(list(diff_hn)))
        new_hn = hn[order(diff_hn)]
    else:
        new_hn = hn.copy()  # return copy of empty array
    return new_ha, new_hn


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


def get_tau_ranked_instance(x, w, tau_rank):
    s = x.dot(w)
    ps = order(s, decreasing=True)[tau_rank]
    return matrix(x[ps, :], nrow=1)


def weight_update_aatp_pairwise_constrained_inner(x, y, hf, w, qval,
                                                  Ca=1.0, Cn=1.0, Cx=1.0,
                                                  withprior=False, w_prior=None,
                                                  w_old=None, sigma2=1.0,
                                                  pseudoanomrank=0,
                                                  nu=1.0,
                                                  pseudoanomrank_always=False,
                                                  order_by_violated=False,
                                                  ignore_aatp_loss=False,
                                                  random_instance_at_start=False,
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
    :param order_by_violated:
    :param ignore_aatp_loss:
    :param random_instance_at_start:
    :param constraint_type:
    :param max_anomalies_in_constraint_set:
    :param max_nominals_in_constraint_set:
    :param optimlib:
    :return:
    """

    # whether to ignore anomalies from the pair-wise constraints.
    # If add_nominal_constraints_only is True, the anomalies would
    # still be part of the AATP hinge-loss, just absent from the constraints
    add_nominal_constraints_only = False

    # whether to ignore anomalies completely.
    # If ignore_anomalies is True, they would neither be part
    # of AATP hinge-loss, nor of constraints.
    ignore_anomalies = False

    # whether to ignore AATP hinge loss.
    # if ignored, then only constraints and prior will be used.
    # ignore_aatp_loss = False

    # logger.debug("order_by_violated: %s" % str(order_by_violated))

    # logger.debug("pseudoanomrank: %d, constraint type: %d" % (pseudoanomrank, constraint_type))

    if ignore_anomalies:
        # remove all anomalies from feedback list
        # logger.debug("hf: %s" % str(zip(hf, y[hf])))
        hf = np.array([v for v in hf if y[v] == 0])
        if False and len(hf > 0):
            logger.debug("hf: %s" % str(zip(hf, y[hf])))

    nf = len(hf)
    if nf == 0:
        return w, None

    m = ncol(x)

    x_tau = None
    if constraint_type == AAD_CONSTRAINT_TAU_INSTANCE and pseudoanomrank > 0:
        x_tau = get_tau_ranked_instance(x, w, pseudoanomrank)

    xi_orig = matrix(x[hf, :], nrow=nf, ncol=m)
    yi_orig = y[hf]
    xi = xi_orig.copy()
    yi = yi_orig.copy()

    if add_nominal_constraints_only:
        # remove all anomalies from the list ha which
        # is used to construct the pair-wise constraints
        ha = np.array([], dtype=int)
    else:
        ha = np.where(yi == 1)[0]

    if len(ha) == 0 and pseudoanomrank > 0 \
            and constraint_type != AAD_CONSTRAINT_TAU_INSTANCE:
        # get the pseudo anomaly instance
        xi = rbind(xi, get_tau_ranked_instance(x, w, pseudoanomrank))
        yi = append(yi, 1)
        ha = append(ha, len(hf))

    hn = np.where(yi == 0)[0]

    if order_by_violated:
        if x_tau is None:
            x_tau = get_tau_ranked_instance(x, w, pseudoanomrank)
        ha, hn = order_by_diff_from_tau(xi, w, ha, hn, x_tau)

    # select only a subset of anomaly data points for constraints
    if len(ha) > max_anomalies_in_constraint_set:
        if order_by_violated:
            # retain the most violated instances
            constr_ha = ha[range(max_anomalies_in_constraint_set)]
        else:
            constr_ha = ha[sample(range(len(ha)), max_anomalies_in_constraint_set)]
    else:
        constr_ha = ha

    # select only a subset of nominal data points for constraints
    if len(hn) > max_nominals_in_constraint_set:
        if order_by_violated:
            # retain the most violated instances
            constr_hn = hn[range(max_nominals_in_constraint_set)]
        else:
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
    theta, bounds, ui, ci, a, b = setup_constraints(xi, yi, constr_ha, constr_hn, x_tau=x_tau,
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

    if ignore_aatp_loss:
        xi_orig = matrix(np.zeros(0), nrow=0, ncol=ncol(xi_orig))
        yi_orig = np.zeros(0, dtype=int)

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
                                                  order_by_violated=False,
                                                  ignore_aatp_loss=False,
                                                  random_instance_at_start=False,
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
            order_by_violated=order_by_violated,
            ignore_aatp_loss=ignore_aatp_loss,
            random_instance_at_start=random_instance_at_start,
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
                order_by_violated=order_by_violated,
                ignore_aatp_loss=ignore_aatp_loss,
                random_instance_at_start=random_instance_at_start,
                constraint_type=constraint_type,
                max_anomalies_in_constraint_set=max_anomalies_in_constraint_set,
                max_nominals_in_constraint_set=max_nominals_in_constraint_set,
                optimlib=OPTIMLIB_CVXOPT)

            logger.debug(w_soln)
            logger.debug(w_solx)
        opt_success = True
    except BaseException as e:
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

