from loda_support import *


# ==================================================
# Simple pairwise
# ==================================================

class SimplePairwiseSolution(object):
    def __init__(self, w=None, success=True, tries=0, loss=0, loss_old=0, delta=0):
        self.w = w
        self.success = success
        self.tries = tries
        self.loss = loss
        self.loss_old = loss_old
        self.delta = delta


def get_simple_pairwise_loss_grad(w, xi, yi, x_tau, w_prior, Ca, Cn, sigma2):
    m = ncol(xi)
    loss_a = rep(0, m)  # the derivative of loss w.r.t w for anomalies
    loss_n = rep(0, m)  # the derivative of loss w.r.t w for nominals
    n_anom = 0
    n_noml = 0
    qval = x_tau.dot(w)
    vals = xi.dot(w)
    for i in range(len(yi)):
        lbl = yi[i]
        if lbl == 1 and vals[i] < qval:
            loss_a[:] = loss_a + Ca * (x_tau - xi[i, :])
            n_anom += 1
        elif lbl == 0 and vals[i] >= qval:
            loss_n[:] = loss_n + Cn * (xi[i, :] - x_tau)
            n_noml += 1
        else:
            # no loss
            pass
    w_prior_diff = (1/sigma2) * (w - w_prior)
    dl_dw = (loss_a / max(1, n_anom)) + (loss_n / max(1, n_noml)) + w_prior_diff
    return dl_dw


# simple pairwise loss:
#   \sum C * l(w; qval, (x,y))
#   = Ca*l_A(x_A.w - qval) + Cn*l_N(qval - x_N.w) + (1/2*sigma2)*||(w - w_prior).(w - w_prior)||
def simple_pairwise_loss(w, xi, yi, x_tau, w_prior, Ca, Cn, sigma2):
    loss_a = 0  # loss w.r.t w for anomalies
    loss_n = 0  # loss w.r.t w for nominals
    n_anom = 0
    n_noml = 0
    qval = x_tau.dot(w)
    vals = xi.dot(w)
    for i in range(len(yi)):
        if yi[i] == 1 and vals[i] < qval:
            loss_a = loss_a + Ca * (qval - vals[i])
            n_anom += 1
        elif yi[i] == 0 and vals[i] >= qval:
            loss_n = loss_n + Cn * (vals[i] - qval)
            n_noml += 1
        else:
            # no loss
            pass
    w_prior_diff = w - w_prior
    w_prior_loss = (1/(2*sigma2)) * w_prior_diff.dot(w_prior_diff)
    loss = (loss_a / max(1, n_anom)) + (loss_n / max(1, n_noml)) + w_prior_loss
    logger.debug("loss: %f" % loss)
    return loss


def prepare_simple_pairwise_optim_functions(w, xi, yi, x_tau, w_prior, Ca, Cn, sigma2):
    def f(z):
        return simple_pairwise_loss(z, xi, yi, x_tau, w_prior, Ca, Cn, sigma2)
    def g(z):
        return get_simple_pairwise_loss_grad(z, xi, yi, x_tau, w_prior, Ca, Cn, sigma2)
    return f, g


def weight_update_simple_pairwise_(x, y, hf, w, w_prior,
                                  Ca=1.0, Cn=1.0, sigma2=1.0, topK=30, max_iters=10000):
    """ Uses scipy optimizer package to solve the optimization problem
    
    Note: This finds worse solutions than using SGD. Hence SGD might be preferred.
    
    :param x: 
    :param y: 
    :param hf: 
    :param w: 
    :param w_prior: 
    :param Ca: 
    :param Cn: 
    :param sigma2: 
    :param topK: 
    :param max_iters: 
    :return: 
    """
    m = ncol(x)

    nf = len(hf)
    xi = matrix(x[hf, :], nrow=nf, ncol=m)
    yi = y[hf]

    s = x.dot(w)
    ps = order(s, decreasing=True)[topK]
    x_tau = matrix(x[ps, :], nrow=1)
    f, g = prepare_simple_pairwise_optim_functions(w, xi, yi, x_tau, w_prior, Ca, Cn, sigma2)
    w_new, success = constr_optim(theta=w, f=f, grad=g, hessian=None, ui=None, ci=None,
                                  a=None, b=None, bounds=None,
                                  method="SLSQP",
                                  )
    soln = SimplePairwiseSolution(w=w_new, success=success, tries=0,
                                  loss=0, loss_old=0,
                                  delta=0)
    return soln


def weight_update_simple_pairwise(x, y, hf, w, w_prior,
                                  Ca=1.0, Cn=1.0, sigma2=1.0, topK=30, max_iters=10000):
    """ Uses SGD to solve the optimization problem.
    
    Note: This seems to work better than Scipy's optimization package
    
    :param x: 
    :param y: 
    :param hf: 
    :param w: 
    :param w_prior: 
    :param Ca: 
    :param Cn: 
    :param sigma2: 
    :param topK: 
    :param max_iters: 
    :return: 
    """
    opt_success = False
    k = 0
    delta = 1e-6
    m = ncol(x)

    nf = len(hf)
    xi = matrix(x[hf, :], nrow=nf, ncol=m)
    yi = y[hf]

    s = x.dot(w)
    ps = order(s, decreasing=True)[topK]
    x_tau = matrix(x[ps, :], nrow=1)
    loss = simple_pairwise_loss(w, xi, yi, x_tau, w_prior, Ca, Cn, sigma2)
    loss_old = 0
    best_loss = np.inf
    best_w = w
    while not opt_success and k <= max_iters:
        k += 1
        loss_old = loss
        gam = 1e-4  # 1/sqrt(k)
        try:
            w_grad = get_simple_pairwise_loss_grad(w, xi, yi, x_tau, w_prior, Ca, Cn, sigma2)
            w = w - gam * w_grad
            if np.sum(w ** 2) > 0:
                w = w / np.sqrt(np.sum(w ** 2))
            else:
                # if sum of weights is zero, make them uniform ...
                w = rep(1.0 / np.sqrt(m), m)
            s = x.dot(w)
            ps = order(s, decreasing=True)[topK]
            x_tau = matrix(x[ps, :], nrow=1)
            loss = simple_pairwise_loss(w, xi, yi, x_tau, w_prior, Ca, Cn, sigma2)
            if loss < best_loss:
                best_loss = loss
                best_w = w
                #logger.debug("best w: \n%s" % str(list(w)))
            if np.abs(loss - loss_old) <= delta:
                opt_success = True
            if k % 1000 == 0:
                logger.debug("Iter: %d; old loss: %f; loss: %f; del_loss: %f" %
                             (k, loss_old, loss, np.abs(loss - loss_old)))
        except BaseException, e:
            logger.warning("Optimization Err at try %d : %s" % (k, str(e)))

    if opt_success:
        soln = SimplePairwiseSolution(w=best_w, success=True, tries=k,
                                      loss=best_loss, loss_old=loss_old,
                                      delta=abs(loss - loss_old))
    else:
        logger.debug("Did not converge in %d iterations..." % k)
        soln = SimplePairwiseSolution(w=best_w, success=False, tries=k,
                                      loss=best_loss, loss_old=loss_old,
                                      delta=abs(loss - loss_old))
    return soln


# ==================================================
# Simple online inference
# ==================================================

def get_relative_loss_gradient_vector(w, xi, yi, x_anomaly, x_nominal,
                                      nu=1.0, Ca=1.0, Cn=1.0):
    m = len(w)
    ha = np.where(yi == 1)[0]
    hn = np.where(yi == 0)[0]
    grad = np.zeros(m, dtype=float)
    if len(hn) > 0:
        # grad = Cn*((apply(matrix(xi[hn,], ncol=m), MARGIN = 2, FUN = mean)) - x_nominal)
        grad = Cn * (np.mean(matrix(xi[hn, :], ncol=m), axis=0) - x_nominal)
    if len(ha) > 0:
        # grad = grad - Ca*((apply(matrix(xi[ha,], ncol=m), MARGIN = 2, FUN = mean)) - x_anomaly)
        grad = Cn * (np.mean(matrix(xi[ha, :], ncol=m), axis=0) - x_nominal)
    g_norm = np.sqrt(grad.dot(grad))
    if g_norm > 0:
        grad = grad / g_norm
    return grad


def weight_update_online_simple(x, y, hf, w,
                                nu=1.0, Ca=1.0, Cn=1.0,
                                sigma2=0.5, R=None,
                                relativeto=RELATIVE_MEAN,
                                tau_anomaly=0.03, tau_nominal=np.nan):
    tau_nominal = tau_anomaly if tau_nominal is np.nan else tau_nominal
    m = len(w)
    nf = len(hf)
    if nf == 0:
        return w
    xi = matrix(x[hf, :], nrow=nf, ncol=m)
    yi = y[hf]
    ha = np.where(yi == 1)[0]
    hn = np.where(yi == 0)[0]

    if relativeto == RELATIVE_MEAN:
        # get mean over the entire dataset
        x_anomaly = np.mean(x, axis=0)
        x_nominal = x_anomaly
    elif relativeto == RELATIVE_QUANTILE:
        x_anomaly = np.apply_along_axis(np.percentile, 0, x, 1-tau_anomaly)
        x_nominal = np.apply_along_axis(np.percentile, 0, x, 1-tau_nominal)
    else:
        raise ValueError("incorrect value for relativeto")
    grad = get_relative_loss_gradient_vector(w, xi, yi, x_anomaly, x_nominal, nu, Ca, Cn)
    w_tilde = w - nu * sigma2 * grad
    w_new = w_tilde/np.sqrt(w_tilde.dot(w_tilde))
    return w_new
