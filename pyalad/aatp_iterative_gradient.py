from loda_support import *


class IterGradSolution(object):
    def __init__(self, w=None, success=True, tries=0, loss=0, loss_old=0, delta=0):
        self.w = w
        self.success = success
        self.tries = tries
        self.loss = loss
        self.loss_old = loss_old
        self.delta = delta


def get_iter_grad_loss_vector (x, y, hf, qval, s, Ca, Cn):
    m = ncol(x)
    loss_a = rep(0, m)  # the derivative of loss w.r.t w for anomalies
    loss_n = rep(0, m)  # the derivative of loss w.r.t w for nominals
    n_anom = 0
    n_noml = 0
    for i in hf:
        lbl = y[i]
        if lbl == 1 and s[i] < qval:
            loss_a[:] = loss_a - Ca * np.exp(qval - s[i]) * x[i, :]
            n_anom += 1
        elif lbl == 0 and s[i] >= qval:
            loss_n[:] = loss_n + Cn * np.exp(s[i] - qval) * x[i, :]
            n_noml += 1
        else:
            # no loss
            pass
    dl_dw = (loss_a / max(1, n_anom)) + (loss_n / max(1, n_noml))
    return dl_dw


# constrOptim(theta, f, grad, ui, ci, hessian = FALSE)
# ui %*% theta - ci >= 0
#
# iterative gradient loss:
#   \sum C * l(w; qval, (x,y))
def iter_grad_loss(w, xi, yi, qval, Ca, Cn):
    loss_a = 0  # loss w.r.t w for anomalies
    loss_n = 0  # loss w.r.t w for nominals
    n_anom = 0
    n_noml = 0
    vals = xi.dot(w)
    for i in range(len(yi)):
        if yi[i] == 1 and vals[i] < qval:
            loss_a = loss_a + Ca * np.exp(qval - vals[i])
            n_anom += 1
        elif yi[i] == 0 and vals[i] >= qval:
            loss_n = loss_n + Cn * np.exp(vals[i] - qval)
            n_noml += 1
        else:
            # no loss
            pass
    loss = (loss_a / max(1, n_anom)) + (loss_n / max(1, n_noml))
    return loss


def weight_update_iter_grad(x, y, hf,
                            Ca=1.0, Cn=1.0, Cx=1.0, topK=30, max_iters=500):
    opt_success = False
    k = 0
    delta = 1e-6
    m = ncol(x)

    w = rep(1.0 / m, m)
    s = x.dot(w)
    qval = quantile(s, (1.0 - (topK*1.0/float(nrow(x))))*100.0)
    loss = iter_grad_loss(w, x, y, qval, Ca, Cn)
    loss_old = 0

    while not opt_success and k <= max_iters:
        k += 1
        loss_old = loss
        gam = 1e-4  # 1/sqrt(k)
        try:
            w_grad = get_iter_grad_loss_vector(x, y, hf, qval, s, Ca, Cn)
            w = w - gam * w_grad
            w = np.array([v if v > 0 else 0 for v in w])  # truncate values such that w_i >= 0
            if np.sum(w) > 0:
                w = w / sum(w)
            else:
                # if all weights are zero, make them uniform ...
                w = rep(1.0 / m, m)
            s = x.dot(w)
            qval = quantile(s, (1.0 - (topK*1.0/float(nrow(x))))*100.0)
            loss = iter_grad_loss(w, x, y, qval, Ca, Cn)
            if np.abs(loss - loss_old) <= delta:
                opt_success = True
            # print(sprintf("Iter: %d; old loss: %f; loss: %f; del_loss: %f", k, loss_old, loss, abs(loss - loss_old)))
        except BaseException, e:
            logger.warning("Optimization Err at try %d : %s" % (k, str(e)))

    if opt_success:
        soln = IterGradSolution(w=w, success=True, tries=k,
                                loss=loss, loss_old=loss_old,
                                delta=abs(loss - loss_old))
    else:
        logger.debug("Did not converge in %d iterations..." % k)
        soln = IterGradSolution(w=w, success=False, tries=k,
                                loss=loss, loss_old=loss_old,
                                delta=abs(loss - loss_old))
    return soln
