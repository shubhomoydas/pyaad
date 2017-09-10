import numpy as np
from r_support import *


def if_aad_loss_linear(w, xi, yi, qval, in_constr_set=None, x_tau=None, Ca=1.0, Cn=1.0, Cx=1.0,
                       withprior=False, w_prior=None, sigma2=1.0):
    """
    Computes AAD loss:
        for square_slack:
            ( score_loss + 1/(2*sigma2) * (w - w_prior)^2 )
        else:
            ( score_loss + 1/(2*sigma2) * (w - w_prior)^2 )

    :param w: numpy.array
        parameter vector with both weights and slack variables
    :param xi: csr_matrix
    :param yi: numpy.array
    :param qval: float
        tau-th quantile value
    :param Ca: float
    :param Cn: float
    :param Cx: float
    :param withprior: boolean
    :param w_prior: numpy.array
    :param w_old: numpy.array
    :param sigma2: float
    :param square_slack: boolean
    :return:
    """
    s = xi.dot(w)

    loss_a = 0  # loss w.r.t w for anomalies
    loss_n = 0  # loss w.r.t w for nominals
    n_anom = 0
    n_noml = 0
    tau_rel_loss = None
    if x_tau is not None:
        tau_rel_loss = x_tau.dot(w)
    for i in range(len(yi)):
        lbl = yi[i]
        if lbl == 1 and s[i] < qval:
            loss_a += Ca * (qval - s[i])
            n_anom += 1
        elif lbl == 0 and s[i] >= qval:
            loss_n += Cn * (s[i] - qval)
            n_noml += 1
        else:
            # no loss
            pass

        if tau_rel_loss is not None and (in_constr_set is None or in_constr_set[i] == 1):
            # TODO: Test this code.
            # add loss relative to tau-th ranked instance
            # loss =
            #   Cx * (x_tau - xi).w  if yi = 1 and (x_tau - xi).w > 0
            #   Cx * (xi - x_tau).w  if y1 = 0 and (xi - x_tau).w > 0
            tau_val = tau_rel_loss[0]
            if lbl == 1 and s[i] < tau_val:
                loss_a += Cx * (tau_val - s[i])
            elif lbl == 0 and s[i] >= tau_val:
                loss_n += Cx * (s[i] - tau_val)
            else:
                # no loss
                pass

    loss = (loss_a / max(1, n_anom)) + (loss_n / max(1, n_noml))

    if withprior and w_prior is not None:
        w_diff = w - w_prior
        loss += (1 / (2 * sigma2)) * (w_diff.dot(w_diff))

    return loss


def if_aad_loss_gradient_linear(w, xi, yi, qval, in_constr_set=None, x_tau=None, Ca=1.0, Cn=1.0, Cx=1.0,
                                withprior=False, w_prior=None, sigma2=1.0):
    """
    Computes jacobian of AAD loss:
        for square_slack:
            jacobian( score_loss + 1/(2*sigma2) * (w - w_prior)^2 )
        else:
            jacobian( score_loss + 1/(2*sigma2) * (w - w_prior)^2 )
    """

    m = ncol(xi)

    grad = np.zeros(m, dtype=float)

    s = xi.dot(w)

    loss_a = rep(0, m)  # the derivative of loss w.r.t w for anomalies
    loss_n = rep(0, m)  # the derivative of loss w.r.t w for nominals
    n_anom = 0
    n_noml = 0
    tau_score = None
    if x_tau is not None:
        tau_score = x_tau.dot(w)
    for i in range(len(yi)):
        lbl = yi[i]
        if lbl == 1 and s[i] < qval:
            loss_a[:] = loss_a - Ca * xi[i, :]
            n_anom += 1
        elif lbl == 0 and s[i] >= qval:
            loss_n[:] = loss_n + Cn * xi[i, :]
            n_noml += 1
        else:
            # no loss
            pass

        # add loss-gradient relative to tau-th ranked instance
        if x_tau is not None and (in_constr_set is None or in_constr_set[i] == 1):
            # TODO: Test this code.
            # add loss-gradient relative to tau-th ranked instance
            # loss =
            #   Cx * (x_tau - xi).w  if yi = 1 and (x_tau - xi).w > 0
            #   Cx * (xi - x_tau).w  if y1 = 0 and (xi - x_tau).w > 0
            # loss_gradient =
            #   Cx * (x_tau - xi)  if yi = 1 and (x_tau - xi).w > 0
            #   Cx * (xi - x_tau)  if y1 = 0 and (xi - x_tau).w > 0
            tau_val = tau_score[0]
            if lbl == 1 and s[i] < tau_val:
                loss_a[:] = loss_a + Cx * (x_tau - xi[i, :])
            elif lbl == 0 and s[i] >= tau_val:
                loss_n[:] = loss_n + Cx * (xi[i, :] - x_tau)
            else:
                # no loss
                pass

    grad[0:m] = (loss_a / max(1, n_anom)) + (loss_n / max(1, n_noml))

    if withprior and w_prior is not None:
        w_diff = w - w_prior
        grad[0:m] += (1 / sigma2) * w_diff

    return grad


def if_aad_loss_exp(w, xi, yi, qval, in_constr_set=None, x_tau=None, Ca=1.0, Cn=1.0, Cx=1.0,
                    withprior=False, w_prior=None, sigma2=1.0):
    """
    Computes AAD loss:
        for square_slack:
            ( score_loss + 1/(2*sigma2) * (w - w_prior)^2 )
        else:
            ( score_loss + 1/(2*sigma2) * (w - w_prior)^2 )

    :param w: numpy.array
        parameter vector with both weights and slack variables
    :param xi: csr_matrix
    :param yi: numpy.array
    :param qval: float
        tau-th quantile value
    :param in_constr_set: list of int
        indicators 0/1 whether to include in constraint set or not 
    :param Ca: float
    :param Cn: float
    :param Cx: float
    :param withprior: boolean
    :param w_prior: numpy.array
    :param w_old: numpy.array
    :param sigma2: float
    :param square_slack: boolean
    :return:
    """
    loss_a = 0  # loss w.r.t w for anomalies
    loss_n = 0  # loss w.r.t w for nominals
    n_anom = 0
    n_noml = 0
    vals = xi.dot(w)
    tau_rel_loss = None
    if x_tau is not None:
        tau_rel_loss = x_tau.dot(w)
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

        if tau_rel_loss is not None and (in_constr_set is None or in_constr_set[i] == 1):
            # add loss relative to tau-th ranked instance
            # loss =
            #   Cx * (x_tau - xi).w  if yi = 1 and (x_tau - xi).w > 0
            #   Cx * (xi - x_tau).w  if y1 = 0 and (xi - x_tau).w > 0
            tau_val = tau_rel_loss[0]
            if yi[i] == 1 and vals[i] < tau_val:
                loss_a += Cx * (tau_val - vals[i])
            elif yi[i] == 0 and vals[i] >= tau_val:
                loss_n += Cx * (vals[i] - tau_val)
            else:
                # no loss
                pass

    loss = (loss_a / max(1, n_anom)) + (loss_n / max(1, n_noml))

    if withprior and w_prior is not None:
        w_diff = w - w_prior
        loss += (1 / (2 * sigma2)) * (w_diff.dot(w_diff))

    return loss


def if_aad_loss_gradient_exp(w, xi, yi, qval, in_constr_set=None, x_tau=None, Ca=1.0, Cn=1.0, Cx=1.0,
                             withprior=False, w_prior=None, sigma2=1.0):
    """
    Computes jacobian of AAD loss:
        for square_slack:
            jacobian( score_loss + 1/(2*sigma2) * (w - w_prior)^2 )
        else:
            jacobian( score_loss + 1/(2*sigma2) * (w - w_prior)^2 )
    """
    vals = xi.dot(w)
    m = ncol(xi)
    loss_a = rep(0, m)  # the derivative of loss w.r.t w for anomalies
    loss_n = rep(0, m)  # the derivative of loss w.r.t w for nominals
    n_anom = 0
    n_noml = 0
    tau_score = None
    if x_tau is not None:
        tau_score = x_tau.dot(w)
    for i in range(len(yi)):
        lbl = yi[i]
        if lbl == 1 and vals[i] < qval:
            exp_diff = np.minimum(np.exp(qval - vals[i]), 1000)  # element-wise
            # exp_diff = np.exp(qval - vals[i])
            loss_a[:] = loss_a - Ca * exp_diff * xi[i, :]
            n_anom += 1
        elif lbl == 0 and vals[i] >= qval:
            exp_diff = np.minimum(np.exp(vals[i] - qval), 1000)  # element-wise
            # exp_diff = np.exp(vals[i] - qval)
            loss_n[:] = loss_n + Cn * exp_diff * xi[i, :]
            n_noml += 1
        else:
            # no loss
            pass

        # add loss-gradient relative to tau-th ranked instance
        if x_tau is not None and (in_constr_set is None or in_constr_set[i] == 1):
            # add loss-gradient relative to tau-th ranked instance
            # loss =
            #   Cx * (x_tau - xi).w  if yi = 1 and (x_tau - xi).w > 0
            #   Cx * (xi - x_tau).w  if y1 = 0 and (xi - x_tau).w > 0
            # loss_gradient =
            #   Cx * (x_tau - xi)  if yi = 1 and (x_tau - xi).w > 0
            #   Cx * (xi - x_tau)  if y1 = 0 and (xi - x_tau).w > 0
            tau_val = tau_score[0]
            if lbl == 1 and vals[i] < tau_val:
                loss_a[:] = loss_a + Cx * (x_tau - xi[i, :])
            elif lbl == 0 and vals[i] >= tau_val:
                loss_n[:] = loss_n + Cx * (xi[i, :] - x_tau)
            else:
                # no loss
                pass

    dl_dw = (loss_a / max(1, n_anom)) + (loss_n / max(1, n_noml))

    if withprior and w_prior is not None:
        w_diff = w - w_prior
        dl_dw[0:m] += (1 / sigma2) * w_diff

    return dl_dw

