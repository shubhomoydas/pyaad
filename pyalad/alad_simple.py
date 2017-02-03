from loda_support import *


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
