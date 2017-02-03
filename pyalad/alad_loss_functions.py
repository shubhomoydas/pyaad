from loda_support import *


class OptimizationContext(object):
    """Stores context information like the number of calls to gradient

    Attributes:
        n_error: int
            The maximum number of times gradient_call() can be called before
            raising ArithmeticException that gradient computation was called too many times
        n_print: int
            Count of calls will be printed every n_print times gradient_call() is called
    """

    def __init__(self, n_error=20000, n_print=10000):
        self.n_error = n_error
        self.n_print = n_print
        self.grad_call_count = 0

    def reset_grad_call_count(self):
        self.grad_call_count = 0

    def gradient_call(self):
        if self.grad_call_count > self.n_error:
            raise ArithmeticError("Too many calls to gradient")
        self.grad_call_count += 1
        if self.n_print > 0 and self.grad_call_count > 0 and np.mod(self.grad_call_count, self.n_print) == 0:
            logger.debug("Gradient called: %d" % self.grad_call_count)


def aatp_slack_loss(theta, xi, yi, qval, Ca=1.0, Cn=1.0, Cx=1.0,
                    withprior=False, w_prior=None, w_old=None, sigma2=1.0, nu=1.0,
                    square_slack=False, optimcontext=None):
    """
    Computes AAD loss:
        for square_slack:
            ( score_loss + 1/(2*sigma2) * (w - w_prior)^2 + Cx*(xi'xi) )
        else:
            ( score_loss + 1/(2*sigma2) * (w - w_prior)^2 + Cx*|xi|) )

    :param theta: numpy.array
    :param xi: numpy.ndarray
    :param yi: numpy.array
    :param qval: float
    :param Ca: float
    :param Cn: float
    :param Cx: float
    :param withprior: boolean
    :param w_prior: numpy.array
    :param w_old: numpy.array
    :param sigma2: float
    :param nu: float
    :param square_slack: boolean
    :param optimcontext: OptimizationContext
    :return:
    """
    m = ncol(xi)
    w = theta[0:m]

    s = xi.dot(w)

    loss_a = 0  # loss w.r.t w for anomalies
    loss_n = 0  # loss w.r.t w for nominals
    n_anom = 0
    n_noml = 0
    for i in range(len(yi)):
        if yi[i] == 1 and s[i] < qval:
            loss_a += Ca * (qval - s[i])
            n_anom += 1
        elif yi[i] == 0 and s[i] >= qval:
            loss_n += Cn * (s[i] - qval)
            n_noml += 1
        else:
            # no loss
            pass
    loss = (loss_a / max(1, n_anom)) + (loss_n / max(1, n_noml))

    if withprior and w_prior is not None:
        w_diff = w - w_prior
        loss += (1 / (2 * sigma2)) * (w_diff.dot(w_diff))

    if len(theta) > m:
        if square_slack:
            slack_loss = theta[m:len(theta)] ** 2
            # logger.debug(slack_loss)
            loss += Cx * np.sum(slack_loss)
        else:
            slack_loss = theta[m:len(theta)]
            # logger.debug(slack_loss)
            loss += Cx * np.sum(slack_loss)
    # logger.debug("Slack Loss: %f" % (loss,))
    return loss


def aatp_slack_loss_gradient(theta, xi, yi, qval, Ca=1.0, Cn=1.0, Cx=1.0,
                             withprior=False, w_prior=None, w_old=None, sigma2=1.0, nu=1.0,
                             square_slack=False, optimcontext=None):
    """
    Computes jacobian of AAD loss:
        for square_slack:
            jacobian( score_loss + 1/(2*sigma2) * (w - w_prior)^2 + Cx*(xi'xi) )
        else:
            jacobian( score_loss + 1/(2*sigma2) * (w - w_prior)^2 + Cx*|xi|) )
    """

    if optimcontext is not None:
        optimcontext.gradient_call()

    m = ncol(xi)
    w = theta[0:m]

    grad = np.zeros(len(theta), dtype=float)

    s = xi.dot(w)

    loss_a = rep(0, m)  # the derivative of loss w.r.t w for anomalies
    loss_n = rep(0, m)  # the derivative of loss w.r.t w for nominals
    n_anom = 0
    n_noml = 0
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
    grad[0:m] = (loss_a / max(1, n_anom)) + (loss_n / max(1, n_noml))

    if withprior and w_prior is not None:
        w_diff = w - w_prior
        grad[0:m] += (1 / sigma2) * w_diff

    if len(theta) > m:
        slackvars = theta[m:len(theta)]
        # logger.debug(slackvars)
        if square_slack:
            grad[m:len(theta)] = 2 * Cx * slackvars
        else:
            grad[m:len(theta)] = Cx
    # logger.debug("grad: \n%s" % str(grad))
    return grad


def aatp_slack_loss_hessian(theta, xi, yi, qval, Ca=1.0, Cn=1.0, Cx=1.0,
                            withprior=False, w_prior=None, w_old=None, sigma2=1.0, nu=1.0,
                            square_slack=False, optimcontext=None):
    """
    Computes hessian of AAD loss:
        for square_slack:
            hessian( score_loss + 1/(2*sigma2) * (w - w_prior)^2 + Cx*(xi'xi) )
        else:
            hessian( score_loss + 1/(2*sigma2) * (w - w_prior)^2 + Cx*|xi|) )
    """
    m = ncol(xi)
    diag = np.zeros(len(theta), dtype=float)
    if withprior and w_prior is not None:
        diag[0:m] = 1./sigma2
    if len(theta) > m:
        if square_slack:
            diag[m:len(theta)] = 2 * Cx
        else:
            diag[m:len(theta)] = 0
    hess = np.diag(diag)
    return hess


