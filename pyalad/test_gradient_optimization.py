
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import logging
from r_support import *
from app_globals import *
from data_plotter import *

from optimization import *


"""

pythonw pyalad/test_gradient_optimization.py

"""

logger = logging.getLogger(__name__)


def generate_data(p=11, n=400):
    true_params = np.random.uniform(low=0.0, high=1.0, size=p+1)

    x = np.sort(np.random.uniform(low=-1.0, high=1.0, size=n))
    X = np.zeros(shape=(n, p+1), dtype=float)
    X[:, 0] = 1.
    for i in xrange(p):
        X[:, i+1] = x ** (i+1)
    # logger.debug("X:\n%s" % str(list(X)))
    e = np.random.normal(loc=0.0, scale=0.2, size=n)

    y_true = X.dot(true_params)
    y = y_true + e
    # logger.debug("y:\n%s" % str(list(y)))

    return X, y, true_params, x, y_true, e


def f(w, x, y):
    loss = np.mean(0.5 * ((x.dot(w) - y) ** 2))
    return loss


def g(w, x, y):
    grad = np.multiply(x, np.transpose([x.dot(w) - y]))
    mean_grad = np.mean(grad, axis=0)
    # logger.debug(mean_grad.shape)
    return mean_grad


def get_loss_grad():
    return f, g


if __name__ == "__main__":
    args = get_command_args(debug=True, debug_args=["--log_file=temp/rmsprop.log", "--debug"])
    configure_logger(args)

    np.random.seed(42)

    X, y, w_true, x, y_true, e = generate_data(p=11, n=400)
    logger.debug("True params:\n%s" % str(list(w_true)))
    logger.debug("Num batches: %d" % get_num_batches(400, 100))

    w0 = np.zeros(len(w_true), dtype=float)
    # w = sgd(w0, X, y, f, g, learning_rate=0.01, batch_size=25, max_epochs=1000)
    # w = sgdRMSPropNestorov(w0, X, y, f, g, learning_rate=0.01, alpha=0.9, ro=0.9, batch_size=25, max_epochs=1000)
    # w = sgdMomentum(w0, X, y, f, g, learning_rate=0.01, alpha=0.9, batch_size=25, max_epochs=1000)
    w = sgdRMSProp(w0, X, y, f, g, learning_rate=0.01, ro=0.9, batch_size=25, max_epochs=1000)
    # w = sgdAdam(w0, X, y, f, g, learning_rate=0.01, ro1=0.9, ro2=0.999, batch_size=25, max_epochs=1000)
    logger.debug("Inferred params:\n%s" % str(list(w)))

    y_pred = X.dot(w)

    pdfpath = "temp/test_data.pdf"
    dp = DataPlotter(pdfpath=pdfpath, rows=1, cols=1)
    pl = dp.get_next_plot()
    pl.plot(x, y_true, 'b-')
    pl.plot(x, y_pred, 'g-')
    pl.plot(x, y, 'r.')
    dp.close()
