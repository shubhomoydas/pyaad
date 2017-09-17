from loda_support import *


"""
python pyalad/test_optimization.py --op=constr_optim --debug --log_file=/Users/moy/work/temp/pyalad.log
python pyalad/test_optimization.py --op=constr_optim_prior --debug --log_file=/Users/moy/work/temp/pyalad.log
python pyalad/test_optimization.py --op=cvxopt --debug --log_file=/Users/moy/work/temp/pyalad.log
"""


def optim_prob_nocedal_example_16_4():
    """Nocedal Example 16.4 (page 475)

    True solution: [1.4, 1.7]

    Sample code for scipy.optimize.minimize:
    http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize

    :return:
    """
    def f(z):
        return (z[0] - 1) ** 2 + (z[1] - 2.5) ** 2

    def g(z):
        return np.array([2 * (z[0] - 1), 2 * (z[1] - 2.5)], float)

    bounds = ((0, None), (0, None))

    if True:
        # test with constraints
        ui = np.array([[1, -2], [-1, -2], [-1, 2]], dtype=float)
        ci = np.array([-2, -6, -2], dtype=float)
    else:
        # test without constraints
        ui = None
        ci = None

    x0 = np.array([2, 0], dtype=float)

    x, success = constr_optim(x0, f, grad=g, ui=ui, ci=ci, method="SLSQP",
                              bounds=bounds, debug=True)
    logger.debug(x)
    logger.debug(f(x))
    #print ui.dot(x) - ci
    #print f([1.4, 1.7])  # true solution


def optim_prob_nocedal_example_16_4_with_prior():
    """Nocedal Example 16.4 (page 475) with an added prior on x

    :return:
    """
    def f(z, withprior=False, prior=None, l=1.0):
        if not withprior:
            return (z[0] - 1) ** 2 + (z[1] - 2.5) ** 2
        else:
            return (z[0] - 1) ** 2 + (z[1] - 2.5) ** 2 + l * (z - prior).dot(z - prior)

    def g(z, withprior=False, prior=None, l=1.0):
        if not withprior:
            return np.array([2 * (z[0] - 1), 2 * (z[1] - 2.5)], float)
        else:
            return np.array([2 * (z[0] - 1) + 2 * l * (z[0] - prior[0]),
                             2 * (z[1] - 2.5) + 2 * l * (z[1] - prior[1])], float)

    def argswrapper(**kwargs):
        return (kwargs.get('withprior', False),
                kwargs.get('prior', None),
                kwargs.get('l', 1.0))

    bounds = ((0, None), (0, None))

    if True:
        # test with constraints
        ui = np.array([[1, -2], [-1, -2], [-1, 2]], dtype=float)
        ci = np.array([-2, -6, -2], dtype=float)
    else:
        # test without constraints
        ui = None
        ci = None

    x0 = np.array([2, 0], dtype=float)
    withprior = True
    prior = np.array([1.2, 1.5], float)

    x, success = constr_optim(x0, f, grad=g, ui=ui, ci=ci, method="SLSQP", bounds=bounds,
                              args=argswrapper(withprior=withprior, prior=prior, l=1.0), debug=True)
    logger.debug(x)
    logger.debug(f(x))
    #print ui.dot(x) - ci
    #print f([1.4, 1.7])  # true solution


def cvxopt_prob_nocedal_example_16_4():
    """Nocedal Example 16.4 (page 475)

    True solution: [1.4, 1.7]

    :return:
    """
    def f(z):
        return (z[0] - 1) ** 2 + (z[1] - 2.5) ** 2

    def g(z):
        return np.array([2 * (z[0] - 1), 2 * (z[1] - 2.5)], float)

    def hessian(z):
        return np.array([2, 0, 0, 2], dtype=float).reshape((2, 2))

    bounds = ((0, None), (0, None))

    if True:
        # test with constraints
        ui = np.array([[1, -2], [-1, -2], [-1, 2]], dtype=float)
        ci = np.array([-2, -6, -2], dtype=float)
    else:
        # test without constraints
        ui = None
        ci = None

    x0 = np.array([2, 0], dtype=float)
    x = cvx_optim(x0, f, g, hessian=hessian, ui=ui, ci=ci, a=None, b=None,
                  debug=False, bounds=bounds, args=None)
    logger.debug("Solution:")
    logger.debug(x)


def cvxopt_prob_1d_no_equality():
    """

    minimize x^2 - 2x + 1
    s.t.
        x <= 2

    :return:
    """
    def f(z):
        return z[0]**2 - 2*z[0] + 1

    def g(z):
        return np.array([2.*z[0] - 2.])

    def hessian(z):
        return np.array([2], dtype=float).reshape((1, 1))

    # test with constraints
    # x <= 2  =>  -x >= -2
    ui = np.array([-1], dtype=float).reshape(1, 1)
    ci = np.array([-2], dtype=float)

    kktsolver = get_kktsolver_no_equality_constraints(fn=f, ui=ui, grad=g, hessian=hessian)

    x0 = np.array([0], dtype=float)
    x = cvx_optim(x0, f, g, hessian=hessian, ui=ui, ci=ci, a=None, b=None,
                  debug=False, bounds=[(None, None)], kktsolver=kktsolver, args=None)
    logger.debug("Solution:")
    logger.debug(x)


def cvxopt_prob_2d_no_equality():
    """

    minimize x1**2 + x2**2 - 2(x1 + x2) + 1
    s.t.
          5 * x1 + 2 * x2 >= 0
        1.5 * x1 + 3 * x2 >= 0

    :return:
    """
    def f(z):
        return z[0]**2 + z[1]**2 - 2*(z[0] + z[1]) + 1

    def g(z):
        return np.array([2.*z[0] - 2., 2.*z[1] - 2.])

    def hessian(z):
        return np.diag([2., 2.])

    ui = np.array([5, 2, 1.5, 3], dtype=float).reshape(2, 2)
    ci = np.array([0, 0], dtype=float)

    kktsolver = get_kktsolver_no_equality_constraints(fn=f, ui=ui, grad=g, hessian=hessian)

    x0 = np.array([0, 0], dtype=float)
    x = cvx_optim(x0, f, g, hessian=hessian, ui=ui, ci=ci, a=None, b=None,
                  debug=False, bounds=[(None, None), (None, None)], kktsolver=kktsolver, args=None)
    logger.debug("Solution:")
    logger.debug(x)


if __name__ == '__main__':

    args = get_command_args(debug=False, debug_args=None)
    configure_logger(args)

    print "Configured logger...%s" % (args.log_file,)

    if args.op == "constr_optim":
        optim_prob_nocedal_example_16_4()
    elif args.op == "constr_optim_prior":
        optim_prob_nocedal_example_16_4_with_prior()
    elif args.op == "cvxopt":
        cvxopt_prob_nocedal_example_16_4()
    elif args.op == "1d":
        cvxopt_prob_1d_no_equality()
    elif args.op == "2d":
        cvxopt_prob_2d_no_equality()
    else:
        raise ValueError("Invaid operation %s" % (args.op,))

