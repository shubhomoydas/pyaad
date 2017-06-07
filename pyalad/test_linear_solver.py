from cvxopt import blas, lapack, solvers
from cvxopt import matrix, spdiag, mul, div, sparse
from cvxopt import spmatrix, sqrt, base

import numpy as np


"""

    2x +  y +  3z = 1
    2x + 6y +  8z = 3
    6x + 8y + 18z = 5

Solution:
     (x, y, z) = ( 3/10, 2/5, 0)

CVXOPT  cvxopt.lapack.gesv(A, B[, ipiv = None]) Solves
    A X = B,
where A and B are real or complex matrices, with A square and nonsingular.
On entry, B contains the right-hand side B; on exit it contains the solution X.

To run:
python pyalad/test_linear_solver.py

"""

tA = np.array([[2, 1, 3],
               [2, 6, 8],
               [6, 8, 18]], dtype=float)
tb = np.array([1, 3, 5], dtype=float)

print tA
print tb

A = matrix(tA)
b = matrix(tb)
print A
print b

x = matrix(b)
print x

lapack.gesv(A, x)
print A * x
print (3./10, 2./5, 0)

