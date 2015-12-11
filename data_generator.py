# Python 2.7

import random as rd
import scipy.sparse
from math import sqrt
import numpy as np
from numpy import linalg as LA


def data_gen():
  """Returns (A, b, lambda) for a sample lasso problem. Random data. """
  rd.seed()

  m = 1500   # number of examples
  n = 5000   # number of features
  p = 100.0/n  # sparsity of density

  x0 = scipy.sparse.rand(n, 1, density=p).toarray()
  A = np.random.randn(m, n)
  A2 = np.square(A)

  temp = np.reciprocal(np.transpose(np.sqrt(A2.sum(axis=0))))
  dia = scipy.sparse.spdiags(temp, 0, n, n).toarray()
  A = np.dot(A, dia)  # normalise columns

  b = np.dot(A, x0) + sqrt(0.001) * np.random.randn(m, 1)

  lambda_max = LA.norm(np.dot(np.transpose(A), b), np.inf)
  lmbda = 0.1 * lambda_max
  
  return (A, b, lmbda)

if __name__ == "__main__":
  data_gen()
