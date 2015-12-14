# Python 2.7

import random as rd
import scipy.sparse
from math import sqrt
import numpy as np
from numpy import linalg as LA


def data_gen():
  """Returns (A, b, lambda, partition) for a sample group-lasso problem.
  Random data. """
  rd.seed()

  m = 1500   # number of examples
  K = 200    # number of blocks
  partition = np.random.randint(1, high=50, size=K)

  n = np.sum(partition)
  p = 100.0/n
  print "n is: ", n
  print "p is: ", p

  # generate block sparse solution vector
  x = np.zeros(n, dtype=np.int)
  start_ind = 0
  cum_part = np.cumsum(partition)
  for i in range(K):
    if rd.random() < p:
      x[start_ind: cum_part[i]] = np.random.randn(partition[i])
    start_ind = cum_part[i]

  # generate random data matrix
  A = np.random.randn(m, n)

  # normalise columns of A
  A2 = np.square(A)
  temp = np.reciprocal(np.transpose(np.sqrt(A2.sum(axis=0))))
  A = np.dot(A, scipy.sparse.spdiags(temp, 0, n, n).toarray())

  # generate measurement b with noise
  b = np.dot(A, x) + np.sqrt(0.001) * np.random.randn(m)
  
  # lambda max
  start_ind = 0
  lambdas = np.zeros(K)
  for i in range(K):
    temp = np.dot(np.transpose(A[:, start_ind:cum_part[i]]), b)
    lambdas[i] = LA.norm(temp)
    start_ind = cum_part[i]
  lambda_max = max(lambdas)

  # regularization parameter
  lmbd = 0.1 * lambda_max

  return (A, b, lmbd, partition) 


if __name__ == "__main__":
  data_gen()
