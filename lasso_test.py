# Python 2.7

# Script for running basic analysis on the pyadmm lasso algorithm

import data_generator as dg
import pyadmm
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt


def lasso_func(A, x, b, lmbda):
    """Outputs result of the lasso function on given input"""
    
    q = (1.0/2) * (LA.norm(np.dot(A, x) - b, 2) ** 2) + lmbda * LA.norm(x, 1)
    return q



def lasso_test():
    """Runs sample"""
    
    (A, b, l) = dg.data_gen()
    (x, hist) = pyadmm.lasso(A, b, l, 1.0, 1.0)

    K = len(hist['objval'])
    x = np.arange(K)
    
    # Plot the objective values
    fig = plt.figure()
    y = hist['objval']
    ax = fig.add_subplot(111)
    ax.plot(x, y)
    plt.xlabel('iter(k)')
    plt.ylabel('f(x^k) + g(z^k)')
    plt.show()

    # Plot norms
    fig2 = plt.figure()
    bx = fig2.add_subplot(211)
    bx.semilogy(x, np.maximum(hist['r_norm'], 1e-8))
    bx.semilogy(x, hist['eps_pri'], 'r--')
    plt.ylabel('||r||_2')

    cx = fig2.add_subplot(212)
    cx.semilogy(x, np.maximum(hist['s_norm'], 1e-8))
    cx.semilogy(x, hist['eps_dual'], 'r--')
    plt.ylabel('||s||_2')
    plt.xlabel('iter(k)')
    plt.show()


if __name__ == "__main__":
    lasso_test()

    
    
