#!/bin/python2.7

# This is a direct 
# port of Stephen Boyds maltab algorithm found at
# http://stanford.edu/~boyd/papers/admm/lasso/group_lasso.html

import numpy as np
import numpy.linalg as npl
import numpy.random as npr
from scipy import linalg as LA

import time


def factor(A, rho):
    """ Returns the factorisation of rho*I + At*A"""
    m, n = A.shape;
    At = np.transpose(A)

    if m >= n:    # if skinny
        L = npl.cholesky(np.dot(At, A) + rho * np.eye(n));
    else:         # if fat
        L = npl.cholesky(np.eye(m) + (1 / float(rho)) * np.dot(A, At));

    U = np.transpose(L)

    return (L, U)


def shrinkage(x, kappa):
    """Computes the shrinkage of x"""
    z = max(1 - kappa / LA.norm(x, 2), 0) * x
    return z


def objective(A, b, lmbd, cum_part, x, z):
    """Calculates and returns the objective for the given lasso function"""
    obj = 0
    start_ind = 0
    for i in range(len(cum_part)):
        sel = z[start_ind: cum_part[i]]
        if sel.size:
            obj += LA.norm(sel)
        start_ind = cum_part[i] + 1
        
    p = (1.0/2) * (np.sum(np.power((np.dot(A, x) - b), 2))) + \
        lmbd * obj
    return p

def init_history():
    """ Initialises a history data structure for tracking values"""
    history = {}
    history["objval"] = list()
    history["r_norm"] = list()
    history["s_norm"] = list()
    history["eps_pri"] = list()
    history["eps_dual"] = list()

    return history

def lasso(A, b, lmbd, p, rho, alpha):
    """
    Solves the lasso problem:
         minimize 1/2*|| Ax - b ||_2^2 + lmbd * sum(norm(x_i))
    via the ADMM method.

    Arguments:
    rho -- the augmented Lagrangian parameter (float)
    alpha -- the over-relaxation parameter (typical values for alpha are
    between 1.0 and 1.8) (float)
    A, b, lmdb -- the function inputs
    p -- a K-element vecor giving the block sizes n_i, so that x_i is in 
    R^{n_i}
    
    Returns:
    x -- the optimal solution in the form of column vector
    hist -- a dictionary of the history values
    """

    # Global constants and defaults
    history = init_history()
    t_start = time.clock()

    QUIET    = False   # Set to False to print logging values
    MAX_ITER = 1000    # Maximum number of update iterations
    ABSTOL   = 1e-4    # Aboslute Tolerance > 0 
    RELTOL   = 1e-2    # Relative Tolerance 

    
    # Data Preprocessing
    m, n = A.shape;
    At = np.transpose(A)
    Atb = np.dot(At, b)

    # Check that sum(p) = total number of elements in x
    if sum(p) != n:
        raise InputError('Invalid Partition')

    cum_part = np.cumsum(p) # Cumulative partition

    
    # ADMM Solver
    x = np.zeros(n)
    z = np.zeros(n)
    u = np.zeros(n)

    (L, U) = factor(A, rho) # cache the factorization
    
    if not QUIET:
        print "%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n" % \
            ("iter", "r norm", "eps pri", "s norm", "eps dual", "objective")

    for k in range(MAX_ITER):
        # x-update
        q = Atb + rho * (z - u) # temporary value
        
        if( m >= n ): # if skinny
            x = LA.solve_triangular(U, LA.solve_triangular(L, q, lower=True))
        else: # if fat
            x = q/rho - np.dot(At, \
                    LA.solve_triangular(U, LA.solve_triangular(L, \
                            np.dot(A, q), lower=True))) / (rho ** 2)
            
        # z-update
        zold = np.copy(z);
        s_ind = 0
        x_hat = alpha * x + (1 - alpha) * zold;
        for i in range(len(p)):
            c = cum_part[i]
            temp = x_hat[s_ind: c] + u[s_ind: c]
            if not temp.size == 0:
                z[s_ind: c] =  shrinkage(temp, lmbd/rho)
            s_ind = c + 1
        
        # u-update
        u = np.copy(u) + x_hat - z;

        # diagnostics, reporting, termination checks
        objval = objective(A, b, lmbd, cum_part, x, z);
        r_norm  = npl.norm(x - z);
        s_norm  = npl.norm(-1 * rho * (z - zold));
        eps_pri = np.sqrt(n) * ABSTOL + RELTOL * np.maximum(npl.norm(x),\
                                                            npl.norm(-z));
        eps_dual= np.sqrt(n) * ABSTOL + RELTOL * npl.norm(rho*u);

        history["objval"].append(objval);
        history["r_norm"].append(r_norm)
        history["s_norm"].append(s_norm)
        history["eps_pri"].append(eps_pri)
        history["eps_dual"].append(eps_dual)

        if not QUIET:
            print "%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n" % \
                (k, r_norm, eps_pri, s_norm, eps_dual, objval);

        if r_norm < eps_pri and s_norm < eps_dual:
            break;


    if not QUIET:
        total_time = (time.clock() - t_start)
        print "Total time taken: ", total_time

    return (x, history)
