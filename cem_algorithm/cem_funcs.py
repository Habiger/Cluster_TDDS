from functools import cache
import numpy as np
from scipy.optimize import minimize
from numba import njit

from miscellaneous.execution_time_decorator import timeit

@njit(fastmath=True, cache=True)
def norm_pdf(x, mu, sigma):
    return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-1/2 * ((x-mu)/sigma)**2)

@njit(fastmath=True, cache=True)
def exp_pdf(x, mu):
    #TODO implement as log(exp) distribution -> original ??
    return 1/mu * np.exp(-x/mu)

@njit(fastmath=True, cache=True)
def p(X: np.ndarray, params_k: np.ndarray):
    """computes for every data point X = (x1, x2) = (tau, deltaV) the value os its probability density, given the parameters for a certain cluster
    """
    return (exp_pdf(X[:, 0], params_k[1]) * norm_pdf(X[:,1], params_k[2], params_k[3]))

@njit(fastmath=True, cache=True)
def pdfs(X: np.ndarray, params: np.ndarray): # new function to compute the pdfs for all clusters for every point
    N = X.shape[0]
    K = params.shape[0]//4
    pdfs_ = np.empty((N, K))
    for k in range(K):
        pdfs_[:, k] = p(X, params[4*k:4*(k+1)])
    return pdfs_

@njit(fastmath=True, cache=True)
def gamma(X: np.ndarray, params: np.ndarray, k: int) -> np.ndarray:
    """for a specific cluster
    """
    nominator = params[4*k] * p(X, params[k * 4:(k + 1) * 4])
    K = len(params)//4
    N = X.shape[0]
    denominator = np.zeros(N)
    for k_ in range(K):
        denominator +=  params[4*k_] * p(X, params[k_ * 4:(k_ + 1) * 4])
    #denominator = np.sum([ for k_ in range(K)], axis=0)
    return nominator/denominator

@njit(fastmath=True, cache=True)
def E_step(X: np.ndarray, params: np.ndarray):
    """calculate_responsibilities
    """
    K = len(params)//4
    N = X.shape[0]
    gammas = np.empty((K, N))
    for k in range(K):
        gammas[k, :] = gamma(X, params, k)
    return gammas

@njit(fastmath=True, cache=True)
def cdll2(params: np.ndarray, X: np.ndarray, gammas: np.ndarray) -> np.ndarray:
    K = len(params)//4
    N = X.shape[0]
    pdfs_ = np.empty((K, N))
    mix_coeffs = params[::4]
    for k in range(K):
        pdfs_[k, :] = p(X, params[k * 4:(k + 1) * 4])
    return -np.sum( gammas * (np.log(mix_coeffs) + np.log(pdfs_.T)).T) # K x N * (K x N).T = K


def M_step(X, params_array, gammas, min_mix_coef):
    """minimize log likelihood
    """
    abstol = 1e-06
    constr = ({"type": "eq", "fun": lambda x: np.sum(x[::4])-1},  # nebenbedingung
              {"type": "ineq", "fun": lambda x: x - abstol},      # parameters must be positive
              {"type": "ineq", "fun": lambda x: x[::4] - min_mix_coef})   # minimum number of points per cluster
    minimizer = minimize(
        cdll2, 
        x0=params_array, 
        constraints=constr,
        args = (X, gammas), 
        method="SLSQP",
        options = {
            "maxiter": 300,
            "eps": 1e-10,
            #"ftol": 1e-03
        } 
    )
    return minimizer.x

@timeit
def run_EM(X, params, em_tol=1e-8, max_iter = 300, min_mix_coef=0.05):
    i=1
    while(i < max_iter):
        gammas = E_step(X, params)
        new_params = M_step(X, params, gammas, min_mix_coef)
        if has_converged(X, new_params, params, em_tol):
            return new_params, i
        params = new_params
        i+=1
    return new_params, i

@njit(fastmath=True, cache=True)
def has_converged(X, params_new, params_old, em_tol):
    ll_new = cdll2(params_new, X, E_step(X, params_new)) 
    ll_old = cdll2(params_old, X, E_step(X, params_old)) 
    diff = np.abs(ll_new-ll_old)
    #diff = sum(np.abs(params_new - params_old))
    if diff < em_tol:
        return True
    else:
        return False