import numpy as np
from scipy.optimize import minimize
from numba import njit, jit, generated_jit

from miscellaneous.execution_time_decorator import timeit

@njit(fastmath=True, cache=True, nogil=True)
def pdf(X: np.ndarray, params_k: np.ndarray):
    """computes for every data point X = (x1, x2) = (tau, deltaV) the value os its probability density, given the parameters for a certain cluster
    """
    exp_x = X[:, 0]
    exp_mu = params_k[1]
    
    norm_x = X[:, 1]
    norm_mu = params_k[2]
    norm_sigma = params_k[3]

    exp_pdf = 1/exp_mu * np.exp(-exp_x/exp_mu)
    norm_pdf = 1/(norm_sigma * np.sqrt(2 * np.pi)) * np.exp(-1/2 * ((norm_x-norm_mu)/norm_sigma)**2)
    return exp_pdf * norm_pdf

@njit(fastmath=True, cache=True, nogil=True)
def gamma(X: np.ndarray, params: np.ndarray, k: int) -> np.ndarray:
    """for a specific cluster
    """
    nominator = params[4*k] * pdf(X, params[k * 4:(k + 1) * 4])
    K = len(params)//4
    N = X.shape[0]
    denominator = np.zeros(N)
    for k_ in range(K):
        denominator +=  params[4*k_] * pdf(X, params[k_ * 4:(k_ + 1) * 4])
    #denominator = np.sum([ for k_ in range(K)], axis=0)
    return nominator/denominator

@njit(fastmath=True, cache=True, nogil=True)
def E_step(X: np.ndarray, params: np.ndarray):
    """calculate_responsibilities
    """
    K = len(params)//4
    N = X.shape[0]
    gammas = np.empty((K, N))
    for k in range(K):
        gammas[k, :] = gamma(X, params, k)
    return gammas

@njit(fastmath=True, cache=True, nogil=True)
def cdll(params: np.ndarray, X: np.ndarray, gammas: np.ndarray) -> np.ndarray:
    K = len(params)//4
    N = X.shape[0]
    pdfs = np.empty((K, N))
    mix_coeffs = params[::4]
    for k in range(K):
        pdfs[k, :] = pdf(X, params[k * 4:(k + 1) * 4])
    return -np.sum( gammas * (np.log(mix_coeffs) + np.log(pdfs.T)).T) # K x N * (K x N).T = K | minus sign because we use a minimization algorithm in M-Step


@njit(fastmath=True, cache=True, nogil=True)
def has_converged(X, params_new, params_old, em_tol):
    ll_new = cdll(params_new, X, E_step(X, params_new)) 
    ll_old = cdll(params_old, X, E_step(X, params_old)) 
    diff = np.abs(ll_new-ll_old)
    if diff < em_tol:
        return True
    else:
        return False

@njit(fastmath =True, cache=True)
def compute_loglikelihood(params, X):
    """for external uses
    """
    gammas = E_step(X, params)
    return -cdll(params, X, gammas)

@njit(fastmath=True, cache=True, nogil=True)
def constraint_1(x):
    return np.sum(x[::4])-1
    

def create_constr_dict(abs_tol_params, min_mix_coef):
    constr = ({"type": "eq", "fun": lambda x: np.sum(x[::4])-1},                        # s.t.: sum of mixture coefficients must be 1
              {"type": "ineq", "fun": lambda x: x - abs_tol_params},      # parameters must be positive
              {"type": "ineq", "fun": lambda x: x[::4] - min_mix_coef})   # minimum number of points per cluster => reduces singularities
    return constr

def M_step(X, params_array, gammas, constraints, minimizer_options):
    """minimize log likelihood    
    """           
    minimizer = minimize(
        cdll, 
        x0=params_array, 
        constraints=constraints,
        args = (X, gammas), 
        method="SLSQP",
        options = {"maxiter": 100}
    )
    return minimizer.x


@timeit
def run_EM(X, init_params, em_tol=1e-5, max_iter=1000, min_mix_coef=0.02, abs_tol_params=1e-8, minimizer_options={}):
    """compared to `run_EM` it uses also njit constraints in `M_step`
    """
    constraints = create_constr_dict(abs_tol_params, min_mix_coef)
    i=1
    params = init_params
    while(i < max_iter):
        gammas = E_step(X, params)
        new_params = M_step(X, params, gammas, constraints, minimizer_options)
        if has_converged(X, new_params, params, em_tol):
            return new_params, i

        params = new_params
        i+=1
    return new_params, i

