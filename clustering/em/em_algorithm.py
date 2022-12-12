import numpy as np

from scipy.optimize import minimize
from numba import njit

from miscellaneous.execution_time_decorator import timeit


@timeit
def run_EM(
    X: np.ndarray,
    starting_values: np.ndarray,
    em_tol=1e-5,
    max_iter=1000,
    min_mix_coef=0.02,
    abs_tol_params=1e-8,
    minimizer_options={},
) -> tuple[np.ndarray, int]:
    """Implementation of the EM-Algorithm."""
    constraints = create_constraints(abs_tol_params, min_mix_coef)
    mixture_params = starting_values
    i = 1
    while i < max_iter:
        gammas = E_step(X, mixture_params)
        new_mixture_params = M_step(X, mixture_params, gammas, constraints, minimizer_options)
        if has_converged(X, new_mixture_params, mixture_params, em_tol):
            return new_mixture_params, i
        mixture_params = new_mixture_params
        i += 1
    return new_mixture_params, i


def create_constraints(abs_tol_params: int, min_mix_coef: int) -> tuple[dict, dict, dict]:
    constr = (
        {
            "type": "eq",
            "fun": lambda x: np.sum(x[::4]) - 1,                   # 1. s.t.: sum of mixture coefficients must be 1
        },
        {
            "type": "ineq",                                        # 2. parameters must be positive (mathematically: greater than `abs_tol_params`)
            "fun": lambda x: x
            - abs_tol_params,                     
        },
        {"type": "ineq", "fun": lambda x: x[::4] - min_mix_coef},  # 3. minimum number of points per cluster => reduces singularities
    )                   
    return constr


@njit(fastmath=True, cache=True)
def E_step(X: np.ndarray, mixture_params: np.ndarray) -> np.ndarray:
    """calculates responsibilities "gamma" = mixture component membership\\
         probabilities for each point and component"""
    n_mixture_components = len(mixture_params) // 4  # number of mixture components
    n_datapoints = X.shape[0]                        # number of data points
    gammas = np.empty((n_mixture_components, n_datapoints))
    for k in range(n_mixture_components):            # for each mixture component
        gammas[k, :] = gamma(X, mixture_params, k)
    return gammas


def M_step(
    X: np.ndarray, mixture_params: np.ndarray, gammas: np.ndarray, constraints: dict, minimizer_options: dict
) -> np.ndarray:  # TODO include minimizer_options into function
    """Computes mixture parameters which maximize the log likelihood."""
    minimizer = minimize(
        negative_cdll,
        x0=mixture_params,
        constraints=constraints,
        args=(X, gammas),
        method="SLSQP",
        options={"maxiter": 100},
    )
    new_mixture_params = minimizer.x
    return new_mixture_params


@njit(fastmath=True, cache=True)
def gamma(X: np.ndarray, mixture_component_params: np.ndarray, k: int) -> np.ndarray:
    """Calculates responsibilities "gamma" for a specific mixture component\\
         and all points."""
    nominator = mixture_component_params[4 * k] * pdf(
        X, mixture_component_params[k * 4 : (k + 1) * 4]       # = "mixture coefficient" * "probability density"
    )   
    n_mixture_components = len(mixture_component_params) // 4  # number of mixture components
    n_datapoints = X.shape[0]                                  # number of data points
    denominator = np.zeros(n_datapoints)
    for k_ in range(
        n_mixture_components
    ):  # TODO try refactoring for performance improvements (redundant calculations), but maybe njit optimizes it already
        denominator += mixture_component_params[4 * k_] * pdf( # normalization factor
            X, mixture_component_params[k_ * 4 : (k_ + 1) * 4]
        )  
    return nominator / denominator


@njit(fastmath=True, cache=True)
def negative_cdll(mixture_component_params: np.ndarray, X: np.ndarray, gammas: np.ndarray) -> np.ndarray:
    """Calculates the negative "Complete Data Log Likelihood"."""
    n_mixture_components = len(mixture_component_params) // 4
    n_datapoints = X.shape[0]
    pdfs = np.empty((n_mixture_components, n_datapoints))
    mix_coeffs = mixture_component_params[::4]
    for k in range(n_mixture_components):
        pdfs[k, :] = pdf(X, mixture_component_params[k * 4 : (k + 1) * 4])
    return -np.sum(
        gammas * (np.log(mix_coeffs) + np.log(pdfs.T)).T
    )  # matrix dimensions: n_mixture_components x n_datapoints * (n_mixture_components x n_datapoints).T = n_mixture_components
       # minus sign: because we use a minimization algorithm in M-Step


@njit(fastmath=True, cache=True)
def pdf(X: np.ndarray, mixture_component_params: np.ndarray) -> np.ndarray:
    """Computes for every data point X = (x, y) = (tau, deltaV) the value\\
         of its probability density, given the parameters for a certain cluster."""
    exp_x = X[:, 0]
    exp_mu = mixture_component_params[1]

    norm_x = X[:, 1]
    norm_mu = mixture_component_params[2]
    norm_sigma = mixture_component_params[3]

    exp_pdf = 1 / exp_mu * np.exp(-exp_x / exp_mu)
    norm_pdf = 1 / (norm_sigma * np.sqrt(2 * np.pi)) * np.exp(-1 / 2 * ((norm_x - norm_mu) / norm_sigma) ** 2)
    return exp_pdf * norm_pdf


@njit(fastmath=True, cache=True)
def has_converged(
    X: np.ndarray,
    new_mixture_params: np.ndarray,
    old_mixture_params: np.ndarray,
    em_tol: int,
) -> bool:
    """If the difference in log likelihoods between consecutive iterations is\\
         smaller than `em_tol` the algorithm has converged."""
    ll_new = negative_cdll(new_mixture_params, X, E_step(X, new_mixture_params))
    ll_old = negative_cdll(old_mixture_params, X, E_step(X, old_mixture_params))
    diff = np.abs(ll_new - ll_old)
    if diff < em_tol:
        return True
    else:
        return False


###################################################################################################


@njit(fastmath=True, cache=True)
def compute_loglikelihood(X: np.ndarray, mixture_params: np.ndarray) -> np.ndarray:
    """For use cases outside of the EM Algorithm, e.g. model selection criteria."""
    gammas = E_step(X, mixture_params)
    return -negative_cdll(mixture_params, X, gammas)
