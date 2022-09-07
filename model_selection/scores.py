import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from em_algorithm.em_funcs import cdll2, E_step
from numba import njit
import copy

def compute_AIC(X: np.ndarray, params: np.ndarray) -> float:
    d = len(params)
    ll = compute_ll(X, params)
    return -ll + d

def compute_BIC(X: np.ndarray, params: np.ndarray) -> float:
    N, d = X.shape[0], len(params)
    ll = compute_ll(X, params)
    return -ll + 1/2 * d * np.log(N)

def compute_KIC(X: np.ndarray, params: np.ndarray) -> float:
    N, d = X.shape[0], len(params)
    ll = compute_ll(X, params)
    return -2*ll + 3 * (d  + 1)

def compute_ll(X: np.ndarray, params) -> float:
    N, d = X.shape[0], len(params)
    gammas = E_step(X, params)
    ll = -cdll2(params, X, gammas) # the function returns the negative ll
    return ll

def compute_silhouette(X: np.ndarray, params: np.ndarray) -> float:
    gammas = E_step(X, params)
    labels = np.argmax(gammas, axis=0)
    if len(np.unique(labels) ) == 1: # silouette score only defined for d > 1
        return 0.
    else:
        return silhouette_score(X, labels, metric = "mahalanobis")

def compute_MDL(X: np.ndarray, params: np.ndarray) -> float:
    N, d, K = X.shape[0], len(params), len(params)//4
    ll = compute_ll(X, params)
    m = (K-1) + K * 3  # model free parameters: first-> [sum(mix_coefs) = 1] => d-1    +    3 per cluster ( 2x mean, 1x std)
    return -ll + 1/2 * m * np.log(N)

def compute_CH(X: np.ndarray, params: np.ndarray):
    gammas = E_step(X, params)
    labels = np.argmax(gammas, axis=0)
    if len(np.unique(labels)) == 1:
        return 0.
    else:
        return calinski_harabasz_score(X, labels)

def compute_CH_scaled(X: np.ndarray, params: np.ndarray):
    gammas = E_step(X, params)
    labels = np.argmax(gammas, axis=0)
    X_scaled = copy.deepcopy(X)
    X_scaled = X_scaled / X_scaled.std(axis=0)

    if len(np.unique(labels)) == 1:
        return 0.
    else:
        return calinski_harabasz_score(X_scaled, labels)

scores = {
    "ll": {"func": compute_ll, "rank_params": {"ascending": False}},
    "AIC": {"func": compute_AIC, "rank_params": {"ascending": True}},
    "BIC": {"func": compute_BIC, "rank_params": {"ascending": True}},
    #"KIC": {"func": compute_KIC, "rank_params": {"ascending": True}},
    "silhouette": {"func": compute_silhouette, "rank_params": {"ascending": False}}, # , "na_option": "bottom" should´nt be necessary anymore because
    "HC": {"func": compute_CH, "rank_params": {"ascending": False}},
    "HC_scaled": {"func": compute_CH_scaled, "rank_params": {"ascending": False}},
    #"MDL": {"func": compute_MDL, "rank_params": {"ascending": True}}
}