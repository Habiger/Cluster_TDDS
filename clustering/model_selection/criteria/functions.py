import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from clustering.em.em_algorithm import E_step, compute_loglikelihood
import copy

def compute_AIC(X: np.ndarray, params: np.ndarray) -> float:
    d = len(params)
    ll = compute_loglikelihood(X, params)
    return -ll + d

def compute_BIC(X: np.ndarray, params: np.ndarray) -> float:
    N, d = X.shape[0], len(params)
    ll = compute_loglikelihood(X, params)
    return -ll + 1/2 * d * np.log(N)

def compute_KIC(X: np.ndarray, params: np.ndarray) -> float:
    N, d = X.shape[0], len(params)
    ll = compute_loglikelihood(X, params)
    return -2*ll + 3 * (d  + 1)

def compute_silhouette(X: np.ndarray, params: np.ndarray) -> float:
    gammas = E_step(X, params)
    labels = np.argmax(gammas, axis=0)
    if len(np.unique(labels) ) == 1: # silouette score only defined for d > 1
        return 0.
    else:
        return silhouette_score(X, labels, metric = "mahalanobis")

def compute_MDL(X: np.ndarray, params: np.ndarray) -> float:
    """`Note:` identical to BIC-score
    """
    N, d, K = X.shape[0], len(params), len(params)//4
    ll = compute_loglikelihood(X, params)
    m = (K-1) + K * 3  # model free parameters: first-> [sum(mix_coefs) = 1] => d-1    +    3 per cluster ( 2x mean, 1x std)
    return -ll + 1/2 * m * np.log(N)

def compute_CH(X: np.ndarray, params: np.ndarray) -> float:
    gammas = E_step(X, params)
    labels = np.argmax(gammas, axis=0)
    if len(np.unique(labels)) == 1:
        return 0.
    else:
        return calinski_harabasz_score(X, labels)

def compute_MML(X, params) -> float:
    N, d, K, d_component = X.shape[0], len(params), len(params)//4, 4
    ll = compute_loglikelihood(X, params)
    return d_component/2 * np.sum(np.log(N*params[::4]/12))  + K/2 * np.log(N/12) + K*(d_component+1)/2 - ll


criteria_dict = {
    "ll": {"func": compute_loglikelihood, "rank_params": {"ascending": False}, "use_in_total_score": True},
    "AIC": {"func": compute_AIC, "rank_params": {"ascending": True}, "use_in_total_score": True},
    "BIC": {"func": compute_BIC, "rank_params": {"ascending": True}, "use_in_total_score": True},
    #"KIC": {"func": compute_KIC, "rank_params": {"ascending": True}},
    "MML": {"func": compute_MML, "rank_params": {"ascending": True}, "use_in_total_score": True},
    "silhouette": {"func": compute_silhouette, "rank_params": {"ascending": False}, "use_in_total_score": True}, # , "na_option": "bottom" should´nt be necessary anymore because
    "CH": {"func": compute_CH, "rank_params": {"ascending": False}, "use_in_total_score": True},
    #"CH_scaled": {"func": compute_CH_scaled, "rank_params": {"ascending": False}, "use_in_total_score": False},
    #"CH_scaled2": {"func": compute_CH_scaled2, "rank_params": {"ascending": False}, "use_in_total_score": True},
    
    #"MDL": {"func": compute_MDL, "rank_params": {"ascending": True}}
}



############### experimental ######################################################################

def compute_CH_scaled2(X: np.ndarray, params: np.ndarray):
    gammas = E_step(X, params) #TODO use PCA and to the proper mahlanobis transform
    labels = np.argmax(gammas, axis=0)
    X_scaled = copy.deepcopy(X)
    X_scaled[:,0] = np.log(X_scaled[:,0])
    X_scaled = X_scaled / X_scaled.std(axis=0)

    if len(np.unique(labels)) == 1:
        return 0.
    else:
        return calinski_harabasz_score(X_scaled, labels)

def compute_CH_scaled(X: np.ndarray, params: np.ndarray):
    gammas = E_step(X, params) #TODO use PCA and to the proper mahlanobis transform
    labels = np.argmax(gammas, axis=0)
    X_scaled = copy.deepcopy(X)
    X_scaled = X_scaled / X_scaled.std(axis=0)

    if len(np.unique(labels)) == 1:
        return 0.
    else:
        return calinski_harabasz_score(X_scaled, labels)
