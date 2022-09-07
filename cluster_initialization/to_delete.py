import copy
from random import choices
from scipy.special import binom
from sklearn.metrics import silhouette_samples

def get_silhouette_weights(df):
    df = df[df.init_cluster != -1].copy()
    df["sil_score"] = silhouette_samples(df[["x", "y"]].values, labels=df.init_cluster, metric="mahalanobis")
    df_score = df.groupby(["init_cluster"]).agg({"sil_score": "mean"}).sort_values(by="sil_score", ascending=False).reset_index()
    weights = {}
    for _idx, row in df_score.iterrows():
        cluster = int(row["init_cluster"])
        weights[cluster] =  row["sil_score"]
    return weights

def correct_mix_coef(selected_init_params):
    selected_init_params_new = []
    for initialization in selected_init_params:
        denominator = sum([pars.mix_coef for cl, pars in initialization.items()])
        corrected_param_dict = {}
        for cl, pars in initialization.items():
            corrected_param_dict[cl] = copy.deepcopy(pars)
            corrected_param_dict[cl].mix_coef = pars.mix_coef/denominator
        selected_init_params_new.append(corrected_param_dict)
    return selected_init_params_new



def get_sampled_init_params(df, init_params, max_no_of_samples_per_cluster_number=10, K_max=15):
    """after getting candidates for cluster centroids (= init_params), 
    we need to sample them to get candidates for a defined number of clusters (=K_)
    """
    if len(init_params.keys()) != 1:
        weights = get_silhouette_weights(df)   # weight candidates according to their silhouette score
    else:
        weights = {0: 1.}

    cluster_numbers = init_params.keys()

    weights = [weights[cl]+1 for cl in cluster_numbers] # silouette score goes from -1, 1 
    K = len(init_params)
    selected_init_clusters = []
    for K_ in range(1, min(K, K_max)+1):

        n_range =  int(min(binom(K, K_), max_no_of_samples_per_cluster_number)) # if possible max_samples, else max feasable amount
        for _n in range(n_range):
            pars = set(choices(list(cluster_numbers), k=K_))  #, weights=weights
            while pars in selected_init_clusters:   # only unique 
                pars = set(choices(list(cluster_numbers), k=K_))  #, weights=weights
            selected_init_clusters.append(pars)
    selected_init_clusters = [list(set_) for set_ in selected_init_clusters]
    selected_init_params = [{cl: copy.deepcopy(init_params[cl]) for cl in l} for l in selected_init_clusters]
    return correct_mix_coef(selected_init_params)
