import numpy as np
import pandas as pd

from copy import deepcopy
from scipy.special import binom
from sklearn.metrics import silhouette_samples

from clustering.initialization.routines.distribution_dataclass import MixtureComponentDistribution
from clustering.initialization.routines.optics import OPTICSRoutine


class OPTICSWeightedRoutine(OPTICSRoutine):
    """currently not working
    """
    name = "OPTICS_weighted"
    assigns_labels = True
    
    def __init__(self):
        super().__init__()

    ##### Sample initial parameters and correct mixing coefficients ################################

    @classmethod
    def get_sampled_init_params(cls, df, init_params, max_no_of_samples_per_cluster_number=10, K_max=15):
        """after getting candidates for cluster centroids (= init_params), 
        we need to sample them to get candidates for a defined number of clusters (=K_)
        """
        if len(init_params.keys()) != 1:
            weights = cls._get_silhouette_weights(df)   # weight candidates according to their silhouette score
        else:
            weights = {0: 1.}

        cluster_numbers, K = init_params.keys(), len(init_params)
        weights_demoninator = sum([weights[cl]+1 for cl in cluster_numbers])
        weights = [(weights[cl]+1)/weights_demoninator for cl in cluster_numbers] # silouette score goes from -1, 1  
        selected_init_clusters = []
        for K_ in range(1, min(K, K_max)+1):
            n_range =  int(min(binom(K, K_), max_no_of_samples_per_cluster_number)) # if possible max_samples, else max possible amount (limited by binomial coefficient)
            for _n in range(n_range):
                pars = set(np.random.choice(list(cluster_numbers), size=K_, p=weights, replace=False))  #
                while pars in selected_init_clusters:   # ensures only unique samples
                    pars = set(np.random.choice(list(cluster_numbers), size=K_, p=weights, replace=False))  #
                selected_init_clusters.append(pars)
        selected_init_clusters = [list(set_) for set_ in selected_init_clusters]
        selected_init_params = [{cl: deepcopy(init_params[cl]) for cl in l} for l in selected_init_clusters]
        return selected_init_params

    @classmethod
    def _get_silhouette_weights(cls, df):
        df = df[df.init_cluster != -1].copy()
        df["sil_score"] = silhouette_samples(df[["x", "y"]].values, labels=df.init_cluster, metric="mahalanobis")
        df_score = df.groupby(["init_cluster"]).agg({"sil_score": "mean"}).sort_values(by="sil_score", ascending=False).reset_index()
        weights = {}
        for _idx, row in df_score.iterrows():
            cluster = int(row["init_cluster"])
            weights[cluster] =  row["sil_score"]
        return weights




