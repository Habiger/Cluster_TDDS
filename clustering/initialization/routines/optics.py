import numpy as np
import pandas as pd
import copy
from scipy.special import binom
from dataclasses import dataclass
from sklearn.cluster import OPTICS
from numpy.linalg import inv

from clustering.miscellaneous.parameter_dataclass import nested_dataclass, Parameter
from clustering.initialization.routines.base_classes import RoutineParameter
from clustering.initialization.routines.distribution_dataclass import MixtureComponentDistribution


@dataclass
class OPTICSParameter(Parameter):
    max_eps: float = np.inf
    min_samples: int = 5
    cluster_method: str = "xi"
    metric: str = "mahalanobis"
    algorithm: str = "brute"


@nested_dataclass
class OPTICSRoutineParameter(RoutineParameter):
    optics: OPTICSParameter = OPTICSParameter()


class OPTICSRoutine:  # TODO maybe leave RoutineParameters to Cluster_init and pass it as arguments
    name = "OPTICS"
    assigns_labels = True

    def __init__(self, **kwargs):
        self.params = OPTICSRoutineParameter(**kwargs)

    ##### Run Optics algorithm and derive initial parameters #########################################
    def algorithm(self, X: np.ndarray) -> tuple[dict, np.ndarray]:
        cov_mat = np.cov(X, rowvar=False)
        clusterer = OPTICS(**self.params.optics.get_dict(), metric_params={"VI": inv(cov_mat)}).fit(X)  # , xi=0.05
        df = pd.DataFrame.from_dict({"x": list(X[:, 0]), "y": list(X[:, 1]), "init_cluster": clusterer.labels_})
        init_params = self._calculate_init_params(df)
        return init_params, clusterer.labels_

    def _calculate_mixing_coefficients(self, df: pd.DataFrame) -> dict[int, float]:
        """Calculates the mixing coefficients as percentage of points belonging to a cluster.
        `Note:` after sampling the mixing coeffcients get recalculated
        """
        df_agg = df.loc[df["init_cluster"] != -1, :].groupby(["init_cluster"]).agg({"x": "count"}).reset_index()
        df_agg["mix_coef"] = df_agg.loc[:, "x"] / sum(df_agg.x)
        mix_coefs = {
            cl: df_agg.loc[df_agg.init_cluster == cl, "mix_coef"].values[0] for cl in df_agg.init_cluster.unique()
        }
        return mix_coefs

    def _calculate_init_params(self, df: pd.DataFrame) -> dict[int, MixtureComponentDistribution]:
        mix_coefs = self._calculate_mixing_coefficients(df)
        init_params = {}
        for cl in df.init_cluster.unique():
            if cl != -1:  # -1 is noise according to OPTICS 
                df_cluster = df.loc[df.init_cluster == cl, ["x", "y"]]
                mixture_component = MixtureComponentDistribution(cl)
                mixture_component.x.mu = np.mean(df_cluster.x)
                mixture_component.y.mu = np.mean(df_cluster.y)
                mixture_component.y.std = np.std(
                    df.y
                )  # TODO currently quick fix; minimize algorithm didn´t work wit low variance
                mixture_component.n_points = len(df_cluster.x)
                mixture_component.mix_coef = mix_coefs[cl]
                init_params[cl] = mixture_component
        return init_params

    ##### Sample initial parameters and correct mixing coefficients ################################

    def get_sampled_init_params(self, possible_starting_values):
        """after getting candidates for cluster centroids (= init_params),
        we need to sample them to get candidates for a defined number of clusters (=K_)
        """
        cluster_numbers, K = possible_starting_values.keys(), len(possible_starting_values)
        selected_init_clusters = []
        for K_ in range(self.params.N_cluster_min, min(K, self.params.N_cluster_max) + 1):
            n_range = int(
                min(binom(K, K_), self.params.N_runs_per_clusternumber)
            )  # if possible max_samples, else max possible amount (limited by binomial coefficient)
            for _n in range(n_range):
                pars = set(np.random.choice(list(cluster_numbers), size=K_, replace=False))
                while pars in selected_init_clusters:  # ensures only unique samples
                    pars = set(np.random.choice(list(cluster_numbers), size=K_, replace=False))
                selected_init_clusters.append(pars)
        selected_init_clusters = [list(set_) for set_ in selected_init_clusters]
        selected_init_params = [
            {cl: copy.deepcopy(possible_starting_values[cl]) for cl in l} for l in selected_init_clusters
        ]
        return selected_init_params

    def get_single_init_param_sample(self, possible_starting_values, K: int, sampled_starting_values_dict: dict):
        """sample new init params; it is possible to get the same as has been used before #TODO ?"""
        previously_selected_combinations = [set(init_param.keys()) for init_param in sampled_starting_values_dict]
        possible_starting_values_keys = list(possible_starting_values.keys())
        idxs = np.random.choice(possible_starting_values_keys, size=K, replace=False)
        # search for new unqiue set of starting values
        i, new_set_of_starting_values_found = 0, set(idxs) not in previously_selected_combinations
        while i < 100 and not new_set_of_starting_values_found:
            idxs = np.random.choice(possible_starting_values_keys, size=K, replace=False)
            new_set_of_starting_values_found = set(idxs) not in previously_selected_combinations
            i += 1
        if new_set_of_starting_values_found:
            return {cl: possible_starting_values[cl] for cl in idxs}
        else:
            return None  # returns None if no new unique set of starting values has been found
