import random

import numpy as np
import pandas as pd

from miscellaneous.parameter import Parameter
from dataclasses import dataclass

@dataclass
class ExperimentParameter(Parameter): #TODO Refactor: split cluster parameter as own dataclass
    """default simulation parameters
    """
    # Parameter for Experiment
    cluster_number_range: tuple = (1, 5)
    min_datapoints: int = 5 # OPTICS Initialization Routine needs at least 5 points because of min_samples parameter

    # Parameter for individual Clusters
    cluster_size_range: tuple = (1, 100)
    exp_scale_range: tuple = (10**-3, 1)
    norm_loc_range: tuple =  (0.5, 7)
    norm_scale_range: tuple = (0.1, 1)

    
class Cluster:
    def __init__(self, params):
        self.params = params
        self.size = self._sample_size()
        self.exp_distribution = self._sample_exp_distr_params()
        self.norm_distribution = self._sample_norm_distr_params()
        self.coordinates = self._sample_coordinates()

    def _sample_size(self):
        possible_size_numbers = [i for i in range(self.params.cluster_size_range[0], self.params.cluster_size_range[1]+1)]
        return random.choice(possible_size_numbers)

    def _sample_exp_distr_params(self):
        params = {
            "scale": np.exp(np.random.uniform(
                low = np.log(self.params.exp_scale_range[0]), 
                high = np.log(self.params.exp_scale_range[1])
                )), 
            "size": self.size
            }
        return params

    def _sample_norm_distr_params(self):
        params = {
            "loc": np.random.uniform(
                low=self.params.norm_loc_range[0], 
                high=self.params.norm_loc_range[1]
                ), 
            "scale": np.random.uniform(
                low=self.params.norm_scale_range[0], 
                high=self.params.norm_scale_range[1]
                ), 
            "size": self.size
            }
        return params

    def _sample_coordinates(self):
        x, y = np.random.exponential(**self.exp_distribution), np.random.normal(**self.norm_distribution)
        return pd.DataFrame({"x": x, "y": y})


    @classmethod
    def from_parametric_bootstrap(cls, row):
        obj = cls.__new__(cls)
        obj.size = row.n_points
        obj.exp_distribution = {
            "scale": row.x_mean, 
            "size": obj.size
            }
        obj.norm_distribution = {
            "loc": row.y_mean, 
            "scale": row.y_std, 
            "size": obj.size
            }
        obj.coordinates = obj._sample_coordinates()
        return obj


class Experiment:
    def __init__(self, **kwargs):
        self.params = ExperimentParameter(**kwargs)
        self.n_cluster = None           # will be set by `_simulate_experiment()`
        self.cluster = None             # will be set by `_simulate_experiment()`
        self._simulate_experiment()

        self.df = self._get_df()
        self.X = self.df[["x", "y"]].to_numpy(copy=True)

    def _simulate_experiment(self):
        """simulates experimental data with at least `min_datapoints`
        """
        n_datapoints = 0
        while n_datapoints < self.params.min_datapoints:
            self.n_cluster = self._sample_cluster_number()
            self.cluster = [Cluster(self.params) for i in range(self.n_cluster)]
            n_datapoints = len(self._get_df().index)

    def _sample_cluster_number(self):
        possible_cluster_numbers = [i for i in range(self.params.cluster_number_range[0], self.params.cluster_number_range[1]+1)]
        return random.choice(possible_cluster_numbers)

    def _get_df(self):
        df_list = []
        for i, cluster in enumerate(self.cluster):
            df = cluster.coordinates.copy()
            df["cluster"] = i
            df_list.append(df)
        return pd.concat(df_list)

    def get_distr_params_df(self):
        """`ToDo:` decide wether to store as metadata in hdf5 or as dataframe
        """
        dict_df = {
            "cluster": [i for i in range(self.n_cluster)],
            "clustersize": [cluster.size for cluster in self.cluster],
            "x_scale": [cluster.exp_distribution["scale"] for cluster in self.cluster],
            "y_loc": [cluster.norm_distribution["loc"] for cluster in self.cluster],
            "y_scale": [cluster.norm_distribution["scale"] for cluster in self.cluster]
        }
        return pd.DataFrame.from_dict(dict_df)


    @classmethod
    def from_parametric_bootstrap(cls, df: pd.DataFrame):
        """draws parametric bootstrap sample from given parameters in `df`
        """
        obj = cls.__new__(cls)              # does not call __init__
        obj.n_cluster = len(df.prediction_cluster.unique())
        obj.cluster = [Cluster.from_parametric_bootstrap(df.loc[i]) for i in df.index]
        obj.df = obj._get_df()
        obj.X = obj.df[["x", "y"]].copy().to_numpy()
        return obj


