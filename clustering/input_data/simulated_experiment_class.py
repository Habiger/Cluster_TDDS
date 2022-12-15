import random
import os
import json
import numpy as np
import pandas as pd

from dataclasses import dataclass

from clustering.miscellaneous.parameter_dataclass import Parameter, nested_dataclass
from clustering.input_data.experiment_baseclass import Experiment

@dataclass
class ClusterParameter(Parameter):
    cluster_size_range: tuple = (1, 100)
    exp_scale_range: tuple = (10**-3, 1)
    norm_loc_range: tuple =  (0.5, 7)
    norm_scale_range: tuple = (0.1, 1)


@nested_dataclass
class SimulatedExperimentParameter(Parameter): 
    store_data: bool = False
    path_to_store_data: str = None
    min_datapoints: int = 5                # OPTICS Initialization Routine needs at least 5 points because of min_samples parameter
    cluster_number_range: tuple = (1, 10)

    cluster: ClusterParameter = ClusterParameter()


class SimulatedExperiment(Experiment):
    def __init__(self, id: str, **kwargs):
        self.id = id
        self.params = SimulatedExperimentParameter(**kwargs)
        self.n_cluster: int = None           # will be set by `_simulate_experiment()`
        self.clusters: list[Cluster] = None             # will be set by `_simulate_experiment()`
        self._simulate_experiment()

        self.df = self._get_df()
        self.X = self.df[["x", "y"]].to_numpy(copy=True)
        self.true_labels = self.df["cluster"]
        if self.params.store_data:
            self._save_data()
       
       
    def _simulate_experiment(self) -> None:
        """simulates experimental data with at least `min_datapoints`
        """
        n_datapoints = 0
        while n_datapoints < self.params.min_datapoints:
            self.n_cluster = self._sample_cluster_number()
            self.clusters = [Cluster(self.params.cluster) for i in range(self.n_cluster)]
            n_datapoints = len(self._get_df().index)

    def _sample_cluster_number(self) -> int:
        possible_cluster_numbers = [i for i in range(self.params.cluster_number_range[0], self.params.cluster_number_range[1]+1)]
        return random.choice(possible_cluster_numbers)

    def _get_df(self) -> pd.DataFrame:
        df_list = []
        for i, cluster in enumerate(self.clusters):
            df = cluster.coordinates.copy()
            df["cluster"] = i
            df_list.append(df)
        return pd.concat(df_list)

    def get_distr_params(self) -> dict:
        """Returns true mixture distributions.
        """
        distr_params = {
            "cluster": [i for i in range(self.n_cluster)],
            "clustersize": [cluster.size for cluster in self.clusters],
            "x_scale": [cluster.exp_distribution["scale"] for cluster in self.clusters],
            "y_loc": [cluster.norm_distribution["loc"] for cluster in self.clusters],
            "y_scale": [cluster.norm_distribution["scale"] for cluster in self.clusters]
        }
        return distr_params

    def _save_data(self) -> None:
        path = os.path.join(self.params.path_to_store_data, f"dataset_{self.id}")
        if not os.path.exists(path):
            os.makedirs(path)
        # save coordinates and labels
        df_filepath = os.path.join(path, "df.csv")
        self.df.to_csv(df_filepath, index=False)
        # save parameters
        self.params.save(path)
        # save true distribution parameters
        distr_params_filepath = os.path.join(path, "distribution_params.json")
        with open(distr_params_filepath, "w") as fp:
            json.dump(self.get_distr_params(), fp, indent=4)

    @classmethod
    def from_parametric_bootstrap(cls, df: pd.DataFrame, dataset_id):    #TODO seperate into own class
        """draws parametric bootstrap sample from given parameters in `df`
        """
        obj = cls.__new__(cls)              # initialize instance without calling self.__init__()
        obj.n_cluster = len(df.prediction_cluster.unique())
        obj.clusters = [Cluster.from_parametric_bootstrap(df.loc[i]) for i in df.index]
        obj.df = obj._get_df()
        obj.X = obj.df[["x", "y"]].copy().to_numpy()
        obj.id = dataset_id
        return obj


class Cluster:
    def __init__(self, params: ClusterParameter):
        self.params = params
        self.size = self._sample_size()
        self.exp_distribution = self._sample_exp_distr_params()  #TODO maybe use mixture component class from cluster_init??
        self.norm_distribution = self._sample_norm_distr_params()
        self.coordinates = self._sample_coordinates()

    def _sample_size(self) -> int:
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
    def from_parametric_bootstrap(cls, row: pd.Series):
        """returns instance of `Cluster` filled with distribution parameters given by parametric bootstrap
        """
        obj = cls.__new__(cls)                 # initialize instance without calling self.__init__()
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






