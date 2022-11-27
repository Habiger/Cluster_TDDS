from nptyping import Bool
import pandas as pd
import numpy as np
from typing import Dict
from random import choices

from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, Point

from cluster_initialization.parameter_dataclass import Params


class Parameter:
    N_max = 2000
    y_scale_range = (0.1, 2)

class RANDOM_inside_routine(Parameter):
    assigns_labels = False  # probably not useful; indented to make automated plotting easier

    @classmethod
    def algorithm(cls, df: pd.DataFrame) -> Dict[int, Params]:
        logx_min, logx_max = np.log(np.min(df.x)), np.log(np.max(df.x))
        y_min, y_max = np.min(df.y), np.max(df.y)
        polygon = cls._get_polygon(df)
        
        params_dict = {}
        for cl in range(cls.N_max):
            params = Params(cl)
            params.x.mu, params.y.mu = cls._get_xy(logx_min, logx_max, y_min, y_max, polygon)
            params.y.std =  np.random.uniform(cls.y_scale_range[0], cls.y_scale_range[1])
            params.mix_coef = 1           # will be normalized in `Cluster_Initialization`
            params_dict[cl] = params
        return params_dict, None   # None indicates that the algorithm returns no labels for the original datapoints

    @classmethod
    def get_sampled_init_params(cls, possible_starting_values_dict, K_max, N_runs_per_clusternumber):
        cluster_keys = list(possible_starting_values_dict.keys())
        sampled_params = []
        for K in range(1, K_max+1):
            for _n in range(0, N_runs_per_clusternumber):
                idx = np.random.choice(cluster_keys, size=K, replace=False)
                sampled_params.append({key: val for key, val in possible_starting_values_dict.items() if key in idx})
        return sampled_params

    @classmethod
    def get_single_init_param_sample(cls, possible_starting_values_dict, K, *args):
        """used for sampling new starting values for replacing misbehaving starting values (singularities)

        Args:
            * init_params (?): `self.init_params` in 'Cluster_initialization` object \\
            * K (int): number of clusters in this init param set

        Returns:
            dict[int]: new shuffled sample of init_params from init_params
        """
        cluster_keys = list(possible_starting_values_dict.keys())
        idx = np.random.choice(cluster_keys, size=K, replace=False)
        return {key: val for key, val in possible_starting_values_dict.items() if key in idx}

    @classmethod
    def _get_polygon(cls, df) -> Polygon:
        X = df[["x", "y"]].to_numpy()
        hull = ConvexHull(X)
        ext_points = X[hull.vertices]
        polygon = Polygon(ext_points)
        return polygon

    @classmethod
    def _get_xy(cls, logx_min, logx_max, y_min, y_max, polygon):
        x = np.exp(np.random.uniform(logx_min, logx_max))
        y = np.random.uniform(y_min, y_max)
        while not polygon.contains(Point([x, y])):
            x = np.exp(np.random.uniform(logx_min, logx_max))
            y = np.random.uniform(y_min, y_max)
        return x, y