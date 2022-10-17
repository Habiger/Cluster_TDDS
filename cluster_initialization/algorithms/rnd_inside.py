from nptyping import Bool
import pandas as pd
import numpy as np
from typing import Dict
from random import choices

from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, Point

from cluster_initialization.parameter_dataclass import Params




class RANDOM_inside_routine:
    assigns_labels = False
    N_max = 2000

    @classmethod
    def algorithm(cls, df: pd.DataFrame) -> Dict[int, Params]:
        logx_min, logx_max = np.log(np.min(df.x)), np.log(np.max(df.x))
        y_min, y_max = np.min(df.y), np.max(df.y)
        polygon = cls._get_polygon(df)
        
        params_dict = {}
        for cl in range(cls.N_max):
            params = Params(cl)
            params.x.mu, params.y.mu = cls._get_xy(logx_min, logx_max, y_min, y_max, polygon)
            params.y.std =  np.random.uniform(0.1, 2)
            params.mix_coef = 1
            params_dict[cl] = params
        return params_dict, None   # None indicates that the algorithm returns no labels for the original datapoints

    @classmethod
    def get_sampled_init_params(cls, _df, init_params, K_max, N_runs_per_clusternumber):
        cluster_keys = list(init_params.keys())
        sampled_params = []
        for K in range(1, K_max+1):
            for _n in range(0, N_runs_per_clusternumber):
                idx = np.random.choice(cluster_keys, size=K, replace=False)
                sampled_params.append({key: val for key, val in init_params.items() if key in idx})
        return sampled_params

    @classmethod
    def get_single_init_param_sample(cls, init_params, K):
        """used for sampling new starting values for replacing misbehaving starting values (singularities)

        Args:
            * init_params (?): `self.init_params` in 'Cluster_initialization` object \\
            * K (int): number of clusters in this init param set

        Returns:
            dict[int]: new shuffled sample of init_params from init_params
        """
        cluster_keys = list(init_params.keys())
        idx = np.random.choice(cluster_keys, size=K, replace=False)
        return {key: val for key, val in init_params.items() if key in idx}

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