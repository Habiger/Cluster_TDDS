import numpy as np

from dataclasses import dataclass
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, Point

from clustering.initialization.routines.distribution_dataclass import (
    MixtureComponentDistribution,
)
from clustering.initialization.routines.base_classes import Routine, RoutineParameter


@dataclass
class RandomInsideRoutineParameter(
    RoutineParameter
):  # TODO check how to alter the parameter outside of this file; maybe usage with object methods not classmethods
    N_max: int = 2000
    y_scale_range: tuple = (0.1, 2)


class RandomInsideRoutine(Routine):
    assigns_labels = (
        False  # probably not useful; indented to make automated plotting easier
    )

    def __init__(self, **kwargs):
        self.params = RandomInsideRoutineParameter(**kwargs)

    def algorithm(
        self, X: np.ndarray
    ) -> tuple[dict[int, MixtureComponentDistribution], None]:
        x, y = X[:, 0], X[:, 1]
        logx_range = (np.log(np.min(x)), np.log(np.max(x)))
        y_range = (np.min(y), np.max(y))
        polygon = self._get_polygon(X)

        mixture = {}
        for cluster_idx in range(self.params.N_max):
            mixture_component = MixtureComponentDistribution(cluster_idx)
            mixture_component.x.mu, mixture_component.y.mu = self._get_xy(
                logx_range, y_range, polygon
            )
            mixture_component.y.std = np.random.uniform(
                self.params.y_scale_range[0], self.params.y_scale_range[1]
            )
            mixture_component.mix_coef = (
                1  # will be normalized in `ClusterInitialization`
            )
            mixture[cluster_idx] = mixture_component
        return (
            mixture,
            None,
        )  # None indicates that the algorithm returns no labels for the original datapoints; see also self.assign_labels

    def get_sampled_init_params(
        self, possible_starting_values_dict: dict
    ) -> list[dict]:
        cluster_keys = list(possible_starting_values_dict.keys())
        sampled_params = []
        for K in range(
            self.params.N_cluster_min, self.params.N_cluster_max + 1
        ):  # K_max
            for _n in range(0, self.params.N_runs_per_clusternumber):
                idx = np.random.choice(cluster_keys, size=K, replace=False)
                sampled_params.append(
                    {
                        key: val
                        for key, val in possible_starting_values_dict.items()
                        if key in idx
                    }
                )
        return sampled_params

    def get_single_init_param_sample(
        self, possible_starting_values_dict: dict, K: int, *args
    ) -> dict:  # TODO args necessary?
        """used for sampling new starting values for replacing misbehaving starting values (singularities)

        Args:
            * init_params (?): `self.init_params` in 'Cluster_initialization` object \\
            * K (int): number of clusters in this init param set

        Returns:
            dict[int]: new shuffled sample of init_params from init_params
        """
        cluster_keys = list(possible_starting_values_dict.keys())
        idx = np.random.choice(cluster_keys, size=K, replace=False)
        return {
            key: val for key, val in possible_starting_values_dict.items() if key in idx
        }

    @staticmethod
    def _get_polygon(X: np.ndarray) -> Polygon:
        """determines convex hull (polygon) of datapoints given by `df`"""
        hull = ConvexHull(X)
        ext_points = X[hull.vertices]
        polygon = Polygon(ext_points)
        return polygon

    @staticmethod
    def _get_xy(
        logx_range: tuple[float], y_range: tuple[float], polygon: Polygon
    ) -> tuple[float, float]:
        """draws random points until they are inside polygon"""
        x = np.exp(np.random.uniform(*logx_range))
        y = np.random.uniform(*y_range)
        while not polygon.contains(Point([x, y])):
            x = np.exp(np.random.uniform(*logx_range))
            y = np.random.uniform(*y_range)
        return x, y
